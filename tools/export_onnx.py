#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

import argparse
import os

import cv2
from loguru import logger

import torch
from torch import nn

from yolox.exp import get_exp
from yolox.models.network_blocks import SiLU
from yolox.utils import replace_module
from pytorch_quantization import nn as quant_nn
from pytorch_quantization import calib
from pytorch_quantization.tensor_quant import QuantDescriptor
from pytorch_quantization import quant_modules
from yolox.data.data_augment import preproc as preprocess
import os
import random


def collect_stats(model, sample_dir, sample_numbers):
    # list all imgs
    imgnames = os.listdir(sample_dir)
    imgnames = [os.path.join(sample_dir, imgname) for imgname in imgnames]
    random.shuffle(imgnames)

    print("start to collect quantization parameters")
    for name, module in model.named_modules():
        if isinstance(module, quant_nn.TensorQuantizer):
            if module._calibrator is not None:
                module.disable_quant()
                module.enable_calib()
            else:
                module.disable()
    for i in range(sample_numbers):
        print("Collect quantization parametes with {}th/{} sample".format(i+1, sample_numbers))
        imgname = imgnames[i]
        img = cv2.imread(imgname)
        img, _ = preprocess(img, (640, 640))
        img = torch.from_numpy(img).unsqueeze(0).float().cuda()
        with torch.cuda.amp.autocast(False):
            model(img)

    # Disable calibrators
    for name, module in model.named_modules():
        if isinstance(module, quant_nn.TensorQuantizer):
            if module._calibrator is not None:
                module.enable_quant()
                module.disable_calib()
            else:
                module.enable()


def compute_amax(model, **kwargs):
    print("start to compute quantization parameters")
    # Load calib result
    for name, module in model.named_modules():
        if isinstance(module, quant_nn.TensorQuantizer):
            if module._calibrator is not None:
                if isinstance(module._calibrator, calib.MaxCalibrator):
                    module.load_calib_amax()
                else:
                    module.load_calib_amax(**kwargs)
            print(F"{name:40}: {module}")
    model.cuda()


def make_parser():
    parser = argparse.ArgumentParser("YOLOX onnx deploy")
    parser.add_argument(
        "--output-name", type=str, default="yolox.onnx", help="output name of models"
    )
    parser.add_argument(
        "--input", default="images", type=str, help="input node name of onnx model"
    )
    parser.add_argument(
        "--output", default="output", type=str, help="output node name of onnx model"
    )
    parser.add_argument(
        "-o", "--opset", default=11, type=int, help="onnx opset version"
    )
    parser.add_argument("--batch-size", type=int, default=1, help="batch size")
    parser.add_argument(
        "--dynamic", action="store_true", help="whether the input shape should be dynamic or not"
    )
    parser.add_argument("--no-onnxsim", action="store_true", help="use onnxsim or not")
    parser.add_argument(
        "-f",
        "--exp_file",
        default=None,
        type=str,
        help="expriment description file",
    )
    parser.add_argument("-expn", "--experiment-name", type=str, default=None)
    parser.add_argument("-n", "--name", type=str, default=None, help="model name")
    parser.add_argument("-c", "--ckpt", default=None, type=str, help="ckpt path")
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    parser.add_argument(
        "--decode_in_inference",
        action="store_true",
        help="decode in inference or not"
    )
    parser.add_argument(
        '--quantization',
        action='store_true',
        help='Do post training quantization before exporting.'
    )
    parser.add_argument('--represent_dataset', '-rd', type=str, help='Path of the represent dataset used for Post Training Quantization.')
    return parser


@logger.catch
def main():
    args = make_parser().parse_args()
    logger.info("args value: {}".format(args))
    exp = get_exp(args.exp_file, args.name)
    exp.merge(args.opts)

    if not args.experiment_name:
        args.experiment_name = exp.exp_name

    if args.quantization: # enable post training quantization
        quant_nn.TensorQuantizer.use_fb_fake_quant = True
        quant_modules.initialize()
        quant_desc_input = QuantDescriptor(calib_method='histogram')
        quant_nn.QuantConv2d.set_default_quant_desc_input(quant_desc_input)
        quant_nn.QuantLinear.set_default_quant_desc_input(quant_desc_input)
    model = exp.get_model()
    if args.ckpt is None:
        file_name = os.path.join(exp.output_dir, args.experiment_name)
        ckpt_file = os.path.join(file_name, "best_ckpt.pth")
    else:
        ckpt_file = args.ckpt

    # load the model state dict
    ckpt = torch.load(ckpt_file, map_location="cpu")
    model.eval()
    if "model" in ckpt:
        ckpt = ckpt["model"]
    model.load_state_dict(ckpt)
    model = replace_module(model, nn.SiLU, SiLU)
    model.head.decode_in_inference = args.decode_in_inference
    model.cuda()

    # Post Training Quantization
    if args.quantization:
        with torch.no_grad():
            collect_stats(model, args.represent_dataset, 10000)
            compute_amax(model, method="percentile", percentile=99.99)

    logger.info("loading checkpoint done.")
    dummy_input = torch.randn(args.batch_size, 3, exp.test_size[0], exp.test_size[1]).cuda()
    model.train()
    torch.onnx._export(
        model,
        dummy_input,
        args.output_name,
        input_names=[args.input],
        output_names=[args.output],
        dynamic_axes={args.input: {0: 'batch'},
                      args.output: {0: 'batch'}} if args.dynamic else None,
        opset_version=13,
        verbose=False,
        training=torch.onnx.TrainingMode.EVAL
    )
    logger.info("generated onnx model named {}".format(args.output_name))

    if not args.no_onnxsim:
        import onnx

        from onnxsim import simplify

        input_shapes = {args.input: list(dummy_input.shape)} if args.dynamic else None

        # use onnxsimplify to reduce reduent model.
        onnx_model = onnx.load(args.output_name)
        model_simp, check = simplify(onnx_model,
                                     dynamic_input_shape=args.dynamic,
                                     input_shapes=input_shapes)
        assert check, "Simplified ONNX model could not be validated"
        onnx.save(model_simp, args.output_name)
        logger.info("generated simplified onnx model named {}".format(args.output_name))


if __name__ == "__main__":
    main()
