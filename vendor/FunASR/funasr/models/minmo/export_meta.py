#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
# Copyright FunASR (https://github.com/alibaba-damo-academy/FunASR). All Rights Reserved.
#  MIT License  (https://opensource.org/licenses/MIT)

import types
import torch
from funasr.register import tables


class model_rebuild(torch.nn.Module):
    def __init__(self, ):
        super().__init__()
        pass

def export_rebuild_model(model, **kwargs):
    model.device = kwargs.get("device")
    is_onnx = kwargs.get("type", "onnx") == "onnx"
    encoder_class = tables.encoder_classes.get(kwargs["audio_encoder"] + "Export")
    # model.audio_encoder = encoder_class(model.audio_encoder, onnx=is_onnx)
    # del model.llm

    model_new = model_rebuild()
    model_new.audio_encoder = encoder_class(model.audio_encoder, onnx=is_onnx)
    model_new.audio_adaptor = model.audio_adaptor

    model_new.forward = types.MethodType(export_forward, model_new)
    model_new.export_dummy_inputs = types.MethodType(export_dummy_inputs, model_new)
    model_new.export_input_names = types.MethodType(export_input_names, model_new)
    
    if hasattr(model, "ctc_decoder"):
        model_new.ctc_decoder = model.ctc_decoder
        model_new.ctc = model.ctc
        model_new.export_output_names = types.MethodType(export_output_names_ctc, model_new)
        model_new.export_dynamic_axes = types.MethodType(export_dynamic_axes_ctc, model_new)
    else:
        model_new.export_output_names = types.MethodType(export_output_names, model_new)
        model_new.export_dynamic_axes = types.MethodType(export_dynamic_axes, model_new)

    model_new.export_name = 'model'
    model_new.device = kwargs.get("device")
    del model
    return model_new.to(kwargs.get("device"))


def export_forward(
    self,
    speech: torch.Tensor,
    speech_lengths: torch.Tensor,
):
    batch = {"x": speech, "ilens": speech_lengths}

    x, olens = self.audio_encoder(**batch)
    encoder_out, encoder_out_lens = self.audio_adaptor(x, olens)
    encoder_out_ctc, encoder_out_lens_ctc = None, None
    if hasattr(self, "ctc_decoder"):
        encoder_out_ctc, encoder_out_lens_ctc = self.ctc_decoder(x, olens)
        encoder_out_ctc = self.ctc.log_softmax(encoder_out_ctc)
        return encoder_out, encoder_out_lens, encoder_out_ctc, encoder_out_lens_ctc
    else:
        return encoder_out, encoder_out_lens


def export_dummy_inputs(self, **kwargs):
    return self.audio_encoder.get_dummy_inputs(**kwargs)

def export_input_names(self):
    return self.audio_encoder.get_input_names()

def export_output_names(self):
    return ["encoder_out", "encoder_out_lens"]


def export_dynamic_axes(self):
    return {
        "x": {0: "batch_size", 2: "x_length"},
        "ilens": {0: "batch_size"},
        "encoder_out": {0: "batch_size", 1: "enc_out_length"},
        "encoder_out_lens": {0: "batch_size"},
    }


def export_output_names_ctc(self):
    return ["encoder_out", "encoder_out_lens", "encoder_out_ctc", "encoder_out_lens_ctc"]


def export_dynamic_axes_ctc(self):
    return {
        "x": {0: "batch_size", 2: "x_length"},
        "ilens": {0: "batch_size"},
        "encoder_out": {0: "batch_size", 1: "enc_out_length"},
        "encoder_out_lens": {0: "batch_size"},
        "encoder_out_ctc": {0: "batch_size", 1: "enc_out_lengt_ctch"},
        "encoder_out_lens_ctc": {0: "batch_size"},
    }