import torch
import onnx
from torch import nn
from .utils import Openai_Textual, OpenClip_Textual, DEFAULT_EXPORT, compute_dif
import os
from onnxmltools.utils import float16_converter
import onnxruntime as ort
import numpy as np


class clip_onnx(nn.Module):
    def __init__(self, model, source: str = ""):
        super().__init__()
        self.model = model
        self.model.eval()
        self.visual_flag = False
        self.textual_flag = False

        self.onnx32_visual_path = None
        self.onnx32_textual_path = None
        self.onnx16_visual_path = None
        self.onnx16_textual_path = None

        self.providers = ["CPUExecutionProvider"]

        if source.startswith("openai"):
            self.wrapper = Openai_Textual
        elif source.startswith("open_clip"):
            self.wrapper = OpenClip_Textual
        else:
            raise AssertionError("Please identify you source, either openai or open_clip")

        for x in self.model.parameters():
            x.requires_grad = False

    def torch_export(self, model, dummy_input, path: str, export_params=DEFAULT_EXPORT):
        torch.onnx.export(model, dummy_input, path, **export_params)

    def onnx_checker(self, path: str):
        try:
            model = onnx.load(path)
            onnx.checker.check_model(model)
            del model
        except:
            onnx.checker.check_model(path)

    def convert_visual(self, dummy_input, visual_path: str = None,
                       export_params=DEFAULT_EXPORT):
        self.model.visual.eval()

        if visual_path is None:
            directory = "onnx32_visual"
            filename = "onnx32_visual.onnx"
            if os.path.isdir(directory) == False:
                os.mkdir(directory)
            visual_path = os.path.join(directory, filename)
        else:
            directory, filename = os.path.dirname(visual_path), os.path.basename(visual_path)
            if os.path.isdir(directory) == False:
                os.mkdir(directory)

        self.torch_export(self.model.visual, dummy_input, visual_path,
                          export_params=export_params)

        self.onnx32_visual_path = visual_path

    def convert_textual(self, dummy_input, textual_path: str = None, export_params=DEFAULT_EXPORT):

        if textual_path is None:
            directory = "onnx32_textual"
            filename = "onnx32_textual.onnx"
            if os.path.isdir(directory) == False:
                os.mkdir(directory)
            textual_path = os.path.join(directory, filename)
        else:
            directory, filename = os.path.dirname(textual_path), os.path.basename(textual_path)
            if os.path.isdir(directory) == False:
                os.mkdir(directory)

        textual = self.wrapper(self.model)
        self.torch_export(textual, dummy_input, textual_path,
                          export_params=export_params)

        self.onnx32_textual_path = textual_path

    def convert_onnx32(self, visual_input=None, visual_path: str = None, textual_input=None, textual_path: str = None,
                       verbose=True,
                       visual_export_params=DEFAULT_EXPORT,
                       textual_export_params=DEFAULT_EXPORT):

        isinstance_visual_input = isinstance(visual_input, (torch.Tensor))
        isinstance_textual_input = isinstance(textual_input, (torch.Tensor))

        if (not isinstance_visual_input) and (not isinstance_textual_input):
            raise Exception("[Marqo CLIP ONNX] Please, choose a dummy input")
        elif not isinstance_visual_input:
            print("[Marqo CLIP ONNX] Convert only textual model")
        elif not isinstance_textual_input:
            print("[Marqo CLIP ONNX] Convert only visual model")

        if isinstance_visual_input:
            self.visual_flag = True
            if verbose:
                print("[Marqo CLIP ONNX] Start convert visual model")
            self.convert_visual(visual_input, visual_path, visual_export_params)
            if verbose:
                print("[Marqo CLIP ONNX] Start check visual model")
            self.onnx_checker(self.onnx32_visual_path)

        if isinstance_textual_input:
            self.textual_flag = True
            if verbose:
                print("[Marqo CLIP ONNX] Start convert textual model")
            self.convert_textual(textual_input, textual_path, textual_export_params)
            if verbose:
                print("[Marqo CLIP ONNX] Start check textual model")
            self.onnx_checker(self.onnx32_textual_path)

        if verbose:
            print("[Marqo CLIP ONNX] Models converts successfully")

    def convert_onnx16_visual(self, visual_path: str = None):
        assert not self.onnx32_visual_path == None, "[Marqo CLIP ONNX] Please convert or load onnx32 model first"

        if visual_path is None:
            directory = "onnx16_visual"
            filename = "onnx16_visual.onnx"
            if os.path.isdir(directory) == False:
                os.mkdir(directory)
            visual_path = os.path.join(directory, filename)
        else:
            directory, filename = os.path.dirname(visual_path), os.path.basename(visual_path)
            if os.path.isdir(directory) == False:
                os.mkdir(directory)

        f16_visual_model = float16_converter.convert_float_to_float16_model_path(self.onnx32_visual_path)
        onnx.save_model(f16_visual_model, visual_path)

        self.onnx16_visual_path = visual_path

    def convert_onnx16_textual(self, textual_path: str = None):
        assert not self.onnx32_textual_path == None, "[Marqo CLIP ONNX] Please convert or load onnx32 model first"

        if textual_path is None:
            directory = "onnx16_textual"
            filename = "onnx16_textual.onnx"
            if os.path.isdir(directory) == False:
                os.mkdir(directory)
            textual_path = os.path.join(directory, filename)
        else:
            directory, filename = os.path.dirname(textual_path), os.path.basename(textual_path)
            if os.path.isdir(directory) == False:
                os.mkdir(directory)

        f16_textual_model = float16_converter.convert_float_to_float16_model_path(self.onnx32_textual_path)
        onnx.save_model(f16_textual_model, textual_path)

        self.onnx16_textual_path = textual_path

    def convert_onnx16(self, visual_path: str = None, textual_path: str = None, verbose=True):

        if verbose:
            print("[Marqo CLIP ONNX] Start convert visual model to float16")
        self.convert_onnx16_visual(visual_path)

        if verbose:
            print("[Marqo CLIP ONNX] Start check onnx16 visual model")
        self.onnx_checker(self.onnx16_visual_path)

        if verbose:
            print("[Marqo CLIP ONNX] Start convert textual model to float16")
        self.convert_onnx16_textual(textual_path)

        if verbose:
            print("[Marqo CLIP ONNX] Start check onnx16 textual model")
        self.onnx_checker(self.onnx16_textual_path)

        if verbose:
            print("[Marqo CLIP ONNX] ONNX conversion finished")

    def check_diff_onnx32(self, image, text, onnx_image, onnx_text):

        print("Start computing difference for onnx32 model:")


        with torch.no_grad():
            torch_image_output = self.model.encode_image(image).detach().cpu().numpy()
            torch_text_output = self.model.encode_text(text).detach().cpu().numpy()

        f32_textual_session = ort.InferenceSession(self.onnx32_textual_path, providers=self.providers)
        f32_visual_session = ort.InferenceSession(self.onnx32_visual_path, providers=self.providers)

        f32onnx_image_output = f32_visual_session.run(None, {"input": onnx_image})
        f32onnx_text_output = f32_textual_session.run(None, {"input": onnx_text})
        print(
            f"float32onnx image sum difference with normalization: {compute_dif(torch_image_output, f32onnx_image_output)}")

        print(
            f"float32onnx text sum difference with normalization: {compute_dif(torch_text_output, f32onnx_text_output)}")

    def check_diff_onnx16(self, image, text, onnx_image, onnx_text):

        print("Start computing difference for onnx16 model:")

        with torch.no_grad():
            torch_image_output = self.model.encode_image(image).detach().cpu().numpy()
            torch_text_output = self.model.encode_text(text).detach().cpu().numpy()

        f16_textual_session = ort.InferenceSession(self.onnx16_textual_path, providers=self.providers)
        f16_visual_session = ort.InferenceSession(self.onnx16_visual_path, providers=self.providers)

        f16onnx_image_output = f16_visual_session.run(None, {"input": onnx_image.astype(np.float16)})
        f16onnx_text_output = f16_textual_session.run(None, {"input": onnx_text})

        print(
            f"float16onnx image sum difference with normalization: {compute_dif(torch_image_output, f16onnx_image_output)}")

        print(
            f"float16onnx text sum difference with normalization: {compute_dif(torch_text_output, f16onnx_text_output)}")


    def check_difference(self, image, text, onnx_image, onnx_text):
        if all([self.onnx32_textual_path, self.onnx32_visual_path]):
            self.check_diff_onnx32(image, text, onnx_image, onnx_text)
        else:
            print("No onnx32 models are loaded! We skip the difference computation")

        if all([self.onnx16_textual_path, self.onnx16_visual_path]):
            self.check_diff_onnx16(image, text, onnx_image, onnx_text)
        else:
            print("No onnx16 models are loaded! We skip the difference computation")

        print("Difference check finished!")

    def load(self, onnx32_visual_path:str = None, onnx32_textual_path:str = None, onnx16_visual_path:str = None, onnx16_textual_path:str = None):

        try:
            self.onnx_checker(onnx32_visual_path)
            self.onnx32_visual_path = onnx32_visual_path
        except:
            print("Invalid onnx32_visual_path. Loading failed")


        try:
            self.onnx_checker(onnx32_textual_path)
            self.onnx32_textual_path = onnx32_textual_path
        except:
            print("Invalid onnx32_textual_path. Loading failed")


        try:
            self.onnx_checker(onnx16_visual_path)
            self.onnx16_visual_path = onnx16_visual_path
        except:
            print("Invalid onnx16_visual_path. Loading failed")


        try:
            self.onnx_checker(onnx16_textual_path)
            self.onnx16_textual_path = onnx16_textual_path
        except:
            print("Invalid onnx16_textual_path. Loading failed")
