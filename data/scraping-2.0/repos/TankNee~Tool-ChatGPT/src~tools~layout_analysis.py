from tools.base_tool import BaseTool
from utils import get_output_path, prompts, logger
from PIL import Image
import numpy as np
from langchain.tools import Tool

import sys

sys.path.append("unilm")

from unilm.dit.object_detection.ditod import add_vit_config
from detectron2.config import CfgNode as CN
from detectron2.config import get_cfg
from detectron2.utils.visualizer import ColorMode, Visualizer
from detectron2.data import MetadataCatalog
from detectron2.engine import DefaultPredictor
import os


class LayoutAnalysis(BaseTool):

    def __init__(self, path, cascade_dit_base_cfg_path, device, llm) -> None:
        super().__init__(llm)
        self.cfg = get_cfg()
        add_vit_config(self.cfg)

        self.cfg.merge_from_file(cascade_dit_base_cfg_path)
        self.cfg.MODEL.WEIGHTS = path
        self.cfg.MODEL.DEVICE = device

        self.predictor = DefaultPredictor(self.cfg)

        self.thing_classes = ["text", "title", "list", "table", "figure"]
        self.md = MetadataCatalog.get(self.cfg.DATASETS.TEST[0])
        self.md.set(thing_classes=self.thing_classes)

    def analysis(self, img_path):
        assert os.path.exists(img_path), f"Image path {img_path} not exists"

        img = Image.open(img_path).convert("RGB")
        img = np.array(img)
        predict_result = self.predictor(img)
        instance = predict_result["instances"]
        v = Visualizer(img[:, :, ::-1],
                       self.md,
                       scale=1.0,
                       instance_mode=ColorMode.SEGMENTATION)
        result = v.draw_instance_predictions(instance.to("cpu"))
        # result_img = result.get_image()[:, :, ::-1]

        # save result image
        output_path = get_output_path(img_path)
        result.save(output_path)

        return instance, output_path

    @prompts(name="Layout Segment Tool",
             desc="useful when you want to recognize the layout of document."
             "The input to this tool should be a path string,"
             "representing the image_path of the document image."
             "The output to this tool is processed image path.")
    def segment_tool(self, img_path):
        _, output_path = self.analysis(img_path)
        return output_path

    @prompts(
        name="Layout Analysis Tool",
        desc="useful when you want to get the layout text info of document."
        "For example, you can see how many tables, text, headings, diagrams and the like are in the document"
        "The input to this tool should be a path string,"
        "representing the image_path of the document image."
        "The output to this tool is a dict, "
        "which contains the list of image file path of document component."
    )
    def meta_info_tool(self, img_path):
        instance, _ = self.analysis(img_path)
        fields = instance.get_fields()
        output_dict = {}
        for pred_class, pred_boxes in zip(fields['pred_classes'],
                                          fields['pred_boxes']):
            pred_class = self.thing_classes[pred_class]
            cut_img = self.cut_img_tool(img_path, pred_boxes)
            if pred_class not in output_dict:
                output_dict[pred_class] = [cut_img]
            else:
                output_dict[pred_class].append(cut_img)
        logger.debug(f"Layout info is {output_dict}")
        return output_dict

    @prompts(
        name="Cut Image Tool",
        desc=
        "useful when you want to cut the image according to the bounding box."
        "The input to this tool should be a path string to the img and bounding box coordinates,"
        "you should get the bounding box coordinates from the output of the layout info tool."
        "bounding box coordinates is a string, which is separated by commas."
        "The order of the bounding box coordinates is [x0, y0, x1, y1],"
        "where (x0, y0) is the upper left corner of the bounding box,"
        "and (x1, y1) is the lower right corner of the bounding box."
        "The path string and bounding box coordinates are separated by a space strictly."
        "The output of this tool is a path to the cut image.")
    def cut_img_tool(self, img_path, boxes):
        # img_path, boxes_str = inputs.split(" ")
        assert os.path.exists(img_path), f"Image path {img_path} not exists"
        # boxes = boxes_str.split(",")
        boxes = [int(box) for box in boxes]
        img = Image.open(img_path).convert("RGB")

        cut_img = img.crop(boxes)
        output_path = get_output_path(img_path)
        cut_img.save(output_path)

        logger.debug(f"Cut image path is {output_path}")

        return output_path

    def get_tools(self):
        return [
            Tool(name=self.segment_tool.name,
                 description=self.segment_tool.desc,
                 func=self.segment_tool),
            Tool(name=self.meta_info_tool.name,
                 description=self.meta_info_tool.desc,
                 func=self.meta_info_tool),
            # Tool(name=self.cut_img_tool.name,
            #      description=self.cut_img_tool.desc,
            #      func=self.cut_img_tool)
        ]