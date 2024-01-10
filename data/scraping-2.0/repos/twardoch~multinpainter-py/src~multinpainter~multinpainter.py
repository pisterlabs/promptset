import io
import os
import json
import logging
from datetime import datetime
from pathlib import Path
from collections import OrderedDict
from typing import Optional, Union, Tuple, Dict, List, Any
import base64

import httpx
import aiohttp
import numpy as np
import openai
import requests
from PIL import Image
from tqdm import tqdm

from multinpainter import __version__
from .utils import image_to_png

__author__ = "Adam Twardoch"
__license__ = "Apache-2.0"

DESCRPTION_MODEL="Salesforce/blip2-opt-2.7b"

class Multinpainter_OpenAI:
    f"""
    A class for iterative inpainting using OpenAI's Dall-E 2 and GPT-3 atificial intelligence models to extend (outpaint) an existing image to new defined dimensions.

    Args:
        image_path (str): Path to the input image file.
        out_path (str): Path to save the output image file.
        out_width (int): Desired width of the output image.
        out_height (int): Desired height of the output image.
        prompt (str, optional): Prompt for the GPT-3 model to generate image content.
        fallback (str, optional): Fallback prompt to use when the original prompt contains
            human-related items. Defaults to None.
        step (int, optional): The number of pixels to shift the square in each direction
            during the iterative inpainting process. Defaults to None.
        square (int, optional): Size of the square region to inpaint in pixels. Defaults to 1024.
        humans (bool, optional): Whether to include human-related items in the prompt.
            Defaults to True.
        verbose (bool, optional): Whether to enable verbose logging. Defaults to False.
        openai_api_key (str, optional): OpenAI API key or OPENAI_API_KEY env variable.
        hf_api_key (str, optional): Huggingface API key or HUGGINGFACEHUB_API_TOKEN env variable.
        prompt_model (str, optional): The Huggingface model to describe image. Defaults to "{DESCRPTION_MODEL}".

    A class for iterative inpainting using OpenAI's Dall-E 2 and GPT-3 models to generate image content from an input image and prompt.

    Attributes:
        image_path (str): Path to input image file.
        out_path (str): Path to save output image file.
        out_width (int): Desired width of output image.
        out_height (int): Desired height of output image.
        prompt (str): Prompt for GPT-3 model to generate image content.
        fallback (str): Fallback prompt to use when the original prompt contains human-related items. Defaults to None.
        step (int): The number of pixels to shift the square in each direction during the iterative inpainting process. Defaults to None.
        square (int): Size of the square region to inpaint in pixels. Defaults to 1024.
        humans (bool): Whether to include human-related items in the prompt. Defaults to True.
        verbose (bool): Whether to enable verbose logging. Defaults to False.
        openai_api_key (str): OpenAI API key or OPENAI_API_KEY env variable.
        hf_api_key (str): Huggingface API key or HUGGINGFACEHUB_API_TOKEN env variable.
        prompt_model (str): Huggingface model to describe image. Defaults to "{DESCRPTION_MODEL}".
        input_width (int): Width of input image.
        input_height (int): Height of input image.
        image (PIL.Image.Image): Input image as a PIL.Image object.
        out_image (PIL.Image.Image): Output image as a PIL.Image object.
        center_of_focus (Tuple[int,int]): Coordinates of center of focus in input image.
        expansion (Tuple[int,int,int,int]): Expansion values for input image to fit output size.
        human_boxes (List[Tuple[int,int,int,int]]): List of bounding boxes for detected humans in input image.

    Methods:
        configure_logging(): Sets up logging configuration based on verbose flag.
        timestamp(): Returns current timestamp in format '%Y%m%d-%H%M%S'.
        snapshot(): Takes snapshot of current output image and saves to disk.
        open_image(): Opens input image and converts to RGBA format.
        save_image(): Saves output image to disk in PNG format.
        to_rgba(image): Converts input image to RGBA format and returns.
        to_png(image): Converts input image to PNG format and returns binary data.
        make_prompt_fallback(): Generates fallback prompt if given prompt contains human-related items.
        create_out_image(): Creates new RGBA image of size out_width x out_height with transparent background and returns.
        describe_image(): Generates prompt using a Huggingface image captioning model.
        detect_humans(): Detects humans in input image using YOLOv5 model from ultralytics package.
        detect_faces(): Detects a face in input image using dlib face detector.
        find_center_of_focus(): Finds center of focus for output image.
        calculate_expansion(): Calculates amount of expansion needed to fit input image into output image while maintaining center of focus.
        paste_input_image(): Pastes input image onto output image, taking into account calculated expansion.
        openai_inpaint(png, prompt): Calls OpenAI's Image Inpainting API to inpaint given square of output image.
        get_initial_square_position(): Calculates position of initial square to be inpainted.
        human_in_square(square_box): Checks if given square contains human.
        inpaint_square(square_delta): Inpaints given square based on square_delta.
        create_planned_squares(): Generates list of squares to be inpainted in a specific order.
        move_square(square_delta, direction): Moves given square in specified direction by step size.
        iterative_inpainting(): Performs iterative inpainting process by calling inpaint_square() method on each square in planned square list.
        inpaint(): Asynchronous main entry point for Multinpainter_OpenAI class.

    Usage:
        import asyncio
        from multinpainter import Multinpainter_OpenAI
        inpainter = Multinpainter_OpenAI(
            image_path="input_image.png",
            out_path="output_image.png",
            out_width=1920,
            out_height=1080,
            prompt="Asian woman in front of blue wall",
            fallback="Solid blue wall",
            square=1024,
            step=256,
            humans=True,
            verbose=True,
            openai_api_key="sk-NNNNNN",
            hf_api_key="hf_NNNNNN",
            prompt_model="{DESCRPTION_MODEL}"
        )
        asyncio.run(inpainter.inpaint())
    """

    def __init__(
        self,
        image_path: Union[str, Path],
        out_path: Union[str, Path] = None,
        out_width: int = 0,
        out_height: int = 0,
        prompt: Optional[str] = None,
        fallback: Optional[str] = None,
        step: Optional[int] = None,
        square: int = 1024,
        humans: bool = True,
        verbose: bool = False,
        openai_api_key: Optional[str] = None,
        hf_api_key: Optional[str] = None,
        prompt_model: str = None,
    ):
        f"""
        - Initialize the Multinpainter_OpenAI instance with the required input parameters.
        - Set up logging configurations.
        - Open the input image and create an output image.
        - Optionally detect humans in the input image using the YOLO model.
        - Optionally detect faces in the input image using the Dlib library.
        - Find the center of focus of the image (center of input image or the face if found).
        - Calculate the expansion of the output image.
        - Paste the input image onto the output image.
        - Create the outpainting plan by generating a list of square regions in different directions.

        Args:
            image_path (Union[str, Path]): The path of the input image file.
            out_path (Union[str, Path]): The path for the output inpainted image file.
            out_width (int): The width of the output image.
            out_height (int): The height of the output image.
            prompt (str, optional): The prompt text to be used in the inpainting process.
            fallback (str, optional): The fallback prompt text, used when inpainting non-human areas. Defaults to None.
            step (int, optional): The step size to move the inpainting square. Defaults to None.
            square (int, optional): The size of the inpainting square. Defaults to 1024.
            humans (bool, optional): Whether to consider humans in the inpainting process. Defaults to True.
            verbose (bool, optional): Whether to show verbose logging. Defaults to False.
            openai_api_key (str, optional): Your OpenAI API key, defaults to the OPENAI_API_KEY environment variable.
            hf_api_key (str, optional): Your Huggingface API key, defaults to the HUGGINGFACEHUB_API_TOKEN env variable.
            prompt_model (str, optional): The Huggingface model to describe image. Defaults to "{DESCRPTION_MODEL}".
        """
        self.verbose = verbose
        self.configure_logging()
        logging.info("Starting iterative OpenAI inpainter...")
        self.openai_api_key = openai_api_key or os.environ.get("OPENAI_API_KEY", None)
        openai.openai_api_key = self.openai_api_key
        self.hf_api_key = hf_api_key or os.environ.get("HUGGINGFACEHUB_API_TOKEN", None)
        self.image_path = Path(image_path)
        logging.info(f"Image path: {self.image_path}")        
        self.open_image()
        self.out_width = out_width
        self.out_height = out_height
        if not out_path:
            out_path = self.image_path.with_name(f"{self.image_path.stem}_outpainted-{self.out_width}x{self.out_height}.png")
        self.out_path = Path(out_path)
        logging.info(f"Output path: {self.out_path}")
        logging.info(f"Output size: {self.out_width}x{self.out_height}")
        self.prompt = prompt
        self.fallback = fallback
        self.prompt_model = prompt_model or DESCRPTION_MODEL # "Salesforce/blip2-opt-6.7b-coco" # 
        self.square = square
        self.step = step or square // 2
        self.center_of_focus = None
        self.humans = humans
        self.face_boxes = None

    def prep_inpainting(self):
        logging.info(f"Square size: {self.square}")
        logging.info(f"Step size: {self.step}")
        self.out_image = self.create_out_image()
        self.detect_faces()
        self.find_center_of_focus()
        self.expansion = self.calculate_expansion()
        self.human_boxes = self.detect_humans() if self.humans else []
        if len(self.human_boxes):
            self.make_prompt_fallback()
        self.paste_input_image()
        self.planned_squares = self.create_planned_squares()

    def configure_logging(self) -> None:
        """
        Configures the logging settings for the application.
        """
        log_level = logging.DEBUG if self.verbose else logging.WARNING
        logging.basicConfig(level=log_level, format="%(levelname)s: %(message)s")

    def timestamp(self) -> str:
        """
        Returns the current timestamp in the format 'YYYYMMDD-HHMMSS'.

        Returns:
            str: The current timestamp as a string.
        """
        return datetime.now().strftime("%Y%m%d-%H%M%S")

    def snapshot(self) -> None:
        """
        Saves a snapshot of the current output image with a timestamp in the file name. Only saves the snapshot if the verbose flag is set to True.
        """
        if self.verbose:
            snapshot_path = Path(
                self.out_path.parent,
                f"{self.out_path.stem}-{self.timestamp()}.png",
            )
            logging.info(f"Saving snapshot: {snapshot_path}")
            self.out_image.save(
                snapshot_path,
                format="PNG",
            )

    def open_image(self) -> None:
        """
        Opens the input image from the specified image path, converts it to RGBA format, and stores the image and its dimensions as instance variables.
        """
        self.image = self.to_rgba(Image.open(self.image_path))
        self.input_width, self.input_height = self.image.size
        logging.info(f"Input size: {self.input_width}x{self.input_height}")

    def save_image(self) -> None:
        """
        Saves the output image to the specified output path with a PNG format.
        """
        self.out_image.save(self.out_path.with_suffix(".png"), format="PNG")
        logging.info(f"Output image saved to: {self.out_path}")

    def to_rgba(self, image: Image) -> Image:
        """
        Converts the given image to RGBA format and returns the converted image.

        Args:
            image (Image): The input image to be converted.

        Returns:
            Image: The converted RGBA image.
        """

        return image.convert("RGBA")


    def make_prompt_fallback(self):
        """
        Generates a non-human version of the prompt using the GPT-3.5-turbo model.
        The method updates the instance variable `prompt_fallback` with the non-human version of the prompt.
        If a fallback prompt is already provided, this method does nothing.
        """

        if self.fallback:
            return False
        prompt = f"""Create a JSON dictionary. Rewrite this text into one Python list of short phrases, focusing on style, on the background, and on overall scenery, but ignoring humans and human-related items: "{self.prompt_human}". Put that list in the `descriptors` item. In the `ignored` item, put a list of the items from the `descriptors` list that have any relation to humans, human activity or human properties. In the `approved` item, put a list of the items from the `descriptors` list which are not in the `ignore` list, but also include items from the `descriptors` list that relate to style or time. Output only the JSON dictionary, no commentary or explanations."""
        logging.info(f"Adapting to non-human prompt:\n{prompt}")
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": ""},
                {"role": "user", "content": prompt},
            ],
        )
        result = response.choices[0].message.content
        logging.info(f"Non-human prompt result: {result}")
        try:
            prompt_fallback = json.loads(result).get("approved", [])
            self.prompt_fallback = ", ".join(prompt_fallback) + ", no humans"
            logging.info(f"Non-human prompt: {self.prompt_fallback}")
        except json.decoder.JSONDecodeError:
            logging.warning(f"Invalid non-human prompt: {result}")

    def create_out_image(self):
        """
        Creates the output image by combining the input image and the generated text.
        The method uses the instance variables `image` and `generated_text` to create the final output image.
        The resulting image is stored in the instance variable `out_image`.
        """
        return Image.new("RGBA", (self.out_width, self.out_height), (0, 0, 0, 0))

    async def describe_image(self, func_describe=None, *args, **kwargs):
        if func_describe is None:
            from .models import describe_image_hf
            func_describe = describe_image_hf

        logging.info("Describing image...")
        self.prompt = await func_describe(self.image, self.prompt_model, self.hf_api_key, *args, **kwargs)


    def detect_humans(self, func_detect=None, *args, **kwargs):
        """
        Detects human faces or bodies in the input image using a pre-trained model.
        The method processes the instance variable `image` and returns a list of detected human bounding boxes.
        Each bounding box is represented as a tuple (x, y, width, height).

        Returns:
            list: A list of detected human bounding boxes in the input image.
        """
        if func_detect is None:
            from .models import detect_humans_yolov8
            func_detect = detect_humans_yolov8

        self.human_boxes = func_detect(self.image, *args, **kwargs)
        logging.info(f"Detected humans: {self.human_boxes}")


    def detect_faces(self, func_detect=None, *args, **kwargs):
        """
        Detects human faces in the input image using a pre-trained model.
        The method processes the instance variable `image` and returns a list of detected face bounding boxes.
        Each bounding box is represented as a tuple (x, y, width, height).

        Returns:
            list: A list of detected face bounding boxes in the input image.
        """
        if func_detect is None:
            from .models import detect_faces_dlib
            func_detect = detect_faces_dlib

        self.face_boxes = func_detect(self.image, *args, **kwargs)
        logging.info(f"Detected faces: {self.face_boxes}")


    def find_center_of_focus(self):
        """
        Calculates the center of focus in the input image based on the positions of detected faces and humans.
        The method processes the instance variable `image` and uses the results of `detect_faces` and `detect_humans`
        to determine an optimal focal point for cropping or other image processing tasks.

        Returns:
            tuple: A tuple (x, y) representing the coordinates of the calculated center of focus in the input image.
        """
        if self.face_boxes:
            x_min, y_min, x_max, y_max = self.face_boxes[0]
            center_x = (x_min + x_max) // 2
            center_y = (y_min + y_max) // 2
            self.center_of_focus = center_x, center_y
        else:
            self.center_of_focus = self.image.size[0] // 2, self.image.size[1] // 2
        logging.info(f"Center of focus: {self.center_of_focus}")

    def calculate_expansion(self):
        x_percentage = self.center_of_focus[0] / self.input_width
        y_percentage = self.center_of_focus[1] / self.input_height

        x_left = int((self.out_width - self.input_width) * x_percentage)
        x_right = self.out_width - self.input_width - x_left
        y_top = int((self.out_height - self.input_height) * y_percentage)
        y_bottom = self.out_height - self.input_height - y_top

        return x_left, x_right, y_top, y_bottom

    def paste_input_image(self):
        """
        Pastes the input image onto the output image, considering the calculated expansion values.
        This method ensures that the input image is placed onto the output image, taking into account
        the expansion values to position the input image correctly within the output image canvas.
        """
        self.out_image.paste(self.image, (self.expansion[0], self.expansion[2]))

    def get_initial_square_position(self):
        """
        Calculates the initial position of the square used for inpainting.

        Returns:
            Tuple[int, int]: The initial (x, y) position of the top-left corner of the square.
        """
        x_init = max(0, self.expansion[0] - (self.square - self.input_width) // 2)
        y_init = max(0, self.expansion[2] - (self.square - self.input_height) // 2)
        return x_init, y_init

    def human_in_square(self, square_box: Tuple[int, int, int, int]) -> bool:
        """
        Determines whether any detected human bounding boxes intersect with the given square_box.

        Args:
            square_box (Tuple[int, int, int, int]): The (x0, y0, x1, y1) coordinates of the square_box.

        Returns:
            bool: True if any detected human bounding boxes intersect with the square_box, False otherwise.
        """
        x0, y0, x1, y1 = square_box

        for box in self.human_boxes:
            bx0, by0, bx1, by1 = box
            if x0 < bx1 and x1 > bx0 and y0 < by1 and y1 > by0:
                return True
        return False

    async def inpaint_square(self, square_delta: Tuple[int, int], func_inpaint=None, *args, **kwargs) -> None:
        """
        Inpaints the square region in the output image specified by square_delta using OpenAI's API.
        Chooses the appropriate prompt based on the presence of humans in the square.

        Args:
            square_delta (Tuple[int, int]): The (x, y) coordinates of the top-left corner of the square region.

        Returns:
            None
        """
        if func_inpaint is None:
            from .models import inpaint_square_openai
            func_inpaint = inpaint_square_openai

        x, y = square_delta
        x1, y1 = x + self.square, y + self.square
        if x >= self.expansion[0] and y >= self.expansion[2] and x1 <= self.expansion[0] + self.input_width and y1 <= self.expansion[2] + self.input_height:
            return

        square = self.out_image.crop((x, y, x1, y1))

        if self.human_in_square((x, y, x1, y1)):
            prompt = self.prompt_human
        else:
            prompt = self.prompt_fallback

        logging.info(f"Inpainting region {x} {y} {x1} {y1} with: {prompt}")
        inpainted_square = await func_inpaint(square, prompt, (self.square, self.square), self.openai_api_key, *args, **kwargs)
        self.out_image.paste(inpainted_square, (x, y))
        self.snapshot()

    def create_planned_squares(self):
        """
        Generates a dictionary that represents the order in which the image squares will be processed during the inpainting process.
        The dictionary has the following keys:
        - `init`: contains the initial square position.
        - `up`: contains the squares above the initial square, in the order they should be processed.
        - `left`: contains the squares to the left of the initial square, in the order they should be processed.
        - `right`: contains the squares to the right of the initial square, in the order they should be processed.
        - `down`: contains the squares below the initial square, in the order they should be processed.
        - `up_left`: contains the squares above and to the left of the initial square, in the order they should be processed.
        - `up_right`: contains the squares above and to the right of the initial square, in the order they should be processed.
        - `down_left`: contains the squares below and to the left of the initial square, in the order they should be processed.
        - `down_right`: contains the squares below and to the right of the initial square, in the order they should be processed.

        Each key in the dictionary is associated with a list of square positions that represent the order in which the inpainting process will occur.
        The order is determined by starting from the initial square and iterating over each direction (up, down, left, right) until there is no more space in that direction.
        Then, for each combination of up/down and left/right directions, the squares are ordered diagonally.

        Returns:
        The generated dictionary.
        """

        init_square = self.get_initial_square_position()

        planned_squares = OrderedDict(
            init=[init_square],
            up=[],
            left=[],
            right=[],
            down=[],
            up_left=[],
            up_right=[],
            down_left=[],
            down_right=[],
        )

        # Calculate up, left, right, and down squares
        x, y = init_square
        for direction in ["up", "left", "right", "down"]:
            cur_x, cur_y = x, y
            while True:
                cur_x, cur_y = self.move_square((cur_x, cur_y), direction)
                if cur_x is None or cur_y is None:
                    break
                planned_squares[direction].append((cur_x, cur_y))

        # Calculate up_left, up_right, down_left, and down_right squares
        for up_down in ["up", "down"]:
            for left_right in ["left", "right"]:
                quadrant = f"{up_down}_{left_right}"
                for up_sq in planned_squares[up_down]:
                    for lr_sq in planned_squares[left_right]:
                        quadrant_sq = (lr_sq[0], up_sq[1])
                        planned_squares[quadrant].append(quadrant_sq)

        logging.info(f"Planned squares: {planned_squares}")
        return planned_squares

    def move_square(
        self, square_delta: Tuple[int, int], direction: str
    ) -> Tuple[int, int]:
        """
        Calculates the position of the square in a given direction.

        Args:
        - square_delta: A tuple (x, y) representing the position of the square.
        - direction: A string representing the direction of the movement. Can be one of 'up', 'down', 'left', 'right'.

        Returns:
        - A tuple (x, y) representing the new position of the square after the movement in the given direction.
        - If the new position is outside the image, the corresponding coordinate is set to None.
        """

        x, y = square_delta

        if direction == "up":
            next_y = max(0, y - self.step)
            if next_y == y:
                return x, None
            return x, next_y
        elif direction == "left":
            next_x = max(0, x - self.step)
            if next_x == x:
                return None, y
            return next_x, y
        elif direction == "right":
            next_x = min(x + self.step, self.out_width - self.square)
            if next_x == x:
                return None, y
            return next_x, y
        elif direction == "down":
            next_y = min(y + self.step, self.out_height - self.square)
            if next_y == y:
                return x, None
            return x, next_y

    async def iterative_inpainting(self):
        """
        Iteratively performs the inpainting process by calling `inpaint_square` on each square in the order defined by `create_planned_squares`.
        Initializes and updates a progress bar to track the progress of the inpainting process.
        """
        if not self.prompt:
            self.prompt = await self.describe_image()
        self.prompt_human = self.prompt
        logging.info(f"Homan prompt: {self.prompt_human}")
        self.prompt_fallback = self.fallback or self.prompt
        logging.info(f"Fallback prompt: {self.prompt_fallback}")

        inpainting_plan = [
            square_delta
            for direction in self.planned_squares
            for square_delta in self.planned_squares[direction]
        ]
        progress_bar = tqdm(
            inpainting_plan, desc="Outpainting square", total=len(inpainting_plan)
        )
        for square_delta in progress_bar:
            await self.inpaint_square(square_delta)

    async def inpaint(self):
        """
        - Asynchronously perform outpainting for each square in the outpainting plan.
        - Save the output image.
        """
        self.prep_inpainting()
        await self.iterative_inpainting()
        self.save_image()
