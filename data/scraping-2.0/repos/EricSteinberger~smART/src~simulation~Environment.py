# Copyright(c) Eric Steinberger 2018

"""
The Environment class is decoupled from Iterative Stroke Sampling to allow for easy development of other algorithms. The
split of ISS into the Environment and the agent is inspired from OpenAI's Gym, altough this architecture doesn't
implement a .step() method and has a different API in general - the concept of Environment+Agent interaction stands.
"""

import copy
import os
import pickle

import numpy as np
import torch
from PIL import Image, ImageFilter

from src import paths
from src.config import BrushConfig, RobotConfig
from src.simulation import commands as cmd


class Environment:

    def __init__(self,
                 args,
                 target_img,
                 ):

        self.args = args
        self.name = args.name
        self.device = torch.device("cuda:0" if args.use_gpu else "cpu")
        self.max_batch_size = args.max_batch_size_env

        self.target_img = torch.from_numpy(np.array(target_img).swapaxes(0, 2)).to(dtype=torch.float32,
                                                                                   device=self.device)
        self.target_img = self._invert(self.target_img, scale=255)

        self.px_per_mm = self.args.px_per_mm
        self.stroke_size_mm = self.args.stroke_size_mm
        self.stroke_size_px = int(self.args.stroke_size_mm * self.px_per_mm)
        self.paper_brightness_255 = args.paper_brightness

        self.canvas_center_xyz_mm = RobotConfig.CANVAS_CENTER

        self.opacity_from_dark_to_bright = args.opacity_from_dark_to_bright
        self.opacity_from_bright_to_dark = args.opacity_from_bright_to_dark
        self.overpaint_punishment_threshold = args.overpaint_punishment_threshold
        self.overpaint_punish_factor = args.overpaint_punish_factor

        self.px_x_size = None
        self.px_y_size = None
        self.painting_size_x_mm = None
        self.painting_size_y_mm = None
        self.complete_edges = None

        # _________________________________________ outputs of an algorithm making a painting
        # this is where ISS will write the stroke commands to to
        self.command_sequence = None
        self.current_img = None
        self.todo_img = None

        # _________________________________________ tracking vars
        self.current_brush = None
        self.current_color = None
        self.num_strokes_done = None
        self.num_switched_color = None
        self.num_got_color = None
        self.num_cleaned = None
        self.num_switched_brush = None
        self.how_many_times_painted_pixel = None

        # crunched between 0 and 1, and _invert strokes dark -> bright ; bright -> dark
        self.strokes = self._load_strokes()

    def reset(self):
        self.command_sequence = cmd.CommandSequence(args=self.args)
        self.current_img = torch.full_like(self.target_img, fill_value=self.paper_brightness_255)
        self.todo_img = self.target_img - self.current_img

        self.px_x_size = self.target_img.shape[1]
        self.px_y_size = self.target_img.shape[2]
        self.painting_size_x_mm = self.px_x_size / self.px_per_mm
        self.painting_size_y_mm = self.px_y_size / self.px_per_mm

        self.complete_edges = [0, self.px_x_size, 0, self.px_y_size]  # 0: left X;  1: right X;  2: bottom Y;  3: top Y

        # _________________________________________ tracking vars
        self.current_brush = BrushConfig.NOTHING_MOUNTED
        self.current_color = None
        self.num_strokes_done = 0
        self.num_switched_color = 0
        self.num_got_color = 0
        self.num_switched_brush = 0
        self.num_cleaned = 0
        self.how_many_times_painted_pixel = torch.zeros((self.px_x_size, self.px_y_size),
                                                        dtype=torch.float32, device=self.device)

    # _________________________________________________ execute action _________________________________________________
    def apply_stroke(self, stroke_id, rotation_id, center_x, center_y):

        # ___________________________________ apply the stroke to simulated painting ___________________________________
        _x = center_x - int(self.stroke_size_px / 2.0)
        _y = center_y - int(self.stroke_size_px / 2.0)

        self.current_img[:, _x:_x + self.stroke_size_px, _y:_y + self.stroke_size_px] = self._do_acryl(
            current_img_patches=self.current_img[:, _x:_x + self.stroke_size_px, _y:_y + self.stroke_size_px]
                .unsqueeze(0),
            stroke_batch=self.strokes[self.current_brush][stroke_id, rotation_id].unsqueeze(0),
            color=self.current_color)[0]

        self.how_many_times_painted_pixel[_x:_x + self.stroke_size_px, _y:_y + self.stroke_size_px] += \
            self.strokes[self.current_brush][stroke_id, rotation_id]

        # ________________________________________________ add to cmd seq ______________________________________________
        self.command_sequence.append(cmd.ApplyStroke(brush=self.current_brush,
                                                     color=self.current_color,
                                                     stroke_id=stroke_id,
                                                     rotation_id=rotation_id,
                                                     canvas_center_xyz_mm=self.canvas_center_xyz_mm,
                                                     center_x_mm=self._px_2_mm(px=center_x),
                                                     center_y_mm=self._px_2_mm(px=center_y),
                                                     painting_size_x_mm=self.painting_size_x_mm,
                                                     painting_size_y_mm=self.painting_size_y_mm,
                                                     ))
        # _________________________________________________ update to-do _______________________________________________
        self._update_todo()

        # _______________________________________________ increment counter ____________________________________________
        self.num_strokes_done += 1

    def change_brush(self, new_brush):
        if new_brush is not BrushConfig.NOTHING_MOUNTED:
            self.clean()
            self.num_switched_brush += 1

        self.command_sequence.append(cmd.ChangeBrush(from_brush=self.current_brush, to_brush=new_brush))
        self.current_brush = new_brush

    def change_color(self, new_color):
        self.clean()
        self.current_color = new_color
        self.num_switched_color += 1
        self.get_color()

    def get_color(self):
        self.command_sequence.append(cmd.GetColor(color=self.current_color))

    def clean(self):
        self.command_sequence.append(cmd.Clean())
        self.num_cleaned += 1

    def finished(self, save=True):
        # The robot should not have a brush on in the end
        self.change_brush(new_brush=BrushConfig.NOTHING_MOUNTED)

        # The order of the strokes might be optimizable.
        # There are also possible ineffient seqs like get_color -> get_color that get removed here
        self.command_sequence.optimize_order()

        # maybe store state
        if save:
            self.store_state_to_disk()

        # the torch array is converted to a pillow image and stored on disk
        img = self._output_curr_image_to_file()

        return copy.deepcopy(self.command_sequence), img

    # ____________________________________________________ general _____________________________________________________
    def get_errors_of_batch(self, strokes, centers_x, centers_y, colors_rgb=None):
        """
        this is a batched op!

        :param strokes expects 3d tensor. [batch_idx, x_px, y_px] -> opacity
        :param centers_x expects 1d tensor of ints. [batch_idx] -> coordinate
        :param centers_y expects 1d tensor of ints. [batch_idx] -> coordinate
        :param colors_rgb expects 2d tensor with size 3 in the 2nd dim (RGB). [batch_idx, color_channel] -> brightness

        :returns error for each stroke. negative is good.
        """

        # ____________________________________ General Evaluation with simulation ______________________________________
        batch_size = centers_x.size(0)
        errors = torch.zeros(batch_size, dtype=torch.float32, device=self.device)

        target_img_patches = self.get_patches(tensor=self.target_img, centers_x=centers_x, centers_y=centers_y)
        current_img_patches = self.get_patches(tensor=self.current_img, centers_x=centers_x, centers_y=centers_y)
        todo_img_patches = self.get_patches(tensor=self.todo_img, centers_x=centers_x, centers_y=centers_y)

        # compute difference between target_img and image after stroke would be done without actually applying it
        todos_after = target_img_patches - self._do_acryl(current_img_patches=current_img_patches,
                                                          stroke_batch=strokes,
                                                          colors_rgb=colors_rgb)

        errors += todos_after.abs().view(batch_size, -1).sum(1) - todo_img_patches.abs().view(batch_size, -1).sum(1)

        # __________________________________ Add punishment for overpainting often _____________________________________
        how_many_times_painted_pixel_patches = self.get_patches_no_color_channel(
            tensor=self.how_many_times_painted_pixel,
            centers_x=centers_x,
            centers_y=centers_y)

        penalty_mask = (how_many_times_painted_pixel_patches > self.overpaint_punishment_threshold).to(torch.float32)
        penalties_per_px = (how_many_times_painted_pixel_patches - self.overpaint_punishment_threshold) \
                           * strokes * penalty_mask * self.overpaint_punish_factor

        penalities = penalties_per_px.view(batch_size, -1).sum(1)

        errors += penalities

        return errors

    def _do_acryl(self, current_img_patches, stroke_batch, color=None, colors_rgb=None):
        """
        this is a batched op!

        :param stroke_batch:            torch tensor like: (batch_size, x, y)
        :param current_img_patches:     torch tensor like: (batch_size, rgb, x, y). will not be modified
        """

        assert (color is not None or colors_rgb is not None)
        assert current_img_patches.shape[0] == stroke_batch.shape[0]
        assert current_img_patches.shape[2] == stroke_batch.shape[1]
        assert current_img_patches.shape[3] == stroke_batch.shape[2]

        if colors_rgb is None:
            _rgb = color.rgb_torch_iss.to(device=self.device) \
                .unsqueeze(0).unsqueeze(-1).unsqueeze(-1).expand_as(current_img_patches)
        else:
            _rgb = colors_rgb.unsqueeze(-1).unsqueeze(-1).expand_as(current_img_patches)

        opacity_from_dark_to_bright = torch.tensor([[[[self.opacity_from_dark_to_bright]]]], device=self.device,
                                                   dtype=torch.float32).expand_as(current_img_patches)
        opacity_from_bright_to_dark = torch.tensor([[[[self.opacity_from_bright_to_dark]]]], device=self.device,
                                                   dtype=torch.float32).expand_as(current_img_patches)

        opacity_map = torch.where(_rgb < current_img_patches,
                                  opacity_from_dark_to_bright,
                                  opacity_from_bright_to_dark)

        stroke_ = stroke_batch.unsqueeze(1).expand(-1, 3, -1, -1) * opacity_map

        old_part = (1 - stroke_) * current_img_patches
        new_part = stroke_ * _rgb
        return old_part + new_part

    def _update_todo(self):
        self.todo_img = self.target_img - self.current_img

    def _px_2_mm(self, px):
        return float(px) / self.px_per_mm

    def _mm_2_px(self, mm):
        return self.px_per_mm * mm

    def _invert(self, tensor, scale=255):
        if scale == 255:  # to make sure 255.0 doesnt cause float conversion
            return 255 - tensor
        else:
            return scale - tensor

    # ____________________________________________________ matrix util _________________________________________________
    def get_patches(self, tensor, centers_x, centers_y):
        """
        :param tensor:          3D Tensor: [color, x, y]
        :param centers_x:
        :param centers_y:
        :return:
        """
        full_batch_size = centers_x.size(0)

        patches = torch.zeros((full_batch_size, 3, self.stroke_size_px, self.stroke_size_px),
                              dtype=tensor.dtype,
                              device=tensor.device)

        x_idxs = self.make_center_indxs_slices(centers=centers_x, range_=self.stroke_size_px)
        y_idxs = self.make_center_indxs_slices(centers=centers_y, range_=self.stroke_size_px)

        sub_batches, _, __ = self.get_sub_batches(batched_tensors_dict={"x_idxs": x_idxs,
                                                                        "y_idxs": y_idxs},
                                                  max_batch_size=self.max_batch_size)
        k = 0
        for c, sub_batch in enumerate(sub_batches):
            sub_batch_size = sub_batch["x_idxs"].size(0)
            _first_dim_patched = tensor[:, sub_batch["x_idxs"]].transpose(0, 1)  # get batch to dim 0

            # TODO find a way to do this without python loop
            for i in range(sub_batch_size):
                patches[k] = _first_dim_patched[i, :, :, sub_batch["y_idxs"][i]]
                k += 1

        return patches

    def get_patches_no_color_channel(self, tensor, centers_x, centers_y):
        """
        :param tensor:          2D Tensor: [x, y]
        :param centers_x:
        :param centers_y:
        :return:
        """
        full_batch_size = centers_x.size(0)

        patches = torch.zeros((full_batch_size, self.stroke_size_px, self.stroke_size_px),
                              dtype=tensor.dtype,
                              device=tensor.device)

        x_idxs = self.make_center_indxs_slices(centers=centers_x, range_=self.stroke_size_px)
        y_idxs = self.make_center_indxs_slices(centers=centers_y, range_=self.stroke_size_px)

        sub_batches, _, __ = self.get_sub_batches(batched_tensors_dict={"x_idxs": x_idxs,
                                                                        "y_idxs": y_idxs},
                                                  max_batch_size=self.max_batch_size)

        k = 0
        for c, sub_batch in enumerate(sub_batches):
            sub_batch_size = sub_batch["x_idxs"].size(0)
            _first_dim_patched = tensor[sub_batch["x_idxs"]]

            # TODO find a way to do this without python loop
            for i in range(sub_batch_size):
                patches[k] = _first_dim_patched[i, :, sub_batch["y_idxs"][i]]
                k += 1

        return patches

    def get_sub_batches(self, batched_tensors_dict, max_batch_size):
        sub_batches = []

        keys = list(batched_tensors_dict.keys())
        original_batch_size = batched_tensors_dict[keys[0]].size(0)
        n_slices = int(np.ceil(original_batch_size / max_batch_size))

        starts = torch.arange(n_slices, dtype=torch.long, device=self.device) * max_batch_size
        ends = torch.arange(1, n_slices + 1, dtype=torch.long, device=self.device) * max_batch_size
        ends[-1] = original_batch_size

        for i in range(n_slices):
            sub_batches.append(
                {
                    k: batched_tensors_dict[k][starts[i]:ends[i]]
                    for k in keys
                }
            )

        return sub_batches, starts, ends

    def make_center_indxs_slices(self, centers, range_):
        idxs = torch.arange(range_, dtype=torch.long, device=self.device)
        idxs = idxs.unsqueeze(0)
        idxs = idxs.repeat(centers.size(0), 1)
        idxs += (centers - int(range_ / 2.0)).unsqueeze(1).expand_as(idxs)
        return idxs

    # __________________________________________________ loading & export ______________________________________________
    def _load_strokes(self):
        def _rotate(_im, deg):
            """ takes care of alpha layer issues. See https://github.com/python-pillow/Pillow/issues/428 """

            _im2 = _im.convert('RGBA')

            # rotate image
            _r_im = _im2.rotate(angle=deg, resample=Image.BILINEAR)
            # a white image same size as rotated image
            fff = Image.new('RGBA', _im.size, (255,) * 4)

            # create a composite image using the alpha layer of rot as a mask
            out = Image.composite(_r_im, fff, _r_im)
            out.filter(ImageFilter.GaussianBlur(radius=4))

            # return out in original mode
            return out.convert(_im.mode)

        strokes = {}
        for brush in BrushConfig.ALL_PAINTING_BRUSHES_LIST:
            strokes_this_brush = torch.zeros(
                size=(len(brush.stroke_names_list),
                      self.args.n_stroke_rotations,
                      self.stroke_size_px,
                      self.stroke_size_px),
                dtype=torch.float32, device=self.device)

            # ----- NEW
            for i, stroke_name in enumerate(brush.stroke_names_list):
                print("Loading stroke: ", stroke_name)
                path_to_img = os.path.join(paths.stroke_images_path, brush.name, stroke_name + ".png")
                image = Image.open(path_to_img).convert('L')  # greyscale

                for r in range(self.args.n_stroke_rotations):
                    r_img = _rotate(_im=image, deg=r * 360.0 / self.args.n_stroke_rotations)
                    strokes_this_brush[i, r] = self._invert(torch.from_numpy(np.array(r_img)), scale=255) / 255.0

            strokes[brush] = strokes_this_brush.to(dtype=torch.float32, device=self.device)

        return strokes

    def _output_curr_image_to_file(self):
        img = self._to_pillow(img=self.current_img)
        img.save(os.path.join(paths.simulated_paintings_imgs_path, self.name + ".jpg"), "JPEG")
        return img

    def _to_pillow(self, img):
        _img = img.cpu().numpy().astype("uint8").swapaxes(0, 2)
        _img = Image.fromarray(self._invert(_img, scale=255))
        return _img

    def state_dict(self):
        return {
            "command_sequence": copy.deepcopy(self.command_sequence),
            "target_img": self.target_img.cpu().clone(),
            "current_img": self.current_img.cpu().clone(),
            "todo_img": self.todo_img.cpu().clone(),
            "current_brush": self.current_brush,
            "current_color": self.current_color,
            "num_strokes_done": self.num_strokes_done,
            "num_switched_color": self.num_switched_color,
            "num_got_color": self.num_got_color,
            "num_switched_brush": self.num_switched_brush,
            "num_cleaned": self.num_cleaned,
            "how_many_times_painted_pixel": self.how_many_times_painted_pixel.cpu().clone()
        }

    def load_state_dict(self, state):
        self.command_sequence = copy.deepcopy(state["command_sequence"])
        self.target_img = state["target_img"].clone().to(self.device)
        self.current_img = state["current_img"].clone().to(self.device)
        self.todo_img = state["todo_img"].clone().to(self.device)
        self.current_brush = state["current_brush"]
        self.current_color = state["current_color"]
        self.num_strokes_done = state["num_strokes_done"]
        self.num_switched_color = state["num_switched_color"]
        self.num_got_color = state["num_got_color"]
        self.num_switched_brush = state["num_switched_brush"]
        self.num_cleaned = state["num_cleaned"]
        self.how_many_times_painted_pixel = state["how_many_times_painted_pixel"].clone().to(self.device)

    def store_state_to_disk(self):
        print("exporting")
        with open(os.path.join(paths.main_fn_command_seqs_path, self.name + ".pk"), 'wb') as f:
            pickle.dump(obj=self.state_dict(), file=f)

    def load_state_from_disk(self):
        with open(os.path.join(paths.main_fn_command_seqs_path, self.name + ".pk"), 'rb') as f:
            self.load_state_dict(pickle.load(file=f))
