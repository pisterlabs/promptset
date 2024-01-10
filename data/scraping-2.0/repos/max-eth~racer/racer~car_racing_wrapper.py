"""
The main portion of this code was taken from OpenAI Gym and it is licensed under the following license:

The MIT License

Copyright (c) 2016 OpenAI (https://openai.com)

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
"""
import math
from itertools import repeat
from Box2D.b2 import fixtureDef, polygonShape

import numpy as np
import matplotlib.pyplot as plt
import pyglet
from pyglet import gl
import skimage.transform
import tempfile
from scipy.spatial.transform import Rotation as R
import skimage.color
import gym.envs.box2d.car_racing

from tqdm import tqdm

from racer.car import Car, COLL_TILE_CATEGORY, COLL_WHEEL_CATEGORY

WINDOW_H = 1000
WINDOW_W = 1000


class CarRacingWrapper(gym.envs.box2d.car_racing.CarRacing):
    def __init__(
        self,
        enable_linear_speed,
        enable_angular_speed,
        enable_abs,
        enable_steering,
        headless,
        num_cars=1,
        focus_car=0,
        prerendered_data=None,
        render_view=False,
        export_frames=False,
    ):
        """
        Wrapper for the car racing environment

        :param enable_linear_speed: whether to enable the linear speed feature
        :param enable_angular_speed: whether to enable the angular speed feature
        :param enable_abs: whether to enable the abs feature
        :param enable_steering: whether to enable the steering feature
        :param imgs: pre-rendered track images for each degree
        """
        super(CarRacingWrapper, self).__init__(verbose=0)
        # collision detection
        """
        Collision Detection
        |-------------+---------------+---------------|
        | Object type | Category bits | collides with |
        |-------------+---------------+---------------|
        | Car         |        0x0004 | Nothing       |
        | Wheel       |        0x0002 | Tile          |
        | Tile        |        0x0001 | Wheel         |
        """
        self.fd_tile = fixtureDef(
            shape=polygonShape(vertices=[(0, 0), (1, 0), (1, -1), (0, -1)]),
            categoryBits=COLL_TILE_CATEGORY,
            maskBits=COLL_WHEEL_CATEGORY,
        )

        assert not (headless is True and prerendered_data is None)
        self.headless = headless
        self.last_action = None
        self.enable_linear_speed = enable_linear_speed
        self.enable_angular_speed = enable_angular_speed
        self.enable_abs = enable_abs
        self.enable_steering = enable_steering
        self.frames_path = tempfile.mkdtemp("frames")
        self.export_frames = export_frames
        self.frame = 0
        self.render_view = render_view
        assert num_cars > 0
        assert 0 <= focus_car < num_cars
        self.focus_car = focus_car
        self.num_cars = num_cars
        self.cars = None

        if render_view:
            assert num_cars == 1
            from pyformulas import screen

            self.canvas = np.zeros((32, 32))
            self.pyscreen = screen(self.canvas, "view")

        if prerendered_data is not None:
            self.track = prerendered_data["track"]
            self.track_imgs = prerendered_data["track_imgs"]
            self.road_vertex_list = None
            self.reset(regen_track=False)
        else:
            self.road_vertex_list = None
            self.reset(regen_track=True)

    def export(self):
        return {
            "track": self.track,
            "track_imgs": self.track_imgs,
        }

    def _create_track(self, basic=False):
        CHECKPOINTS = 12

        # Create checkpoints
        checkpoints = []
        for c in range(CHECKPOINTS):
            alpha = 2 * math.pi * c / CHECKPOINTS + self.np_random.uniform(
                0, 2 * math.pi * 1 / CHECKPOINTS
            )
            rad = self.np_random.uniform(
                gym.envs.box2d.car_racing.TRACK_RAD / 3,
                gym.envs.box2d.car_racing.TRACK_RAD,
            )
            if c == 0:
                alpha = 0
                rad = 1.5 * gym.envs.box2d.car_racing.TRACK_RAD
            if c == CHECKPOINTS - 1:
                alpha = 2 * math.pi * c / CHECKPOINTS
                self.start_alpha = 2 * math.pi * (-0.5) / CHECKPOINTS
                rad = 1.5 * gym.envs.box2d.car_racing.TRACK_RAD
            checkpoints.append((alpha, rad * math.cos(alpha), rad * math.sin(alpha)))

        # Go from one checkpoint to another to create track
        x, y, beta = 1.5 * gym.envs.box2d.car_racing.TRACK_RAD, 0, 0
        dest_i = 0
        laps = 0
        track = []
        no_freeze = 2500
        visited_other_side = False
        while True:
            alpha = math.atan2(y, x)
            if visited_other_side and alpha > 0:
                laps += 1
                visited_other_side = False
            if alpha < 0:
                visited_other_side = True
                alpha += 2 * math.pi
            while True:  # Find destination from checkpoints
                failed = True
                while True:
                    dest_alpha, dest_x, dest_y = checkpoints[dest_i % len(checkpoints)]
                    if alpha <= dest_alpha:
                        failed = False
                        break
                    dest_i += 1
                    if dest_i % len(checkpoints) == 0:
                        break
                if not failed:
                    break
                alpha -= 2 * math.pi
                continue
            r1x = math.cos(beta)
            r1y = math.sin(beta)
            p1x = -r1y
            p1y = r1x
            dest_dx = dest_x - x  # vector towards destination
            dest_dy = dest_y - y
            proj = r1x * dest_dx + r1y * dest_dy  # destination vector projected on rad
            while beta - alpha > 1.5 * math.pi:
                beta -= 2 * math.pi
            while beta - alpha < -1.5 * math.pi:
                beta += 2 * math.pi
            prev_beta = beta
            proj *= gym.envs.box2d.car_racing.SCALE
            if proj > 0.3:
                beta -= min(
                    gym.envs.box2d.car_racing.TRACK_TURN_RATE, abs(0.001 * proj)
                )
            if proj < -0.3:
                beta += min(
                    gym.envs.box2d.car_racing.TRACK_TURN_RATE, abs(0.001 * proj)
                )
            x += p1x * gym.envs.box2d.car_racing.TRACK_DETAIL_STEP
            y += p1y * gym.envs.box2d.car_racing.TRACK_DETAIL_STEP
            track.append((alpha, prev_beta * 0.5 + beta * 0.5, x, y))
            if laps > 4:
                break
            no_freeze -= 1
            if no_freeze == 0:
                break

        # Find closed loop range i1..i2, first loop should be ignored, second is OK
        i1, i2 = -1, -1
        i = len(track)
        while True:
            i -= 1
            if i == 0:
                return False  # Failed
            pass_through_start = (
                track[i][0] > self.start_alpha and track[i - 1][0] <= self.start_alpha
            )
            if pass_through_start and i2 == -1:
                i2 = i
            elif pass_through_start and i1 == -1:
                i1 = i
                break
        if self.verbose == 1:
            print("Track generation: %i..%i -> %i-tiles track" % (i1, i2, i2 - i1))
        assert i1 != -1
        assert i2 != -1

        track = track[i1 : i2 - 1]

        first_beta = track[0][1]
        first_perp_x = math.cos(first_beta)
        first_perp_y = math.sin(first_beta)
        # Length of perpendicular jump to put together head and tail
        well_glued_together = np.sqrt(
            np.square(first_perp_x * (track[0][2] - track[-1][2]))
            + np.square(first_perp_y * (track[0][3] - track[-1][3]))
        )
        if well_glued_together > gym.envs.box2d.car_racing.TRACK_DETAIL_STEP:
            return False

        self.track = track
        return True

    def create_tiles(self):
        # Red-white border on hard turns
        border = [False] * len(self.track)
        for i in range(len(self.track)):
            good = True
            oneside = 0
            for neg in range(gym.envs.box2d.car_racing.BORDER_MIN_COUNT):
                beta1 = self.track[i - neg - 0][1]
                beta2 = self.track[i - neg - 1][1]
                good &= (
                    abs(beta1 - beta2) > gym.envs.box2d.car_racing.TRACK_TURN_RATE * 0.2
                )
                oneside += np.sign(beta1 - beta2)
            good &= abs(oneside) == gym.envs.box2d.car_racing.BORDER_MIN_COUNT
            border[i] = good
        for i in range(len(self.track)):
            for neg in range(gym.envs.box2d.car_racing.BORDER_MIN_COUNT):
                border[i - neg] |= border[i]

        self.road = []
        self.road_poly = []
        self.basic_road_poly = []
        # Create tiles
        for i in range(len(self.track)):
            alpha1, beta1, x1, y1 = self.track[i]
            alpha2, beta2, x2, y2 = self.track[i - 1]
            road1_l = (
                x1 - gym.envs.box2d.car_racing.TRACK_WIDTH * math.cos(beta1),
                y1 - gym.envs.box2d.car_racing.TRACK_WIDTH * math.sin(beta1),
            )
            road1_r = (
                x1 + gym.envs.box2d.car_racing.TRACK_WIDTH * math.cos(beta1),
                y1 + gym.envs.box2d.car_racing.TRACK_WIDTH * math.sin(beta1),
            )
            road2_l = (
                x2 - gym.envs.box2d.car_racing.TRACK_WIDTH * math.cos(beta2),
                y2 - gym.envs.box2d.car_racing.TRACK_WIDTH * math.sin(beta2),
            )
            road2_r = (
                x2 + gym.envs.box2d.car_racing.TRACK_WIDTH * math.cos(beta2),
                y2 + gym.envs.box2d.car_racing.TRACK_WIDTH * math.sin(beta2),
            )
            vertices = [road1_l, road1_r, road2_r, road2_l]
            self.fd_tile.shape.vertices = vertices
            t = self.world.CreateStaticBody(fixtures=self.fd_tile)
            t.userData = t
            c = 0.01  # *(i%3)
            t.color = [
                gym.envs.box2d.car_racing.ROAD_COLOR[0] + c,
                gym.envs.box2d.car_racing.ROAD_COLOR[1] + c,
                gym.envs.box2d.car_racing.ROAD_COLOR[2] + c,
            ]
            t.road_visited = False
            t.road_friction = 1.0
            t.fixtures[0].sensor = True
            self.road_poly.append(([road1_l, road1_r, road2_r, road2_l], t.color))
            self.basic_road_poly.append(
                ([road1_l, road1_r, road2_r, road2_l], [255, 255, 255])
            )
            self.road.append(t)
            if border[i]:
                side = np.sign(beta2 - beta1)
                b1_l = (
                    x1 + side * gym.envs.box2d.car_racing.TRACK_WIDTH * math.cos(beta1),
                    y1 + side * gym.envs.box2d.car_racing.TRACK_WIDTH * math.sin(beta1),
                )
                b1_r = (
                    x1
                    + side
                    * (
                        gym.envs.box2d.car_racing.TRACK_WIDTH
                        + gym.envs.box2d.car_racing.BORDER
                    )
                    * math.cos(beta1),
                    y1
                    + side
                    * (
                        gym.envs.box2d.car_racing.TRACK_WIDTH
                        + gym.envs.box2d.car_racing.BORDER
                    )
                    * math.sin(beta1),
                )
                b2_l = (
                    x2 + side * gym.envs.box2d.car_racing.TRACK_WIDTH * math.cos(beta2),
                    y2 + side * gym.envs.box2d.car_racing.TRACK_WIDTH * math.sin(beta2),
                )
                b2_r = (
                    x2
                    + side
                    * (
                        gym.envs.box2d.car_racing.TRACK_WIDTH
                        + gym.envs.box2d.car_racing.BORDER
                    )
                    * math.cos(beta2),
                    y2
                    + side
                    * (
                        gym.envs.box2d.car_racing.TRACK_WIDTH
                        + gym.envs.box2d.car_racing.BORDER
                    )
                    * math.sin(beta2),
                )
                self.road_poly.append(
                    ([b1_l, b1_r, b2_r, b2_l], (1, 1, 1) if i % 2 == 0 else (1, 0, 0))
                )

    def load_road(self, basic):
        vertices = []
        colors = []
        if not basic:
            current_color = (0.4, 0.8, 0.4, 1.0)
            colors.append(current_color)
            vertices.append(
                (
                    -gym.envs.box2d.car_racing.PLAYFIELD,
                    +gym.envs.box2d.car_racing.PLAYFIELD,
                    0,
                )
            )
            colors.append(current_color)
            vertices.append(
                (
                    +gym.envs.box2d.car_racing.PLAYFIELD,
                    +gym.envs.box2d.car_racing.PLAYFIELD,
                    0,
                )
            )
            colors.append(current_color)
            vertices.append(
                (
                    +gym.envs.box2d.car_racing.PLAYFIELD,
                    -gym.envs.box2d.car_racing.PLAYFIELD,
                    0,
                )
            )
            colors.append(current_color)
            vertices.append(
                (
                    -gym.envs.box2d.car_racing.PLAYFIELD,
                    -gym.envs.box2d.car_racing.PLAYFIELD,
                    0,
                )
            )

            current_color = (0.4, 0.9, 0.4, 1.0)
            k = gym.envs.box2d.car_racing.PLAYFIELD / 20.0
            for x in range(-20, 20, 2):
                for y in range(-20, 20, 2):
                    vertices.append((k * x + k, k * y + 0, 0))
                    colors.append(current_color)
                    vertices.append((k * x + 0, k * y + 0, 0))
                    colors.append(current_color)
                    vertices.append((k * x + 0, k * y + k, 0))
                    colors.append(current_color)
                    vertices.append((k * x + k, k * y + k, 0))
                    colors.append(current_color)

        for poly, color in self.basic_road_poly if basic else self.road_poly:
            for p in poly:
                colors.append((color[0], color[1], color[2], 1))
                vertices.append((p[0], p[1], 0))

        return pyglet.graphics.vertex_list(
            len(vertices),
            ("v3f", tuple(e for v in vertices for e in v)),
            ("c4f", tuple(e for c in colors for e in c)),
        )

    def render_road(self, basic=False):
        if self.road_vertex_list is None:
            self.road_vertex_list = self.load_road(False)
            self.basic_road_vertex_list = self.load_road(True)
        if basic:
            self.basic_road_vertex_list.draw(gl.GL_QUADS)
        else:
            self.road_vertex_list.draw(gl.GL_QUADS)

    def draw_crosshair(self, img, x, y):
        img[x, :, 0] = 255
        img[:, y, 0] = 255
        self.imshow(img)

    def crop_current(self, car):
        rot_mat = R.from_euler("z", +car.hull.angle, degrees=False)
        rot_car_x, rot_car_y, _ = rot_mat.apply(
            [car.hull.position[1], car.hull.position[0], 0]
        )
        car_x = 500 - int(rot_car_x)
        car_y = 500 + int(rot_car_y)

        img = self.track_imgs[int(math.degrees(-car.hull.angle)) % 360]

        crop = img[
            car_x - 24 : car_x + 8,
            car_y - 16 : car_y + 16,
            :,
        ]

        if self.render_view:
            self.pyscreen.update(crop)

        return crop

    def imshow(self, img):
        plt.imshow(img)
        plt.show()

    def render_whole_track(self):
        if self.viewer is None:
            self.setup_viewer()

        self.transform.set_scale(1, 1)
        self.transform.set_translation(
            gym.envs.box2d.car_racing.WINDOW_W / 2,
            gym.envs.box2d.car_racing.WINDOW_H / 2,
        )

        imgs = []

        for i in tqdm(range(360), desc="Generating track images"):
            self.transform.set_rotation(math.radians(i))
            win = self.viewer.window
            win.switch_to()
            win.dispatch_events()

            win.clear()
            t = self.transform
            pixel_scale = 1
            if hasattr(win.context, "_nscontext"):
                pixel_scale = (
                    win.context._nscontext.view().backingScaleFactor()
                )  # pylint: disable=protected-access
            VP_W = int(pixel_scale * gym.envs.box2d.car_racing.WINDOW_W)
            VP_H = int(pixel_scale * gym.envs.box2d.car_racing.WINDOW_H)

            gl.glViewport(0, 0, VP_W, VP_H)
            t.enable()
            self.render_road(basic=True)
            t.disable()

            gl.glFinish()

            image_data = (
                pyglet.image.get_buffer_manager().get_color_buffer().get_image_data()
            )
            arr = np.fromstring(image_data.get_data(), dtype=np.uint8, sep="")
            arr = arr.reshape(VP_H, VP_W, 4)
            arr = arr[::-1, :, 0:3]
            imgs.append(arr)

        return imgs

    def step(self, *actions):
        assert len(actions) == len(self.cars)
        self.last_action = actions[self.focus_car]

        for car, action in zip(self.cars, actions):
            if action is not None:
                car.steer(-action[0])
                car.gas(action[1])
                car.brake(action[2])

            car.step(1.0 / gym.envs.box2d.car_racing.FPS)

        self.states = [self.get_state(car) for car in self.cars]

        self.world.Step(1.0 / gym.envs.box2d.car_racing.FPS, 6 * 30, 2 * 30)
        self.t += 1.0 / gym.envs.box2d.car_racing.FPS

        step_reward = 0
        done = False

        # calculate reward
        focus_car = self.cars[self.focus_car]

        if (
            actions[self.focus_car] is not None
        ):  # First step without action, called from reset()
            tiles_of_wheels = {
                tile for wheel in focus_car.wheels for tile in wheel.tiles
            }

            if all(t.road_friction != 1.0 for t in tiles_of_wheels):
                # penalty if the car goes off the road
                self.reward -= 10
            else:
                self.reward -= 0.1

            step_reward = self.reward - self.prev_reward
            self.prev_reward = self.reward
            if self.tile_visited_count == len(self.track):
                done = True
            x, y = focus_car.hull.position
            if (
                abs(x) > gym.envs.box2d.car_racing.PLAYFIELD
                or abs(y) > gym.envs.box2d.car_racing.PLAYFIELD
            ):
                done = True
                step_reward = -100
        self.max_reward = max(self.reward, self.max_reward)

        return self.states, step_reward, done, {}

    def render_indicators(self, W, H):
        focus_car = self.cars[self.focus_car]
        gl.glBegin(gl.GL_QUADS)
        s = W / 40.0
        h = H / 40.0
        gl.glColor4f(0, 0, 0, 1)
        gl.glVertex3f(W, 0, 0)
        gl.glVertex3f(W, 5 * h, 0)
        gl.glVertex3f(0, 5 * h, 0)
        gl.glVertex3f(0, 0, 0)

        def vertical_ind(place, val, color):
            gl.glColor4f(color[0], color[1], color[2], 1)
            gl.glVertex3f((place + 0) * s, h + h * val, 0)
            gl.glVertex3f((place + 1) * s, h + h * val, 0)
            gl.glVertex3f((place + 1) * s, h, 0)
            gl.glVertex3f((place + 0) * s, h, 0)

        def horiz_ind(place, val, color):
            gl.glColor4f(color[0], color[1], color[2], 1)
            gl.glVertex3f((place + 0) * s, 4 * h, 0)
            gl.glVertex3f((place + val) * s, 4 * h, 0)
            gl.glVertex3f((place + val) * s, 2 * h, 0)
            gl.glVertex3f((place + 0) * s, 2 * h, 0)

        true_speed = np.sqrt(
            np.square(focus_car.hull.linearVelocity[0])
            + np.square(focus_car.hull.linearVelocity[1])
        )
        vertical_ind(5, 0.02 * true_speed, (1, 1, 1))

        if self.last_action is not None:
            vertical_ind(8, 2.5 * self.last_action[1], (0, 1, 0))  # gas
            vertical_ind(9, 2.5 * self.last_action[2], (1, 0, 0))  # brake

        horiz_ind(20, -10.0 * focus_car.wheels[0].joint.angle, (0, 1, 0))
        horiz_ind(30, -0.8 * focus_car.hull.angularVelocity, (1, 0, 0))
        gl.glEnd()
        self.score_label.text = "R: %04i" % self.reward
        self.score_label.draw()

        if self.last_action is not None:
            self.gas_label.text = "G:{0:.2f}".format(self.last_action[1])
            self.brake_label.text = "B:{0:.2f}".format(self.last_action[2])

        self.gas_label.draw()
        self.brake_label.draw()

    def setup_viewer(self):
        from gym.envs.classic_control import rendering

        self.viewer = rendering.Viewer(
            gym.envs.box2d.car_racing.WINDOW_W, gym.envs.box2d.car_racing.WINDOW_H
        )
        self.score_label = pyglet.text.Label(
            "0000",
            font_size=36,
            x=20,
            y=gym.envs.box2d.car_racing.WINDOW_H * 2.5 / 40.00,
            anchor_x="left",
            anchor_y="center",
            color=(255, 255, 255, 255),
        )
        self.gas_label = pyglet.text.Label(
            "0000",
            font_size=12,
            x=300,
            y=gym.envs.box2d.car_racing.WINDOW_H * 2.5 / 40.00,
            anchor_x="left",
            anchor_y="center",
            color=(255, 255, 255, 255),
        )
        self.brake_label = pyglet.text.Label(
            "0000",
            font_size=12,
            x=300,
            y=gym.envs.box2d.car_racing.WINDOW_H * 2.5 / 40.00 - 20,
            anchor_x="left",
            anchor_y="center",
            color=(255, 255, 255, 255),
        )
        self.transform = rendering.Transform()

    def render(self, mode="human", store_frames=False):
        if self.headless:
            return
        assert mode in ["human", "state_pixels", "rgb_array"]
        assert mode == "human"
        if self.viewer is None:
            self.setup_viewer()

        if "t" not in self.__dict__:
            return  # reset() not called yet

        # zoom = 0.1 * SCALE * max(1 - self.t, 0) + ZOOM * SCALE * min(self.t, 1)  # Animate zoom first second
        zoom = gym.envs.box2d.car_racing.ZOOM * gym.envs.box2d.car_racing.SCALE
        focus_car = self.cars[self.focus_car]
        scroll_x = focus_car.hull.position[0]
        scroll_y = focus_car.hull.position[1]
        angle = -focus_car.hull.angle
        vel = focus_car.hull.linearVelocity
        if np.linalg.norm(vel) > 0.5:
            angle = math.atan2(vel[0], vel[1])
        self.transform.set_scale(zoom, zoom)
        self.transform.set_translation(
            gym.envs.box2d.car_racing.WINDOW_W / 2
            - (scroll_x * zoom * math.cos(angle) - scroll_y * zoom * math.sin(angle)),
            gym.envs.box2d.car_racing.WINDOW_H / 4
            - (scroll_x * zoom * math.sin(angle) + scroll_y * zoom * math.cos(angle)),
        )
        self.transform.set_rotation(angle)

        for car in self.cars:
            car.draw(self.viewer, mode != "state_pixels")

        arr = None
        win = self.viewer.window
        win.switch_to()
        win.dispatch_events()

        win.clear()
        t = self.transform
        if mode == "rgb_array":
            VP_W = gym.envs.box2d.car_racing.VIDEO_W
            VP_H = gym.envs.box2d.car_racing.VIDEO_H
        elif mode == "state_pixels":
            VP_W = gym.envs.box2d.car_racing.STATE_W
            VP_H = gym.envs.box2d.car_racing.STATE_H
        else:
            pixel_scale = 1
            if hasattr(win.context, "_nscontext"):
                pixel_scale = (
                    win.context._nscontext.view().backingScaleFactor()
                )  # pylint: disable=protected-access
            VP_W = int(pixel_scale * gym.envs.box2d.car_racing.WINDOW_W)
            VP_H = int(pixel_scale * gym.envs.box2d.car_racing.WINDOW_H)

        gl.glViewport(0, 0, VP_W, VP_H)
        t.enable()
        self.render_road()
        for geom in self.viewer.onetime_geoms:
            geom.render()
        self.viewer.onetime_geoms = []
        t.disable()
        self.render_indicators(
            gym.envs.box2d.car_racing.WINDOW_W, gym.envs.box2d.car_racing.WINDOW_H
        )

        if store_frames:
            pyglet.image.get_buffer_manager().get_color_buffer().save(
                "tmp/frames/" + str(self.frame) + ".png"
            )
            self.frame += 1

        win.flip()
        return self.viewer.isopen

    def get_state(self, car):
        state = self.crop_current(car)
        vectors = (
            []
        )  # DO NOT CHANGE THE ORDER OF THE INPUTS. car_racing_env.feature_names relies on it

        if self.enable_linear_speed:
            linear_speed = np.sqrt(
                np.square(car.hull.linearVelocity[0])
                + np.square(car.hull.linearVelocity[1])
            )
            print(linear_speed)
            vectors.append(np.array([linear_speed]))

        if self.enable_angular_speed:
            vectors.append(np.array([car.hull.angularVelocity]))

        if self.enable_abs:
            vectors.append(np.array([car.wheels[i].omega for i in range(4)]))

        if self.enable_steering:
            vectors.append(np.array([car.wheels[0].joint.angle]))

        image = state[:, :, 0] / 255
        image = image.astype(np.float32)
        return (
            image.reshape(1, 1, image.shape[0], image.shape[0]),
            np.concatenate(vectors, axis=0).astype(np.float32),
        )

    def reset(self, regen_track, num_cars=None):
        """
        Reset the environment. If num_cars is provided, you can change the amount of cars in the environment.
        :param regen_track: whether to generate a new track
        :param num_cars: if not None, the amount of cars will be changed to this value
        :return: the result of the first step (with action None)
        """
        self._destroy()
        self.max_reward = -100000000000000
        self.reward = 0.0
        self.prev_reward = 0.0
        self.frame = 0
        self.tile_visited_count = 0
        self.spin_reward = 0
        self.t = 0.0
        if num_cars is not None:
            self.num_cars = num_cars

        if regen_track:
            while True:
                success = self._create_track()
                if success:
                    break
                if self.verbose == 1:
                    print(
                        "retry to generate track (normal if there are not many instances of this message)"
                    )
            self.create_tiles()
            self.track_imgs = self.render_whole_track()
            self.road_vertex_list = None
        else:
            self.create_tiles()

        if self.cars is not None:
            for car in self.cars:
                car.destroy()

        self.cars = tuple(
            Car(self.world, *self.track[0][1:4]) for _ in range(self.num_cars)
        )

        # just a list of random colors
        colors = [
            (0.8, 0.0, 0.0),
            (0.0, 0.0, 0.8),
            (0.0, 0.8, 0.0),
            (0.8, 0.8, 0.0),
            (0.0, 0.8, 0.8),
            (0.8, 0.0, 0.8),
        ]
        for color, car in zip(colors, self.cars):
            car.hull.color = color
        # any more cars will just keep the default color, which is red

        return self.step(*[None for _ in range(self.num_cars)])[0]

    def _destroy(self):
        if not self.road:
            return
        for t in self.road:
            self.world.DestroyBody(t)
        self.road = []


if __name__ == "__main__":
    from pyglet.window import key

    a = np.array([0.0, 0.0, 0.0])

    def key_press(k, mod):
        global restart
        if k == 0xFF0D:
            restart = True
        if k == key.LEFT:
            a[0] = -1.0
        if k == key.RIGHT:
            a[0] = +1.0
        if k == key.UP:
            a[1] = +1.0
        if k == key.DOWN:
            a[2] = +0.8  # set 1.0 for wheels to block to zero rotation

    def key_release(k, mod):
        if k == key.LEFT and a[0] == -1.0:
            a[0] = 0
        if k == key.RIGHT and a[0] == +1.0:
            a[0] = 0
        if k == key.UP:
            a[1] = 0
        if k == key.DOWN:
            a[2] = 0

    env = CarRacingWrapper(True, True, True, True, headless=False)
    env.render()
    env.viewer.window.on_key_press = key_press
    env.viewer.window.on_key_release = key_release
    record_video = False
    if record_video:
        from gym.wrappers.monitor import Monitor

        env = Monitor(env, "/tmp/video-test", force=True)
    isopen = True
    while isopen:
        total_reward = 0.0
        steps = 0
        restart = False
        while True:
            s, r, done, info = env.step(a)
            total_reward += r
            steps += 1
            isopen = env.render()
    env.close()
