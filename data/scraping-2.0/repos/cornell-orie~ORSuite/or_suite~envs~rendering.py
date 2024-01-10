import numpy as np
import pyglet
from pyglet.gl import *

__author__ = 'Nicolas Dickreuter'
"""
This code is adapted from Nicolas Dickreuter's render file for an OpenAI gym environment
https://github.com/dickreuter/neuron_poker/blob/master/gym_env/rendering.py
"""


WHITE = (255, 255, 255)
RED = (255, 0, 0)
GREEN = (0, 255, 0)


class PygletWindow:
    """Rendering class"""

    def __init__(self, X, Y):
        """Initialization"""
        self.active = True
        self.window = pyglet.window.Window(width=X, height=Y + 50)
        self.top = Y

        # make OpenGL context current
        self.window.switch_to()
        self.reset()

    def circle(self, x, y, radius, color):
        """Draw a circle"""
        y = self.top - y
        circle = pyglet.shapes.Circle(x, y, radius, color=color)
        circle.draw()

    def text(self, text, x, y, font_size=20):
        """Draw text"""
        y = self.top - y
        label = pyglet.text.Label(text, font_size=font_size,
                                  x=x, y=y, anchor_x='left', anchor_y='top')
        label.draw()

    def line(self, x1, x2, y, width, color):
        y = self.top - y
        line = pyglet.shapes.Line(x1, y, x2, y, width, color)
        line.draw()

    def image(self, x, y, image, scale):
        y = self.top - y
        image.anchor_x = image.width // 2
        image.anchor_y = image.height // 2
        sprite = pyglet.sprite.Sprite(image, x, y)
        sprite.scale = scale
        sprite.draw()

    def reset(self):
        """New frame"""
        pyglet.clock.tick()
        self.window.dispatch_events()
        glClear(pyglet.gl.GL_COLOR_BUFFER_BIT)

    """ render function adapted from OpenAI gym rendering.py """
    def render(self, return_rgb_array=False):
        # glClearColor(1,1,1,1)
        # self.window.clear()
        # self.window.switch_to()
        # self.window.dispatch_events()
        # self.transform.enable()
        # for geom in self.geoms:
        #     geom.render()
        # for geom in self.onetime_geoms:
        #     geom.render()
        # self.transform.disable()

        # pyglet.clock.tick()
        # self.window.dispatch_events()
        # glClear(pyglet.gl.GL_COLOR_BUFFER_BIT)
        self.window.flip()
        arr = None
        if return_rgb_array:
            buffer = pyglet.image.get_buffer_manager().get_color_buffer()
            image_data = buffer.get_image_data()
            arr = np.frombuffer(image_data.get_data(), dtype=np.uint8)
            # In https://github.com/openai/gym-http-api/issues/2, we
            # discovered that someone using Xmonad on Arch was having
            # a window of size 598 x 398, though a 600 x 400 window
            # was requested. (Guess Xmonad was preserving a pixel for
            # the boundary.) So we use the buffer height/width rather
            # than the requested one.
            arr = arr.reshape(buffer.height, buffer.width, 4)
            arr = arr[::-1,:,0:3]
        
        # self.onetime_geoms = []
        return arr if return_rgb_array else self.isopen

    def update(self):
        """Draw the current state on screen"""
        self.window.flip()

    def close(self):
        self.window.close()


# if __name__ == '__main__':
#     pg = PygletWindow(400, 400)

#     pg.reset()
#     pg.circle(5, 5, 100, 1, 5)
#     pg.text("Test", 10, 10)
#     pg.text("Test2", 30, 30)
#     pg.update()
#     input()

#     pg.circle(5, 5, 100, 1, 5)
#     pg.text("Test3333", 10, 10)
#     pg.text("Test2123123", 303, 30)
#     pg.update()
#     input()