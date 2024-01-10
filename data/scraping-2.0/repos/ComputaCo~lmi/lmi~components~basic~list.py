from enum import Enum, auto
import itertools
import math
import attr
from abc import abstractmethod

import roman
from langchain.schema import HumanMessage
from yarl import cached_property

from lmi.components.abstract.component import Component
from lmi.components.basic.text import Text


@attr.s(auto_attribs=True)
class List(Component):
    class BulletStyle(Enum):
        NUMBERS = auto()
        ROMAN_NUMERALS = auto()
        LOWERCASE_LETTERS = auto()
        CAPITAL_LETTERS = auto()
        BULLETS = auto()
        DOTS = auto()
        DASHES = auto()

        @staticmethod
        def __iter__(self):
            if self is List.BulletStyle.NUMBERS:
                return iter(iter(range(math.inf)))
            elif self is List.BulletStyle.ROMAN_NUMERALS:
                return iter(map(roman.toRoman, range(math.inf)))
            elif self is List.BulletStyle.LOWERCASE_LETTERS:

                def to_alpha(n):
                    if n < 26:
                        return chr(n + ord("a"))
                    else:
                        return to_alpha(n // 26 - 1) + to_alpha(n % 26)

                return iter(map(to_alpha, range(math.inf)))
            elif self is List.BulletStyle.CAPITAL_LETTERS:

                def to_Alpha(n):
                    if n < 26:
                        return chr(n + ord("A"))
                    else:
                        return to_Alpha(n // 26 - 1) + to_Alpha(n % 26)

                return iter(map(to_Alpha, range(math.inf)))

            elif self is List.BulletStyle.BULLETS:
                return iter(itertools.cycle(["•"]))
            elif self is List.BulletStyle.DOTS:
                return iter(itertools.cycle(["·"]))
            elif self is List.BulletStyle.DASHES:
                return iter(itertools.cycle(["-"]))

            else:
                raise ValueError("Invalid NumberingStyle")

    bullet_style: BulletStyle = BulletStyle.NUMBERS
    format_string = r"{bullet}. {item}\n"

    items: list[Component] = []

    @cached_property
    def children(self):
        return [
            Text(
                text=self.format_string.format(
                    bullet=next(iter(self.bullet_style)),
                    item=item.render_to_text(),
                ),
                parent=self,
                name=f"{i}",
            )
            for i, item in enumerate(self.items)
        ]
