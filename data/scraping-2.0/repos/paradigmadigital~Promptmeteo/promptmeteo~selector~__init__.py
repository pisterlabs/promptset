#!/usr/bin/python3

#  Copyright (c) 2023 Paradigma Digital S.L.

#  Permission is hereby granted, free of charge, to any person obtaining a copy
#  of this software and associated documentation files (the "Software"), to deal
#  in the Software without restriction, including without limitation the rights
#  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#  copies of the Software, and to permit persons to whom the Software is
#  furnished to do so, subject to the following conditions:

#  The above copyright notice and this permission notice shall be included in
#  all copies or substantial portions of the Software.

#  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
#  THE SOFTWARE.

import os
from enum import Enum
from typing import List
from typing import Dict

from langchain.embeddings.base import Embeddings

from .base import BaseSelector
from .base import BaseSelectorSupervised
from .base import BaseSelectorUnsupervised


class SelectorTypes(str, Enum):

    """
    Enum with the avaialable selector types.
    """

    SUPERVISED: str = "supervised"
    UNSUPERVISED: str = "unsupervised"


class SelectorFactory:

    """
    Factory of Selectors
    """

    @classmethod
    def factory_method(
        cls,
        language: str,
        embeddings: Embeddings,
        selector_k: int,
        selector_type: str,
        selector_algorithm: str,
    ) -> BaseSelector:
        """
        Returns and instance of a BaseSelector object depending on the
        `selector_algorithm`.
        """

        if selector_type == SelectorTypes.SUPERVISED.value:
            selector_cls = BaseSelectorSupervised

        elif selector_type == SelectorTypes.UNSUPERVISED.value:
            selector_cls = BaseSelectorUnsupervised

        else:
            raise ValueError(
                f"`{cls.__name__}` error in `factory_method()` . "
                f"{selector_type} is not in the list of supported "
                f"providers: {[i.value for i in SelectorTypes]}"
            )

        return selector_cls(
            language=language,
            embeddings=embeddings,
            selector_k=selector_k,
            selector_algorithm=selector_algorithm,
        )
