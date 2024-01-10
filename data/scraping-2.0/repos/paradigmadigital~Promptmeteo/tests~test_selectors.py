import pytest

from langchain.embeddings import FakeEmbeddings

from promptmeteo.selector import SelectorTypes
from promptmeteo.selector import SelectorFactory
from promptmeteo.selector.base import SelectorAlgorithms
from promptmeteo.selector.base import BaseSelectorSupervised
from promptmeteo.selector.base import BaseSelectorUnsupervised


class TestSelectors:
    def test_selector_factory(self):
        for selector_type in SelectorTypes:
            for selector_algorithm in SelectorAlgorithms:
                SelectorFactory.factory_method(
                    language="es",
                    selector_k=1,
                    selector_type=selector_type.value,
                    selector_algorithm=selector_algorithm.value,
                    embeddings=FakeEmbeddings(size=64),
                )

        with pytest.raises(ValueError) as error:
            SelectorFactory.factory_method(
                language="es",
                selector_k=1,
                selector_type="WRONG_SELECTOR_TYPE",
                selector_algorithm=selector_algorithm.value,
                embeddings=FakeEmbeddings(size=64),
            )

        assert error.value.args[0] == (
            f"`SelectorFactory` error in `factory_method()` . "
            f"WRONG_SELECTOR_TYPE is not in the list of supported "
            f"providers: {[i.value for i in SelectorTypes]}"
        )

        with pytest.raises(ValueError) as error:
            selector = SelectorFactory.factory_method(
                language="es",
                selector_k=1,
                selector_type=selector_type.value,
                selector_algorithm="WRONG_ALGORITHM_NAME",
                embeddings=FakeEmbeddings(size=64),
            )

            assert error.value.args[0] == (
                f"{selector.__class__.__name__} error in __init__. "
                f"`selector_algorithm` value `WRONG_ALGORITHM_NAME` is not in "
                f"the available values: {[i.value for i in SelectorAlgorithms]}"
            )

    def test_supervised_selector(self):
        for selector_algorithm in SelectorAlgorithms:
            selector = BaseSelectorSupervised(
                language="es",
                embeddings=FakeEmbeddings(size=64),
                selector_k=1,
                selector_algorithm=selector_algorithm,
            )

            assert selector.selector is not None

            selector.train(
                examples=["TEST_EXAMPLE"], annotations=["TEST_ANNOTATION"]
            )

            assert (
                selector.template
                == """
                EJEMPLO: TEST_EXAMPLE
                RESPUESTA: TEST_ANNOTATION

                EJEMPLO: {__INPUT__}
                RESPUESTA: """.replace(
                    " " * 4, ""
                )[
                    1:
                ]
            )
