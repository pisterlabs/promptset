from __future__ import annotations

from langchain.chains.base import Chain
from typing import Any, Dict, List, Optional
from pydantic import Extra
from langchain.chains.base import Chain

from re import findall, sub, escape
import base64
from typing import Any, Dict, List, Optional
from langchain.utils.input import get_colored_text

from Crypto.Cipher import AES
from Crypto.Util.Padding import unpad

from langchain.callbacks.manager import RunManager, AsyncRunManager


class DeAnonymizationChain(Chain):
    key: bytes = b""
    output_key: str = "text"
    input_key: str = "text"
    verbose: bool = False

    def __init__(self, key: bytes, verbose: bool = False):
        super().__init__()
        self.key = key
        self.verbose = verbose

    class Config:
        extra = Extra.forbid
        arbitrary_types_allowed = True

    @property
    def input_keys(self) -> List[str]:
        return [self.input_key]

    @property
    def output_keys(self) -> List[str]:
        return [self.output_key]

    def de_anonymize(
        self,
        answer: str,
        run_manager: Optional[RunManager | AsyncRunManager] = None,
    ) -> str:
        if run_manager:
            _colored_text = get_colored_text(answer, "green")
            _text = "Received anonymized answer:\n" + _colored_text
            run_manager.on_text(text=_text, end="\n", verbose=self.verbose)

        matches = findall(r"<([A-Z_]+)-([A-Za-z0-9+/=]*)>", answer)
        de_anonymized = answer
        for entity, anonymized in matches:
            try:
                cipher = AES.new(self.key, AES.MODE_CBC, self.key)
                un_stringified = base64.decodebytes(anonymized.encode("utf-8"))
                decrypted = unpad(
                    cipher.decrypt(un_stringified), AES.block_size
                ).decode("utf-8")

                de_anonymized = sub(
                    escape(f"<{entity}-{anonymized}>"), decrypted, de_anonymized
                )
            except Exception as e:
                if run_manager:
                    _colored_text = get_colored_text(
                        f"An error occured while de-anonymizing '<{entity}-{anonymized}>':\n{e}",
                        "red",
                    )
                    _text = _colored_text
                    run_manager.on_text(text=_text, end="\n", verbose=self.verbose)
                continue

        return de_anonymized

    def _call(
        self,
        inputs: Dict[str, Any],
        run_manager: Optional[RunManager] = None,
    ) -> Dict[str, str]:
        answer = inputs[self.input_key]
        result = self.de_anonymize(answer, run_manager=run_manager)
        return {self.output_key: result}

    async def _acall(
        self,
        inputs: Dict[str, Any],
        run_manager: Optional[AsyncRunManager] = None,
    ) -> Dict[str, str]:
        answer = inputs[self.input_key]
        result = self.de_anonymize(answer, run_manager=run_manager)
        return {self.output_key: result}

    @property
    def _chain_type(self) -> str:
        return "de_anonymization_chain"
