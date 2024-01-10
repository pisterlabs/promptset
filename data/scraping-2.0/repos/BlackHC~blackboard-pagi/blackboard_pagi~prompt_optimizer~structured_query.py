#  Blackboard-PAGI - LLM Proto-AGI using the Blackboard Pattern
#  Copyright (c) 2023. Andreas Kirsch
#
#  This program is free software: you can redistribute it and/or modify
#  it under the terms of the GNU Affero General Public License as published
#  by the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU Affero General Public License for more details.
#
#  You should have received a copy of the GNU Affero General Public License
#  along with this program.  If not, see <https://www.gnu.org/licenses/>.

import typing
from typing import Tuple

from langchain.output_parsers import PydanticOutputParser
from langchain.schema import OutputParserException
from pydantic import create_model

from blackboard_pagi.prompt_optimizer.track_hyperparameters import Hyperparameter, track_hyperparameters
from blackboard_pagi.prompts.chat_chain import ChatChain

T = typing.TypeVar("T")


@track_hyperparameters
def structured_query(chain: ChatChain, question: str, return_type: type[T]) -> Tuple[T, ChatChain]:
    """Asks a question and returns the result in a single block."""
    # TOOD: deduplicate
    if typing.get_origin(return_type) is typing.Annotated:
        return_info = typing.get_args(return_type)
    else:
        return_info = (return_type, ...)

    output_model = create_model("StructuredOutput", result=return_info)  # type: ignore
    parser = PydanticOutputParser(pydantic_object=output_model)
    question_and_formatting = question + "\n\n" + parser.get_format_instructions()

    num_retries = Hyperparameter("num_retries_on_parser_failure") @ 3
    prompt = question_and_formatting
    for _ in range(num_retries):
        try:
            reply_content, chain = chain.query(prompt)
            parsed_reply = parser.parse(reply_content)
            break
        except OutputParserException as e:
            prompt = (
                Hyperparameter("error_prompt") @ "Tried to parse your last output but failed:\n\n"
                + str(e)
                + Hyperparameter("retry_prompt") @ "\n\nPlease try again and avoid this issue."
            )
    else:
        raise OutputParserException(f"Failed to parse output after {num_retries} retries.")

    result = parsed_reply.result  # type: ignore
    return result, chain
