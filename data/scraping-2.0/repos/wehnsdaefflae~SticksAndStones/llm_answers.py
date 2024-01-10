# coding=utf-8
import json
from dataclasses import dataclass
from typing import Generator

import openai
import tiktoken


def segment_text(text: str, segment_length: int = 500, overlap: int = 100, truncation_sign: str = "[...]") -> Generator[str, None, None]:
    len_t = len(text)
    if segment_length >= len_t:
        yield text
        return

    len_s = len(truncation_sign)
    cursor = 0
    while cursor < len_t:
        if cursor < 1:
            cursor_end = cursor + segment_length - len_s
            new_cursor = cursor_end - overlap
            if cursor >= new_cursor:
                raise ValueError("Segment length is too short.")
            segment = f"{text[cursor:cursor_end]}{truncation_sign}"
            cursor = new_cursor
            yield segment

        elif cursor + segment_length < len_t:
            cursor_end = cursor + segment_length - 2 * len_s
            new_cursor = cursor_end - overlap
            if cursor >= new_cursor:
                raise ValueError("Segment length is too short.")
            segment = f"{truncation_sign}{text[cursor:cursor_end]}{truncation_sign}"
            cursor = new_cursor
            yield segment

        else:
            segment = f"{truncation_sign}{text[cursor:]}"
            yield segment
            return


def get_max_tokens(model_name: str) -> int:
    model_tokens_mapping = {
        "gpt-4": 8_192,
        "gpt-4-0613": 8_192,
        "gpt-4-32k": 32_768,
        "gpt-4-32k-0613": 32_768,
        "gpt-4-0314": 8_192,
        "gpt-4-32k-0314": 32_768,
        "gpt-3.5-turbo": 4_097,
        "gpt-3.5-turbo-16k": 16_385,
        "gpt-3.5-turbo-0613": 4_097,
        "gpt-3.5-turbo-16k-0613": 16_385,
        "gpt-3.5-turbo-0301": 4_097,
        "text-davinci-003": 4_097,
        "text-davinci-002": 4_097,
        "code-davinci-002": 8_001,
        "babbage-002": 16_384,
        "davinci-002": 16_384
    }

    return model_tokens_mapping[model_name]


def make_element(content: str | None, _tag_name: str) -> str:
    if content is None:
        return ""

    return (
        f"<{_tag_name}>\n"
        f"{indent(content).rstrip()}\n"
        f"</{_tag_name}>\n"
        f"\n"
    )


def get_token_len(messages: list[dict[str, str]], model_name: str) -> int:
    encoding = tiktoken.encoding_for_model(model_name)
    messages_json = json.dumps(messages)
    tokenized_prompt = encoding.encode(messages_json)
    len_tokenized_prompt = len(tokenized_prompt)
    return len_tokenized_prompt


def indent(text: str, indent_str: str = "    ") -> str:
    return "\n".join(f"{indent_str}{line}" for line in text.splitlines())


@dataclass(frozen=True)
class ChatResponse:
    output: str
    summary: str

    def __str__(self) -> str:
        return f"Response(output='{self.output}', summary='{self.summary}')"


def _summarize_prompt(
        content: str,
        context: str | None,
        additional_instruction: str | None,
        _content_tag: str, _context_tag: str) -> str:

    if context is None:
        instruction = f"Summarize the text in the outermost `{_content_tag}` tag. Do not mention the tag."

    else:
        instruction = (
            f"Summarize the text in the outermost `{_content_tag}` tag as a seamless continuation "
            f"of the text in the outermost `{_context_tag}` tag. Do not mention the tags."
        )

    if additional_instruction is not None:
        instruction += f" {additional_instruction.strip()}"

    prompt = (
            make_element(context, _context_tag) +
            make_element(content, _content_tag) +
            instruction
    )
    return prompt


def summarize(
        content: str,
        *args: any,
        context: str | None = None,
        additional_instruction: str | None = None,
        max_input_ratio: float = .7,
        segment_length: int = 2_000,
        _margin: float = .1,
        _content_tag: str = "Content",
        _context_tag: str = "Context",
        **kwargs: any) -> str:

    model_name = kwargs["model"]
    max_tokens = get_max_tokens(model_name)

    summarize_prompt = _summarize_prompt(content, context, additional_instruction, _content_tag, _context_tag)
    messages = [{"role": "user", "content": summarize_prompt}]
    len_tokenized_prompt = get_token_len(messages, model_name) * (1. + _margin)

    if max_input_ratio >= len_tokenized_prompt / max_tokens:
        openai.api_key_path = "openai_api_key.txt"
        response_message = openai.ChatCompletion.create(*args, messages=messages, **kwargs)
        first_choice, = response_message.choices
        first_message = first_choice.message
        output = first_message.content
        return output

    print("segmenting...")
    rolling_summary = None
    summaries = list()
    segments = list(segment_text(content, segment_length=segment_length))
    for i, each_segment in enumerate(segments):
        print(f"segment {i + 1} of {len(segments)} segments")
        each_summary = summarize(
            each_segment,
            *args,
            context=rolling_summary,
            additional_instruction=additional_instruction,
            max_input_ratio=max_input_ratio,
            segment_length=segment_length,
            _margin=_margin,
            _content_tag=_content_tag,
            _context_tag=_context_tag,
            **kwargs
        )

        if rolling_summary is None:
            rolling_summary = each_summary

        else:
            rolling_summary = summarize(
                f"{rolling_summary.strip()}\n{each_summary.strip()}",
                *args,
                additional_instruction=additional_instruction,
                max_input_ratio=max_input_ratio,
                segment_length=segment_length,
                _margin=_margin,
                _content_tag=_content_tag,
                _context_tag=_context_tag,
                **kwargs
            )

        summaries.append(each_summary.strip())

    return "\n\n".join(summaries)


# output debug log


def _response_prompt(
        instruction: str,
        recap: str | None,
        data: str | None,
        _recap_tag: str, _data_tag: str) -> str:

    recap_element = make_element(recap, _recap_tag)
    data_element = make_element(data, _data_tag)

    prompt = (
            recap_element +
            data_element +
            instruction.rstrip() + f" (IMPORTANT: Do not imitate the XML syntax from the `<{_recap_tag}>` and `<{_data_tag}>` tag, if present.)"
    )

    return prompt


def _get_pruned_message(
        instruction: str, data: str, recap: str,
        ratio_instruction: float, ratio_data: float, ratio_recap: float, ratio_response: float,
        data_tag: str, recap_tag: str, summary_tag: str,
        margin: float,
        args: tuple[any, ...], kwargs: dict[str, any]) -> list[dict[str, str]]:

    ratio_recap = 0. if recap is None else ratio_recap
    ratio_data = 0. if data is None else ratio_data
    sum_ratios = ratio_instruction + ratio_recap + ratio_data + ratio_response
    ratio_response_target = ratio_response / sum_ratios
    model_name = kwargs["model"]
    max_tokens = get_max_tokens(model_name)
    prompt = _response_prompt(instruction, recap, data, recap_tag, data_tag)
    messages = [{"role": "user", "content": prompt}]
    len_tokenized_prompt = get_token_len(messages, model_name) * (1. + margin)
    sum_input_ratios = ratio_instruction + ratio_recap + ratio_data
    while ratio_response_target < len_tokenized_prompt / max_tokens:
        ratio_instruction_is = len(instruction) / len_tokenized_prompt
        ratio_instruction_delta = ratio_instruction_is - ratio_instruction / sum_input_ratios
        ratio_recap_is = 0. if recap is None else len(recap) / len_tokenized_prompt
        ratio_recap_delta = ratio_recap_is - ratio_recap / sum_input_ratios
        ratio_data_is = 0. if data is None else len(data) / len_tokenized_prompt
        ratio_data_delta = ratio_data_is - ratio_data / sum_input_ratios

        max_delta = max(ratio_instruction_delta, ratio_recap_delta, ratio_data_delta)

        if ratio_instruction_delta == max_delta:
            instruction = summarize(instruction, *args, context=recap, **kwargs)

        elif ratio_data_delta == max_delta:
            focus_instruction = (
                f"Transcribe this into a more concise format. "
                f"Ignore information that is not relevant to the instruction \"{instruction.strip()}\""
            )
            data = summarize(data, *args, context=recap, additional_instructions=focus_instruction, **kwargs)

        else:
            focus_conversation = "Be very concise but preserve literal information and conversational character."
            recap_text = summarize(recap, *args, additional_instruction=focus_conversation, **kwargs)
            recap = make_element(recap_text, summary_tag)

        prompt = _response_prompt(instruction, recap, data, recap_tag, data_tag)
        messages = [{"role": "user", "content": prompt}]
        len_tokenized_prompt = get_token_len(messages, model_name) * (1. + margin)

    return messages


def _make_recap(instruction: str, response: str, data: str | None = None, recap: str | None = None, data_tag: str = "AdditionalData") -> str:
    recap = (
            (f"" if recap is None else recap) +
            make_element(
                (f"" if data is None else make_element(data.rstrip(), data_tag)) +
                make_element(instruction.rstrip(), "Instruction"),
                "UserRequest") +
            make_element(response.rstrip(), "AssistantResponse")
    )
    return recap


def respond(
        instruction: str, *args: any,
        data: str | None = None,
        recap: str | None = None,
        ratio_instruction: float = .1,
        ratio_recap: float = .3,
        ratio_data: float = .3,
        ratio_response: float = .3,
        _margin: float = .1,
        _recap_tag: str = "ConversationLog",
        _summary_tag: str = "ConversationSummary",
        _data_tag: str = "AdditionalData",
        **kwargs: any) -> ChatResponse:

    messages = _get_pruned_message(
        instruction, data, recap,
        ratio_instruction, ratio_data, ratio_recap, ratio_response,
        _data_tag, _recap_tag, _summary_tag, _margin, args, kwargs)

    openai.api_key_path = "openai_api_key.txt"

    response_message = openai.ChatCompletion.create(*args, messages=messages, **kwargs)
    first_choice, = response_message.choices
    first_message = first_choice.message

    output = first_message.content
    updated_recap_content = _make_recap(instruction, output, data, recap, _data_tag)
    return ChatResponse(output, updated_recap_content)


def respond_stream(
        instruction: str, *args: any,
        data: str | None = None,
        recap: str | None = None,
        ratio_instruction: float = .1,
        ratio_recap: float = .3,
        ratio_data: float = .3,
        ratio_response: float = .3,
        _margin: float = .1,
        _recap_tag: str = "ConversationLog",
        _summary_tag: str = "ConversationSummary",
        _data_tag: str = "AdditionalData",
        **kwargs: any) -> Generator[str, None, str]:

    messages = _get_pruned_message(
        instruction, data, recap,
        ratio_instruction, ratio_data, ratio_recap, ratio_response,
        _data_tag, _recap_tag, _summary_tag, _margin, args, kwargs)

    full_output = list()
    for chunk in openai.ChatCompletion.create(*args, messages=messages, stream=True, **kwargs):
        content = chunk["choices"][0].get("delta", {}).get("content")
        if content is not None:
            full_output.append(content)
            yield content  # Yielding the content for each chunk

    output = "".join(full_output)
    recap = _make_recap(instruction, output, data, recap, _data_tag)
    return recap


def default() -> None:
    response = respond(
        "Tell me about your day.",
        model="gpt-3.5-turbo")

    print(response)


def stream() -> None:
    chunks = respond_stream("Tell me about your day.", recap="", model="gpt-3.5-turbo")
    try:
        while True:
            each_chunk = next(chunks)
            print(each_chunk, end="")

    except StopIteration as e:
        final_recap = e.value

        print(final_recap)


def main() -> None:
    # default()
    stream()


if __name__ == '__main__':
    main()
