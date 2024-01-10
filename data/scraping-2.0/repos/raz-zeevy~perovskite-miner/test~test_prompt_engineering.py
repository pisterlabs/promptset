import pytest

from prompt_engineering.prompt_engineering_consts import preview_prompt

from prompt_engineering.prompt_engineering import (
    read_pdf,
    clean_text,
    PaperPrompt,
    truncation
)

from apis.gpt_api import openai_count_tokens

questions = ["What is the meaning of life?", "Can you summarize the methodology?", "This is a question?"]
paper_pdf_path = "pdf_mock.pdf"
expected_pdf_text = clean_text(read_pdf(paper_pdf_path))


def test_read_pdf():
    with pytest.raises(Exception):
        read_pdf("non_existing_file.pdf")

    with pytest.raises(Exception):
        read_pdf("test_prompt_engineering.py")


def test_clean_text():
    text = read_pdf('pdf_mock.pdf')
    cleaned_text = clean_text(text)
    with(open('expected_clean_text.txt', 'r')) as f:
        expected_clean_text = f.read()
    assert cleaned_text == expected_clean_text, "Unexpected text after cleaning"


def test_paper_prompt_generate_prompts_short():
    paper_prompt = PaperPrompt(paper_pdf_path=paper_pdf_path, questions=questions, max_tokens=15000,
                               answers_max_tokens=500)

    assert paper_prompt.number_of_api_calls == 1, "Unexpected number of API calls"
    assert paper_prompt.questions_per_api_call == (3, 0), "Unexpected number of questions per API call"
    assert paper_prompt.tokens_per_api_call == [5188], "Unexpected number of tokens per API call"
    assert len(paper_prompt.contents) == paper_prompt.number_of_api_calls, "Unexpected number of content generated"
    assert paper_prompt.contents[0][0] == preview_prompt, "Unexpected preview prompt content"
    assert paper_prompt.contents[0][1] == expected_pdf_text, "Unexpected paper prompt content"
    assert paper_prompt.contents[0][2] == 'What is the meaning of life? Can you summarize the methodology? ' \
                                          'This is a question?', "Unexpected questions prompt content"


def test_paper_prompt_generate_prompts_long():
    paper_prompt = PaperPrompt(paper_pdf_path=paper_pdf_path, questions=questions, max_tokens=6000,
                               answers_max_tokens=1000)

    assert paper_prompt.number_of_api_calls == 2, "Unexpected number of API calls"

    assert len(paper_prompt.contents) == paper_prompt.number_of_api_calls, "Unexpected number of content generated"
    assert paper_prompt.contents[0][0] == preview_prompt, "Unexpected preview prompt content"
    assert paper_prompt.contents[0][1] == expected_pdf_text, "Unexpected paper prompt content"
    assert paper_prompt.contents[0][2] == "What is the meaning of life? Can you summarize the methodology?", \
        "Unexpected questions prompt content"

    assert paper_prompt.contents[1][0] == preview_prompt, "Unexpected preview prompt content"
    assert paper_prompt.contents[1][1] == paper_prompt.contents[0][1], "Unexpected paper prompt content"
    assert paper_prompt.contents[1][2] == "This is a question?", "Unexpected questions prompt content"


def test_paper_prompt_generate_prompts_with_shrinking():
    paper_prompt = PaperPrompt(paper_pdf_path=paper_pdf_path, questions=questions, max_tokens=4000,
                               answers_max_tokens=1500)

    assert paper_prompt.number_of_api_calls == 3, "Unexpected number of API calls"

    with(open('expected_shrank_text.txt', 'r')) as f:
        expected_shrank_text = f.read()

    assert len(paper_prompt.contents) == paper_prompt.number_of_api_calls, "Unexpected number of content generated"

    assert paper_prompt.contents[0][0] == preview_prompt, "Unexpected preview prompt content"
    assert paper_prompt.contents[0][1] == expected_shrank_text, "Unexpected paper prompt content"
    assert paper_prompt.contents[0][2] == "What is the meaning of life?", "Unexpected questions prompt content"

    assert len(paper_prompt.contents) == paper_prompt.number_of_api_calls, "Unexpected number of content generated"
    assert paper_prompt.contents[1][0] == preview_prompt, "Unexpected preview prompt content"
    assert paper_prompt.contents[1][1] == expected_shrank_text, "Unexpected paper prompt content"
    assert paper_prompt.contents[1][2] == "Can you summarize the methodology?", "Unexpected questions prompt content"

    assert len(paper_prompt.contents) == paper_prompt.number_of_api_calls, "Unexpected number of content generated"
    assert paper_prompt.contents[2][0] == preview_prompt, "Unexpected preview prompt content"
    assert paper_prompt.contents[2][1] == expected_shrank_text, "Unexpected paper prompt content"
    assert paper_prompt.contents[2][2] == "This is a question?", "Unexpected questions prompt content"


def test_truncation():
    text = read_pdf('pdf_mock.pdf')
    tokens_for_paper = 1000
    shrank_text = truncation(text, tokens_for_paper)
    tokens_deviation = 100
    assert tokens_for_paper + tokens_deviation > openai_count_tokens(shrank_text) > \
           tokens_for_paper - tokens_deviation, "Unexpected number of tokens after truncation"
    with(open('expected_text_after_truncation.txt', 'r')) as f:
        expected_text = f.read()
    assert shrank_text == expected_text, "Unexpected text after truncation"


if __name__ == "__main__":
    pytest.main([__file__])
