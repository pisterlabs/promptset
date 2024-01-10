# --8<-- [start:import]
import promptogen as pg
# --8<-- [end:import]

# --8<-- [start:summarizer]
summarizer = pg.Prompt(
    name="Text Summarizer and Keyword Extractor",
    description="Summarize text and extract keywords.",
    input_parameters=[
        pg.ParameterInfo(name="text", description="Text to summarize"),
    ],
    # --8<-- [start:output_parameters]
    output_parameters=[
        pg.ParameterInfo(name="summary", description="Summary of text"),
        pg.ParameterInfo(name="keywords", description="Keywords extracted from text"),
    ],
    # --8<-- [end:output_parameters]
    template=pg.IOExample(
        input={'text': "This is a sample text to summarize."},
        output={
            'summary': "This is a summary of the text.",
            'keywords': ["sample", "text", "summarize"],
        },
    ),
    examples=[
        pg.IOExample(
            input={
                'text': "One sunny afternoon, a group of friends decided to gather at the nearby park to engage in various games and activities. They played soccer, badminton, and basketball, laughing and enjoying each other's company while creating unforgettable memories together."},
            output={
                'summary': "A group of friends enjoyed an afternoon playing sports and making memories at a local park.",
                'keywords': ["friends", "park", "sports", "memories"],
            },
        )
    ],
)
# --8<-- [end:summarizer]

# --8<-- [start:summarizer_omit]
summarizer = pg.Prompt(
    name="Text Summarizer and Keyword Extractor",
    # ...(other parameters omitted)...
)
# --8<-- [end:summarizer_omit]

# --8<-- [start:format_prompt_without_input]
formatter = pg.KeyValuePromptFormatter()
print(formatter.format_prompt_without_input(summarizer))
# --8<-- [end:format_prompt_without_input]


# --8<-- [start:format_prompt]
input_value = {
    'text': "In the realm of software engineering, developers often collaborate on projects using version control systems like Git. They work together to create and maintain well-structured, efficient code, and tackle issues that arise from implementation complexities, evolving user requirements, and system optimization.",
}
print(formatter.format_prompt(summarizer, input_value))
# --8<-- [end:format_prompt]


# --8<-- [start:text_llm]
import openai
import os

openai.api_key = os.getenv("OPENAI_API_KEY")
openai.organization = os.getenv("OPENAI_ORG_ID")


def generate_chat_completion(text: str, model: str) -> str:
    resp = openai.ChatCompletion.create(
        model=model,
        messages=[{"role": "user", "content": text}],
        max_tokens=2048,
        stream=True,
    )
    raw_resp = ""
    for chunk in resp:
        chunk_content = chunk["choices"][0]["delta"].get("content", "")
        raw_resp += chunk_content

    return raw_resp


text_llm = pg.FunctionBasedTextLLM(
    generate_text_by_text=lambda input_text: generate_chat_completion(input_text, "gpt-3.5-turbo"),
)
# --8<-- [end:text_llm]

# --8<-- [start:text_llm_omit]
# ...(omitted)...

text_llm = pg.FunctionBasedTextLLM(
    # ...(omitted)...
)
# --8<-- [end:text_llm_omit]

# --8<-- [start:generate]
raw_req = formatter.format_prompt(summarizer, input_value)
print(raw_req)

raw_resp = text_llm.generate(raw_req)
print(raw_resp)
# --8<-- [end:generate]

# --8<-- [start:parse]
summarized_resp = formatter.parse(summarizer, raw_resp)
print(summarized_resp)
# --8<-- [end:parse]
