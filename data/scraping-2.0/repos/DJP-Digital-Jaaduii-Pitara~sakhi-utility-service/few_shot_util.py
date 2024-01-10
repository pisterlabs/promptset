import outlines
import os

from openai import AzureOpenAI


@outlines.prompt
def few_shots(instructions, examples, question):
    """{{ instructions }}

    Examples
    --------
       {% for item_group, items in examples.items() %}
            {% if items %}
                {% for item in items %}
                    Q: {{ item.question }}
                    A: {{ item.answer }}
                    {% if not loop.last %}

                    {% endif %}
                {% endfor %}
            {% endif %}
       {% endfor %}
    Question
    --------

    Q: {{ question }}
    A:
    """


client = AzureOpenAI(
    azure_endpoint=os.environ["OPENAI_API_BASE"],
    api_key=os.environ["OPENAI_API_KEY"],
    api_version=os.environ["OPENAI_API_VERSION"]
)
gpt_model = os.environ["GPT_MODEL"]


def invokeLLM(instructions, examples, question):
    prompt = few_shots(instructions, examples, question)
    print("PROMPT::: ", prompt)

    res = client.chat.completions.create(
        model=gpt_model,
        temperature=0,
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": question}
        ],
    )

    return res.choices[0].message.model_dump()
