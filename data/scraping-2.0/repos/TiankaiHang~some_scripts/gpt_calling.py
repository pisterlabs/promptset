import openai
import time


def call_openai_completion(engine, messages):
    while True:
        try:
            response = openai.ChatCompletion.create(
                engine=engine,
                messages=messages,
            )
            return response
        except openai.error.RateLimitError:
            print("RateLimitError, retrying...")
            time.sleep(5)
            continue
        except Exception as e:
            print("Error:", e)
            exit(-1)


def main():
    llm_engine = "gpt-4"
    prompt = """I want you to help me make the paragraph shorter. Here is a paragraph of the paper (in latex), please polish and compress it without changing the meaning or losing the details. {paragraph}"""

    paragraph = r"""To enhance the diversity of instructions, we first manually write 10 instructions for each task. Then we use GPT-4 to rewrite and expand the diversity of these instructions, thereby mimicking user input to the system. Subsequently, one instruction is chosen at random during the training process.
"""

    prompt = prompt.format(
        paragraph=paragraph
    )

    messages=[
        {"role": "system", "content": "You are a senior researcher on computer vision."},
        {"role": "user", "content": prompt}
    ]
    response = call_openai_completion(engine=llm_engine, messages=messages)
    response_message = response["choices"][0]["message"]
    response_message = response_message.to_dict()["content"]

    print(paragraph)
    print()
    print(response_message)


if __name__ == '__main__':
    main()
