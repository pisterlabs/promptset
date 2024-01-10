from learning.llm.llm import AnthropicLlm
from learning.tools.email import email_tool


def main():
    llm = AnthropicLlm()

    result = email_tool.run(
        llm=llm,
        context="I want to send an email to jeromeswannack@gmail.com with a poem about frogs",
    )

    print(result)


if __name__ == "__main__":
    import logging

    logging.basicConfig(level=logging.INFO)
    main()
