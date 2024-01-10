from langchain.llms import OpenAI

llm = OpenAI()


def llm_string_in_string_out():
    print(llm("Tell me a joke"))


def batch_calls_richer_outputs():
    llm_result = llm.generate(["Tell me a joke", "Tell me a poem"] * 5)
    len(llm_result.generations)
    print(llm_result.generations[0])
    print(llm_result.generations[-1])
    print(llm_result.llm_output)


if __name__ == "__main__":
    # llm_string_in_string_out()
    batch_calls_richer_outputs()
