from langchain.llms import GPT4All
llm = GPT4All(
    model=(r"C:/Users/zitou/.cache/gpt4all/nous-hermes-13b.ggmlv3.q4_0.bin"))

llm("The first man on the moon was ... Let's think step by step")
