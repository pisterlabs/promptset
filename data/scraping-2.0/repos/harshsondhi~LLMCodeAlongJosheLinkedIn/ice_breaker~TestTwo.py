from langchain.llms import OpenAI
import os
import openai

os.environ["OPENAI_API_KEY"] = "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"
openai.api_key = "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"

# response = openai.Completion.create(
#            model='text-davinci-003',
#            prompt='Give me two reasons to learn OpenAI API with python',
#            max_tokens=300)
#
#
# print(response['choices'][0]['text'])
llm = OpenAI()
# print(llm('Here is a fun fact about Pluto:'))

result = llm.generate(["Here is the fact about pluto:", "here is the fact about mars:"])
# print(result.schema())
print(result.llm_output)

print(result.generations[1][0].text)
