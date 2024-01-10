from openai import OpenAI
#import openai
#client = OpenAI()
client = OpenAI(api_key="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")
#client.api_key = ("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")
#OpenAI.api_key.encode.__get__ = ("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")
client.models.list()
print(client.models.list())