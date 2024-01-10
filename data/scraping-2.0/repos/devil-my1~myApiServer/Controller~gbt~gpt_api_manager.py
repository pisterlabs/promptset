import openai


MY_API_KEY = "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"
openai.api_key = MY_API_KEY
openai.Model.list()
