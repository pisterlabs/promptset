from dotenv import load_dotenv
import openai
import os

load_dotenv()
openai.api_key = os.getenv('OPENAI_API_KEY')

#Add .jsonl file to OpenAI
openai.File.create(
    file=open("dataset/COVID-QAOnly_Clean.jsonl"),
    purpose='fine-tune'
)

print(openai.File.list())
openai_file_list = openai.File.list()
fileID = openai_file_list['data'][-1]['id']
print(fileID)

#curl https://api.openai.com/v1/fine-tunes/ft-l7FjXlmwoe0O4f1A14J7HWul -H "Authorization: Bearer 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"
#curl https://api.openai.com/v1/fine-tunes -H 'Authorization: Bearer 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'
