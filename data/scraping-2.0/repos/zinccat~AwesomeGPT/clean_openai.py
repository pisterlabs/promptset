from openai import OpenAI
from key import OPENAI_API_KEY

client = OpenAI(api_key=OPENAI_API_KEY)

my_assistants = client.beta.assistants.list(
    order="desc",
)

for assistant in my_assistants.data:
    client.beta.assistants.delete(assistant.id)

files = client.files.list()
for file in files.data:
    client.files.delete(file.id)

# deleted_assistant_file = client.beta.assistants.files.delete(
#     assistant_id="asst_jnwfJ8PZXwzcHX1D4pj9u1w6",
#     file_id="file-kICKW2xHQ7lpnRe8HPSktvoE"
# )

# response = client.beta.assistants.delete("asst_jnwfJ8PZXwzcHX1D4pj9u1w6")
# print(response)