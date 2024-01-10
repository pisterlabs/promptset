from langchain.chat_models import ChatOpenAI
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.prompts.chat import ChatPromptTemplate

template = "You are a helpful assistant that extract the {column} given the data `{data}`"
human_template = "What is the age of {name}"

chat_prompt = ChatPromptTemplate.from_messages([
    ("system", template), # role
    ("human", human_template), # content
])

loader = CSVLoader(file_path='./csv_sample.csv')
data = loader.load()

text_list = []
for record in data:
    text_list.append(record.page_content)

chat_prompt_output = chat_prompt.format_messages(
                        column="age",
                        data=("\n".join(text_list)),
                        name="Minh Barrett")
print(chat_prompt_output)

chat_model = ChatOpenAI()
print(chat_model(chat_prompt_output))
