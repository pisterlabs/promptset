from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import ChatPromptTemplate
import settings

OPEN_AI_API_KEY = settings.OPEN_AI_API_KEY

def main():
    chat_model = ChatOpenAI(openai_api_key=OPEN_AI_API_KEY, model_name="gpt-3.5-turbo")

    template = "You are a helpful assistant that translates {input_language} to {output_language}."
    human_template = "{text}"

    chat_prompt = ChatPromptTemplate.from_messages([
      ("system", template),
      ("human", human_template),
    ])

    format_message = chat_prompt.format_messages(input_language="English", output_language="French", text="I love programming.")

    print(chat_model.invoke(format_message))


if __name__ == "__main__":
    main()