import dotenv

dotenv.load_dotenv()

from chatgpt import ChatGPT
import traceback
import openai

models = [
    "gpt-4-1106-preview",
    "gpt-4",
    "gpt-3.5-turbo-1106",
    "gpt-3.5-turbo",
]
for model in models:
    chat = ChatGPT(model)
    print(model)
    try:
        response = chat.chatgpt("Tell me a joke in one short sentence.")
    except KeyboardInterrupt:
        pass
    except openai.InternalServerError as e:
        print("Error", e)
    except Exception as e:
        traceback.print_exc()