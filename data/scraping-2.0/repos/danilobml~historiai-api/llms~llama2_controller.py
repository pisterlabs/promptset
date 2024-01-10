from langchain.llms import HuggingFaceHub
import dotenv

dotenv.load_dotenv()

llm = HuggingFaceHub(repo_id="meta-llama/Llama-2-70b-chat")


def send_prompt_to_llama2_70(prompt):
    response = llm(prompt)

    return response
