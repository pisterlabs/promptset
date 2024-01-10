import asyncio

from flask import Blueprint, jsonify, request
from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    AIMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.schema import AIMessage, HumanMessage, SystemMessage
from config import chat_model as chat_model
from flask_login import login_required


ask_gpt = Blueprint("ask_gpt", __name__)


@ask_gpt.route("/ask-gpt", methods=["POST"])
@login_required
def ask_gpt_function():
    prompts = [{"title": "askGPTSection", "prompt": request.form.get("section-prompt")}]
    output_dict = {}

    for prompt in prompts:
        # asyncio.run creates a new event loop and runs the coroutine until it's done
        prompt, result = asyncio.run(
            send_question(prompt)
        )  # pass qa to ask_gpt function
        output_dict[prompt] = result
    print(output_dict)
    return jsonify(output_dict), 200


async def send_question(prompt_obj):  # receive qa as an argument
    try:
        # Let's run the blocking function in a separate thread using asyncio.to_thread
        chat = ChatOpenAI(model=chat_model, temperature=0.7)
        print(prompt_obj["prompt"])
        messages = [
            SystemMessage(
                content="Answer with a formal and knowledgable language. Add extensive relevant content, provide implementation solutions and details for the text provided."
            ),
            HumanMessage(content=prompt_obj["prompt"]),
        ]
        response = chat(messages)
        print(response)
    except Exception as e:
        response = None
        print(e)
    finally:
        response_content = response.content
        print(
            prompt_obj["title"],
            {
                "question": prompt_obj["prompt"],
                "response": response,
            },
        )
        return prompt_obj["title"], {
            "question": prompt_obj["prompt"],
            "response": response_content,
        }
