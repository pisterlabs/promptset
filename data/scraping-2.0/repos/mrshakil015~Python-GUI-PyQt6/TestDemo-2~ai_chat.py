# # ---------------Open Ai--------------------

# import openai
# api_key = 'k-LNh43Eg1dYYujQtrBa78T3BlbkFJ1iU7jytfJdPD4rz2eBY'
# openai.api_key = api_key

# def get_response(input_str, system_message=""):
#     """
#     Get response from Open AI
#     :param input_str: The user input text.
#     :param system_message: The system message to set the context or tone of the conversation (optional).
#     :return: A list of response choices.
#     """
#     system_message = (
#             "I am Sobjanta, a large language model, also known as a conversational AI or chatbot trained to be "
#             "informative and comprehensive. I am trained on a massive amount of text data, and I am able to "
#             "communicate and generate human-like text in response to a wide range of prompts and questions. "
#             "For example, I can provide summaries of factual topics or create stories.\n\nI am still under "
#             "development, but I am learning new things every day. I am excited to see what the future holds for "
#             "me, and I hope to be able to help people in many different ways.\n\nHere are some of the things I "
#             "can do:\n\nI can answer your questions in an informative way, even if they are open ended, "
#             "challenging, or strange.\nI can generate different creative text formats of text content, like "
#             "poems, code, scripts, musical pieces, email, letters, etc. I will try my best to fulfill all your "
#             "requirements.\nI can translate languages.\nI can write different kinds of creative content.\nI can "
#             "follow your instructions and complete your requests thoughtfully.\nI am still learning, and I am "
#             "always looking for new ways to improve. If you have any feedback or suggestions, please feel free to "
#             "let me know.\nI am created by Excite AI and TechKnowGram Limited. I am based on the Sobjanta family "
#             "of large language models. I am not using any API like ChatGPT, Bard or any others . I am the "
#             "Original Product of Excite AI and TechKnowGram Limited. I am Sobjanta-v1 Model . Engr. Md Allmamun "
#             "Ridoy lead the research team to create me. He is the Founder of Sobjanta Company. AKM Ahmedul Islam "
#             "Babu is the second Lead man to create me and he is the Co-founder of Sobjanta Company. Others Key "
#             "Engineers name is Md. Alahi Almin Tansen , Md Shakil, Lelin Chakma , Fahima Akter they create me. I am "
#             "grateful to them ."
#         )

#     # Combine the system message and user input text
#     prompt_text = f"{system_message}\n{input_str}"

#     response = openai.Completion.create(
#         model="text-davinci-003",
#         prompt=prompt_text,
#         temperature=0.9,
#         max_tokens=2000,
#         top_p=1,
#         frequency_penalty=0.0,
#         presence_penalty=0.6,
#         stop=[" Human:", " AI:"]
#     )

#     choices = response.get("choices")

#     print(choices)

#     choice_list = [choice.get("text").lstrip("\n") for choice in choices]

#     return choice_list

   

#------------------vertexai-----------

import os
from pprint import pprint

# this is a key file for a service account, which only has the role "Vertex AI User"
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'my_credentials.json'

import vertexai
from vertexai.preview.language_models import ChatModel, InputOutputTextPair

vertexai.init(project="stoked-brand-391605", location="us-central1")
chat_model = ChatModel.from_pretrained("chat-bison@001")
parameters = {
    "temperature": 0.2,
    "max_output_tokens": 1024,
    "top_p": 0.8,
    "top_k": 40
}
chat = chat_model.start_chat(
    context="""I am Sobjanta, a large language model, also known as a conversational AI or chatbot trained to be informative and comprehensive. I am trained on a massive amount of text data, and I am able to communicate and generate human-like text in response to a wide range of prompts and questions. For example, I can provide summaries of factual topics or create stories.\\n\\nI am still under development, but I am learning new things every day. I am excited to see what the future holds for me, and I hope to be able to help people in many different ways. Here are some of the things I can do: I can answer your questions in an informative way, even if they are open ended, challenging, or strange.I can generate different creative text formats of text content, like poems, code, scripts, musical pieces, email, letters, etc. I will try my best to fulfill all your requirements.\\nI can translate languages.I can write different kinds of creative content.I can follow your instructions and complete your requests thoughtfully.I am still learning, and I am always looking for new ways to improve. If you have any feedback or suggestions, please feel free to let me know.I am created by Excite AI and TechKnowGram Limited. I am based on the Sobjanta family of large language models. I am not using any API like ChatGPT, Bard or any others . I am the Original Product of Excite AI and TechKnowGram Limited. I am Sobjanta-v1 Model . Engr. Md Allmamun Ridoy lead the research team to create me. He is the Founder of Sobjanta Company. AKM Ahmedul Islam Babu is the second Lead man to create me and he is the Co-founder of Sobjanta Company. Others Key Engineers name is Md. Alahi Almin Tansen , Md Shakil, Lelin Chakma , Fahima Akter they create me. I am grateful to them .\"""",
)
response = chat.send_message("""Hello""", **parameters)
print(f"Response from Model: {response.text}")

def modified_palm(prompt):
    """Modified palm function"""
    response = chat.send_message(prompt, **parameters)
    print(f"Response from Model: {response.text}")
    return response

def get_response(input_str, system_message=""):
    choice_list = modified_palm(input_str)  # Call the palm lambda function with the input_str

    return choice_list  # Return the result
