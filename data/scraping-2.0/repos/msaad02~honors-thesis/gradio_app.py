from text_search.text_retriever_class import TextRetriever, TypesenseRetriever
import time
import gradio as gr
from openai import OpenAI
from termcolor import colored

client = OpenAI()

# text_retriever = TextRetriever(
#     main_categorization_model_dir="./text_search/models/main_category_model",
#     subcategorization_model_dir="./text_search/models/subcategory_models/",
#     embeddings_file="./text_search/data/embeddings.pickle",
# )

text_retriever = TypesenseRetriever(
    main_categorization_model_dir="./text_search/models/main_category_model",
    subcategorization_model_dir="./text_search/models/subcategory_models/",
)

system = """You are a helpful chatbot for SUNY Brockport who answers questions using the context given. Be enthusiastic, straightforward, and brief in your responses. Do not answer questions unrelated to SUNY Brockport. If the answer is not clear from the context, say "I'm sorry, I don't know"."""

prompt = lambda context, question: f"Context: {context}\n\nQuestion: {question}"

def rag(question, history, search_method, max_results):
    context = text_retriever.retrieve(question, top_n=max_results)

    print(colored(question, 'green'))
    print(colored(context, 'yellow'))
    print("-"*150)
    
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": prompt(context, question)},
        ],
        temperature=1,
        stream=True
    )

    message = ""
    for chunk in response:
        content = chunk.choices[0].delta.content
        if content is not None:
            message += content
            yield message


chatbot = gr.Chatbot(
    height="68vh",
    avatar_images=("./user.png", "./BrockportGPT Logo.png"),
    bubble_full_width=False
)

demo = gr.ChatInterface(
    rag, 
    chatbot=chatbot,
    additional_inputs=[
        gr.Dropdown([
            "Classifier - Semantic - Reranking",
            "Classifier - Typesense",
            "Semantic"
        ], value="Classifier - Semantic - Reranking", label="Search Method"),
        gr.Slider(2, 6, render=False)
    ]
).launch(show_api=False, inbrowser=True)