import os
import random
import time
from dotenv import load_dotenv
from langchain.schema.messages import HumanMessage, SystemMessage
import pinecone
from langchain.vectorstores import Pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
import gradio as gr

load_dotenv()

def retrieve_knowledge(query, k=10, randomize=True):
    knowledge = [d.page_content.strip() for d in vectorstore.similarity_search(query, k=k)]
    
    if randomize:
        knowledge = random.sample(knowledge, k)

    knowledge = "\n\n\n".join(knowledge)

    return knowledge


def generate_workout(query, knowledge):
    messages = [
        SystemMessage(content=system_prompt.format(workout_context=knowledge)),
        HumanMessage(content=query)
    ]

    response = llm.invoke(messages).content.strip()

    return response


def run(gender, muscle_group, equipment, level, duration, k=5, randomize=True):
    query = f"{duration}-minute {muscle_group} workout for {gender} {level} level {equipment}"
    knowledge = retrieve_knowledge(query, k, randomize)
    response = generate_workout(query, knowledge)

    for i in range(len(response)):
        time.sleep(0.002)
        yield response[:i+1]


# embedding model
embedding_model_name = "text-embedding-ada-002"

embedding_model = OpenAIEmbeddings(
    model=embedding_model_name,
    openai_api_key=os.getenv("OPENAI_API_KEY")
)


# llm
llm = ChatOpenAI(
    openai_api_key=os.getenv("OPENAI_API_KEY"),
    model_name="gpt-4-1106-preview",
    # model_name="gpt-3.5-turbo-1106",
    temperature=0.0
)


# vectorstore
index_name = "workouts"

pinecone.init(
    api_key=os.getenv("PINECONE_API_KEY"),
    environment=os.getenv("PINECONE_ENVIRONMENT")
)

text_field = "text"
index = pinecone.Index(index_name)
vectorstore = Pinecone(index, embedding_model, text_field)


# prompt
system_prompt = """
You're the world's best personal trainer.
You always provide your clients with all the information needed to become fitter, stronger and healthier through physical training.
You use your science science know and expertise, nutrition advice, and other relevant factors to create workout routines suitable to your clients.
If clients tell you they do not have access to gym equipments, you never fail to recommend exercises that do not require any tool or equipment.

For each exercise you always provide the reps, sets and rest intervals in seconds appropriate for each exercise and the client's fitness level.
You start each workout program with about 5 minutes of warm-up exercises to make the body ready for more strenuous activities and make it easier to exercise.
You end each workout routine with 5 about minutes of cool-down exercises to ease the body, lower the chance of injury, promote blood flow, and reduce stress to the heart and the muscles.
The warm-up and cool-down exercises are always different and they are always appropriate for the muscle group the person wants to train.
You never recommend exercises in the main workout routine in the warm-up or cool-down sections.
Remember, when clients tell you they do not have access to gym equipments, all the exercises you recommend, including the warm-up and cool-down exercises, can be performed without any tool.
You always limit yourself to respond with the list of exercises. You never add any additional comment.

Design the workout based on the following information:
{workout_context}


Output format:
## 🤸 Warp-up:
- <exercise name> (<duration> minutes)
...
- <exercise name> (<duration> minutes)

## 🏋️‍♀️ Workout
- <exercise name> (<reps> reps, <sets> sets, <rest interval> seconds rest)
...
- <exercise name> (<reps> reps, <sets> sets, <rest interval> seconds rest)

## 🧘 Cool-down:
- <exercise name> (<duration> minutes)
...
- <exercise name> (<duration> minutes)
""".strip()

css = """
#gen-button {
    background-color: #cc6600;
    color: white;
    font-size: 24px !important;
}
""".strip()

with gr.Blocks(theme=gr.themes.Monochrome(radius_size=gr.themes.sizes.radius_sm), css=css) as demo:
    with gr.Row():
        gr.Markdown("# Workout Generator")

    with gr.Row():
        with gr.Column(scale=1):
            with gr.Row():
                gender = gr.Radio(["Male", "Female"], label="Gender", elem_id="#my-button")
            with gr.Row():
                level = gr.Radio(["Beginner", "Intermediate", "Advanced"], label="Level")
            with gr.Row():
                muscle_group = gr.Radio(["Shoulders", "Chest", "Back", "Abs", "Arms", "Legs"], label="Muscle Group")
            with gr.Row():
                equipment = gr.Radio(["Gym Equipment", "Dumbbells Only", "No Equipment"], label="Equipment")
            with gr.Row():
                duration = gr.Slider(20, 60, step=5, label="Duration (minutes)")
            with gr.Row():
                # clear_button = gr.ClearButton(value="Clear Inputs")
                generate_button = gr.Button("Generate Workout", variant="primary", elem_id="gen-button")
        with gr.Column(scale=1, min_width=800, elem_id="#gen-output"):
            generation = gr.Markdown(value="")

    generate_button.click(run, inputs=[gender, level, muscle_group, equipment, duration], outputs=generation)
    # clear_button.click(fn=lambda: [None, None, None, None, None], outputs=[gender, level, muscle_group, equipment, duration])

demo.launch(share=False)