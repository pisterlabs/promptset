import gradio as gr
import time
import pandas as pd
import os
from dotenv import load_dotenv, find_dotenv

from typing import Literal
from loguru import logger
from modules.swarmGPT import GPTSwarm, SwarmConfig, OAIModels
import openai
load_dotenv(find_dotenv())


def get_model_output(base_chat_history, iteration_history, instruction, system_messages, temperature, token_length, feedback_rounds,
                     model_for_system_messages, model_for_answer_generation, model_for_answer_iteration, model_for_synthesizing):
    # Log the configuration settings
    config = SwarmConfig(
        num_agents=system_messages,
        max_response_tokens=token_length,
        iterations=feedback_rounds,
        system_message_model=OAIModels(model_for_system_messages),
        answer_generation_model=OAIModels(model_for_answer_generation),
        answer_iteration_model=OAIModels(model_for_answer_iteration),
        response_aggregation_model=OAIModels(model_for_synthesizing),
        temperature=temperature,
         # Assuming API key is stored in an environment variable
    )
    logger.info(config)
    swarm = GPTSwarm(instruction, config=config)
    # exit()
    synthesized_response, iterations = swarm.generate_response()

    # Appending iterations to iteration_history
    for iter_num, iter_data in iterations.items():
        for k, v in iter_data.items():
            formatted_message = f"Iteration {iter_num} - {k}: {v}"
            iteration_history.append(("", formatted_message.replace("\n", "<br>")))

    # Original logic
    base_chat_history.append((f"USER:{instruction}", f"AI:{synthesized_response}"))
    return base_chat_history, iteration_history


def create_chat_interface(config: SwarmConfig):
    CSS ="""
    .contain { display: flex; flex-direction: column; }
    .gradio-container { height: 100vh !important; }
    #component-0 { height: 100%; }
    #component-1 { height: 100%; }
    #iteration_chat {height: 100% !important; overflow: auto;}
    #base_chat {height: 100% !important; overflow: auto;}
    #chat_column {height: 95vh; overflow: auto;}
    #output_column {height: 95vh; overflow: auto;}
    #iter_column {height: 95vh; overflow: auto;}
    """
    with gr.Blocks(css=CSS) as chat_interface:
        with gr.Row():
            with gr.Column():
                gr.Markdown("## Settings")
                with gr.Row():
                    with gr.Column():
                    # Config settings widgets
                        system_messages_slider = gr.Slider(minimum=1, maximum=10, step=1, value=config.num_agents, label="How many system messages")
                        token_length_slider = gr.Slider(minimum=100, maximum=16000, value=config.max_response_tokens, label="Response Token Length")
                        feedback_rounds_slider = gr.Slider(minimum=1, maximum=5, step=1, value=config.iterations, label="Feedback Rounds")
                        temperature = gr.Slider(minimum=0, maximum=1.2, step=.1, value=config.temperature, label="Temperature")
                    with gr.Column():
                        model_for_system_messages_dropdown = gr.Dropdown(list(OAIModels), label="Model for System Messages", value=config.system_message_model.value)
                        model_for_answer_generation_dropdown = gr.Dropdown(list(OAIModels), label="Model for Answer Generation", value=config.answer_generation_model.value)
                        model_for_answer_iteration_dropdown = gr.Dropdown(list(OAIModels), label="Model for Answer Iteration", value=config.answer_iteration_model.value)
                        model_for_synthesizing_dropdown = gr.Dropdown(list(OAIModels), label="Model for Synthesizing Final Response", value=config.response_aggregation_model.value)
                    chat_input = gr.Textbox("Tell me a story about a hydra.")
            with gr.Column(elem_id="chat_column"):
                with gr.Tab("Output", elem_id="output_column"):
                    base_chat_display = gr.Chatbot([], elem_id="base_chat")
                with gr.Tab("Iterations", elem_id="iter_column"):
                    iteration_chat_display = gr.Chatbot([], elem_id="iteration_chat")
        # Assuming get_model_output will make use of these configurations
        chat_input.submit(
            get_model_output,  # This function needs to be modified to accept the new inputs
            inputs=[base_chat_display, iteration_chat_display, chat_input, system_messages_slider, temperature, token_length_slider, feedback_rounds_slider,
                    model_for_system_messages_dropdown, model_for_answer_generation_dropdown, model_for_answer_iteration_dropdown,
                    model_for_synthesizing_dropdown],
            outputs=[base_chat_display, iteration_chat_display],
        )
    return chat_interface

def main():
    logger.info("Calling main")

def init_interface():
    default_config = SwarmConfig()
    chat_interface = create_chat_interface(default_config)
    logger.info("Chat interfaced initialized")
    chat_interface.launch(server_name="0.0.0.0", server_port=8001, inbrowser=True)



def main():
    logger.info("Calling main")
    openai.api_key=os.getenv("OPENAI_API_KEY")
    logger.info("Initializing model")
    init_interface()
    while True:
        time.sleep(0.5)


if __name__ == "__main__":
    main()