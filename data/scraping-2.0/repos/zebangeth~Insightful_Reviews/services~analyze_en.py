import os

import openai
from anthropic import Anthropic, HUMAN_PROMPT, AI_PROMPT
import streamlit as st

from configs import OPENAI_GPT3, CLAUDE_INSTANT, CLAUDE_2, REVIEW_NUM_CAP, OPENAI_CAP

# --- OpenAI Stream Completion ---
openai.api_key  = st.secrets["OpenAI_API_KEY"]

def gpt_stream_completion(prompt, model=OPENAI_GPT3):
    system_prompt, user_prompt = prompt[0], prompt[1]
    messages = [
        {"role": "system", "content": system_prompt}, 
        {"role": "user", "content": user_prompt}, 
        ]
    
    stream = openai.ChatCompletion.create(
        model=model, 
        messages=messages, 
        temperature=0,
        stream=True)
    
    completing_content = ""
    for chunk in stream:
        chunk_content = chunk["choices"][0].get("delta", {}).get("content")
        if chunk_content: 
            completing_content += chunk_content
            st.markdown(completing_content)
    
    return

# --- Anthropic Completion ---
anthropic = Anthropic(
    api_key=st.secrets["Anthropic_API_KEY"],
)

def claude_stream_completion(prompt, model=CLAUDE_INSTANT): 
    messages = f"{HUMAN_PROMPT} {prompt[2]} {AI_PROMPT}"

    stream = anthropic.completions.create(
        model=model,
        max_tokens_to_sample=90000,
        prompt=messages,
        temperature=0,
        stream=True,
    )

    completing_content = ""
    for chunk in stream:
        chunk_content = chunk.completion
        if chunk_content: 
            completing_content += chunk_content
            st.markdown(completing_content)

    return

# --- Prompt Generation ---
def generate_prompt(prod_info, num_of_reviews, review_texts, user_position, analysis_focus, input_question):
    common_prompt_part1 = f"""
    You are a seasoned e-commerce review analyst.
    Your task is to analyze the most recent {num_of_reviews} product reviews for this {prod_info} product on the Taobao platform.\n
    Points to consider in the analysis:
    """

    common_prompt_part2 = """
    2. Your analysis should include:
      a. Key findings, concisely listed in a summary paragraph;
      b. Detailed analysis, discussing each key finding in depth;
      c. Recommendations for improvement, based on your findings, to be presented at the end of the analysis.\n
    Only provide specific customer review content as evidence for key findings when necessary, and include the date of the review.\n
    3. Please use markdown syntax to present your analysis results.
    """
    
    focus_to_prompt = {
        "⚙️ Product Function": "Comments related to product functional features, such as the main functions, practicality of these functions, and customer reactions to them.",
        "💎 Product Quality": "Comments related to product quality issues, such as durability, consistency, and customer feedback on product quality.",
        "🎨 Product Design": "Comments related to product design, such as aesthetics, color, shape, size, and customer feedback on the design.",
        "🖐️ User Experience": "Comments related to the user experience, such as ease of use, comfort, and problems encountered by customers during use.",
        "💰 Price Reasonableness": "Comments related to the product's price, such as whether the price is reasonable, how it compares to the product's value, and customer feedback on the price.",
        "💳 Customer Service & Ordering Experience": "Comments related to customer service, such as the responsiveness of online service personnel, service quality, professionalism, and convenience of the ordering process, along with other customer feedback.",
        "📦 Packaging & Logistics": "Comments related to packaging and logistics issues, such as whether the product's packaging is intact, the design of the packaging, delivery speed, and customer feedback on packaging and delivery.",
    }

    position_to_prompt = {
        "👨🏻‍💻 E-commerce Operations": "Analyze from the perspective of an e-commerce operations manager, focusing on factors that might affect sales volumes and customer satisfaction, such as product popularity, sales strategies, ordering experience, pricing, as well as customer feedback and suggestions.",
        "🤵🏻‍♂️ Customer Service": "Analyze from the perspective of a customer service manager, focusing on aspects such as response speed of online customer service personnel, service quality, professionalism, and the convenience of the ordering process, along with other customer feedback. Avoid summarizing any comments unrelated to customer service or the ordering experience.",
        "👩🏻‍🔬 Product R&D": "Analyze from the perspective of a product R&D manager, focusing on customer feedback regarding product functions and design, such as functionality, user experience, and product improvement needs, providing a comprehensive summary. Avoid summarizing any comments unrelated to the product and user experience.",
        "👩🏻‍🔧 Production/Quality Control": "Analyze from the perspective of a production and quality control department manager, with a focus on customer feedback related to product quality, such as quality issues and defects.",
        "✈️ Logistics/Supply Chain": "Analyze from the perspective of a logistics and supply chain department manager, focusing on customer feedback regarding product packaging and delivery, such as packaging and logistics issues, delivery speed, and overall logistics experience. Avoid summarizing any comments unrelated to logistics or packaging.",
    }

    if input_question: 
        # generate prompt according to the specific question the user has asked
        system_prompt = f"""
        You are a seasoned e-commerce review analyst.
        Your task is to analyze the most recent {num_of_reviews} product reviews for this {prod_info} product on the Taobao platform.\n
        And through the content of customer reviews, answer the following question: {input_question}
        """
    else: 
        # generate prompt according to the user position and area of interest (focus analysis)
        # if both user position and focus analysis are selected by the user, generate prompt based on focus analysis
        if analysis_focus != "Not selected":
            system_prompt = common_prompt_part1 \
                + """1. The analysis should not be exhaustive but should focus on: a. Firstly, filter out customer comments related to """ + focus_to_prompt[analysis_focus] \
                + "b. Then analyze and summarize the filtered content in detail.\n" + common_prompt_part2
        elif user_position != "Not selected":
            system_prompt = common_prompt_part1 + "1. The analysis should not be exhaustive, please analyze solely from the perspective of a " + position_to_prompt[user_position] + common_prompt_part2
        else:
            system_prompt = common_prompt_part1 + "1. Please categorize and analyze the customer comments from different perspectives comprehensively, such as the main advantages and disadvantages of the product, its functions, design, user experience, pricing, packaging, customer service, quality of the product, and any other issues raised by customers." + common_prompt_part2

    user_prompt = f"\nList of reviews:\n```{review_texts}```"
    complete_prompt = system_prompt + user_prompt

    return [system_prompt, user_prompt, complete_prompt]
