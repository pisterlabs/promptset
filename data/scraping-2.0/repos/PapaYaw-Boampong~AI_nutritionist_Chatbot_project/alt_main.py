import streamlit as st
import os
import openai
from openai import OpenAI

# Now you can access your variables
openai_api_key = os.getenv("OPENAI_API_KEY")

client = OpenAI(
    api_key = st.secrets["openai"]["api_key"]
)


# Function to get completion from OpenAI Chat API
def get_completion_from_messages(messages,
                                 model="gpt-3.5-turbo",
                                 temperature=0,
                                 max_tokens=500):
    response = client.chat.completions.create(
        messages=messages,
        model=model,
        max_tokens=max_tokens,
        temperature=0,  # this is the degree of randomness of the model's output
    )
    return response.choices[0].message.content


# Function to categorize user input and extract relevant information
def get_input_category(user_input):
    delimiter = "####"
    system_message = f"""
        You will receive user inquiries related to the AI nutritionist service. \
        Each user query will be delimited with {delimiter} characters.

        Classify each query into a primary category and a secondary category.
        If the query is off-topic, provide the classification as Off-Topic.
        Further Extract keywords like, food titles, recipes, and\
        ingredients if provided and none in no keyword found. Store as a keyword.
        Also return keyword like, food titles, recipes, and\ 

        Provide your output in JSON format with the keys: primary, secondary, and keyword\

        Primary categories: Personalized Meal Planning, Nutritional Information, \
        Dietary Advice, Health Goals, General Inquiry, or Off-Topic.

        Meal Planning secondary categories:
        Generate meal plan
        Modify existing plan
        Preferences update
        Meal substitution

        Nutritional Information secondary categories:
        Specific food nutrition
        General nutrition facts
        Caloric content
        Ingredient analysis

        Dietary Advice secondary categories:
        Healthy eating tips
        Dietary restrictions
        Balanced diet suggestions
        Nutritional guidance

        Health Goals secondary categories:
        Weight management
        Fitness advice
        Wellness goals
        Health improvement strategies

        General Inquiry secondary categories:
        Application features
        User feedback
        Technical support
        Speak to a human

        Off-Topic categories:
        Not related to the nutritionist service
        Unintelligible input
        Irrelevant content
        Non-queries
    """

    messages = [
        {'role': 'system', 'content': system_message},
        {'role': 'user', 'content': f"{delimiter}{user_input}{delimiter}"},
    ]

    response = get_completion_from_messages(messages)
    return response


# Function to process user input and generate responses

def process_user_message(user_input, all_messages, user_info, debug=True):
    delimiter = "```"

    # Get categorized information
    product_information = get_input_category(user_input).replace("\n", "")

    # Include user information in the prompt
    user_info_prompt = f"The user is {user_info}."
    prompt = f"You are an AI-powered nutritionist assistant, if user info supplied use, dont not further ask. {user_info_prompt} {product_information}"

    messages = [
        {'role': 'system', 'content': prompt},
        {'role': 'user', 'content': f"{delimiter}{user_input}{delimiter}"},
        {'role': 'assistant', 'content': f"Relevant product information:\n{product_information}"}
    ]

    final_response = get_completion_from_messages(all_messages + messages)

    # Update AI memory
    all_messages.append(messages[1])  # Only keep the user message in the history
    all_messages.append({'role': 'assistant', 'content': final_response})
    # Model self-evaluation
    user_message = f"""
        Customer message: {delimiter}{user_input}{delimiter} ?
        Agent response: {delimiter}{final_response}{delimiter}

        Does the response sufficiently answer the question?
    """

    messages = [
        {'role': 'system', 'content': prompt},
        {'role': 'user', 'content': user_message}
    ]

    evaluation_response = get_completion_from_messages(messages)

    if "Y" in evaluation_response:
        return final_response, all_messages
    else:
        neg_str = "Could you be more clear with your request"
        return neg_str, all_messages


# Streamlit App
def main():
    st.title("AI Nutritionist Chatbot")

    # Initialize variables
    all_messages = []
    user_info = {'age': 0, 'height': 0, 'activity_level': '', 'allergies': '', 'nutrition_goal': ''}

    # Side panel for user information
    st.sidebar.title("User Information")
    user_info['age'] = st.sidebar.number_input("Age", min_value=0, max_value=150, value=25)
    user_info['height'] = st.sidebar.number_input("Height (cm)", min_value=0, max_value=300, value=170)
    user_info['activity_level'] = st.sidebar.selectbox("Activity Level", ["Sedentary", "Lightly Active", "Moderately Active", "Very Active"])
    user_info['allergies'] = st.sidebar.text_area("Allergies")
    user_info['nutrition_goal'] = st.sidebar.text_area("Desired Nutrition Goal")

    # Input box for user
    user_input = st.text_input("You:", "")

    # Button to submit user input
    if st.button("Ask"):
        # Process user input
        response, all_messages = process_user_message(user_input, all_messages, user_info)

        # Display chat history
        st.text_area("Chat History", value="\n".join([f"{msg['role']}: {msg['content']}" for msg in all_messages]))

        # Display chatbot response
        st.text_area("AI Nutritionist:", value=response, height=len(response.split("\n")) * 20)  # Adjust height dynamically

# Run the Streamlit app
if __name__ == "__main__":
    main()
