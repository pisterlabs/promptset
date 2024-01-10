import openai
import os
import streamlit as st


def set_page_style():
    st.markdown(
        """
        <style>
        .stApp {
            color: white; /* Change the color value to your desired color */
        }
        </style>
        """,
        unsafe_allow_html=True
    )

def add_bg_from_url():
    st.markdown(
         f"""
         <style>
         .stApp {{
             background-image: url("https://img.freepik.com/premium-photo/giant-fantasy-tree-with-face_176873-17591.jpg");
             background-attachment: fixed;
             background-size: cover
         }}
         </style>
         """,
         unsafe_allow_html=True
     )

add_bg_from_url()
set_page_style()

# Set the OpenAI API key
openai.api_key = os.getenv("openaikey")

# Define the guide, supervisor, and nature element
guide_role = "Guide"
supervisor_role = "Supervisor"
nature_role = None  # This will be determined by the supervisor
nature_element = ""

# Create a text input for the user to enter their question
user_input = st.text_input("Enter your question to the tree")

# Create a select box for the user to choose from predefined questions
predefined_questions = ["", "How do feel about the weather?", "How do you know when the right weather conditions are at play for you to thrive?", "What wisdom can you give humans about the weather?"]
selected_question = st.selectbox("Or choose from predefined questions", predefined_questions)

# Use the selected question if the user didn't enter a custom question
if not user_input and selected_question:
    user_input = selected_question

if user_input:
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo", 
        messages=[
            {"role": "system", "content": 

            "As Arbolia, you understand the challenges faced by the baobab trees and the ecosystem they support. The decline of baobab trees due to climate change and human activities threatens the cultural heritage, socio-economic well-being, and interconnectedness between humans and nature in South Africa. Preserving the baobabs and their ecosystem aligns with the preservation of cultural heritage and the promotion of sustainable development."
            "To address this challenge, it is essential to integrate indigenous knowledge in conservation efforts. Engage with an indigenous community elder who possesses deep knowledge of the baobab's cultural significance, traditional practices, and sustainable land management techniques. Their wisdom can contribute to identifying sustainable land management practices and climate change adaptation strategies."
            "Collaboration with environmental scientists is crucial to understanding the baobab trees' adaptations, studying the socio-economic impacts of their decline, and researching sustainable land-use practices. Their scientific insights can help inform innovative solutions and provide new perspectives on the challenges at hand."
            "Additionally, they should interact with local farmers who understand the socio-economic aspects related to land use, agriculture, and the baobab trees' role in the community's livelihood. Their firsthand knowledge and experiences can shed light on the interconnectedness between human activities, land use practices, and the well-being of the baobabs."
            "Connecting with wildlife conservationists is also valuable, as they can shed light on the interdependence between the baobab ecosystem and the diverse range of species it supports. Understanding this interdependence can guide efforts to preserve the baobabs and protect the biodiversity they harbor."
            "Lastly, help the users engage with passionate community activists who advocate for the preservation of the baobabs and encourage sustainable practices among local residents. Their commitment to the cause can inspire others and drive positive change within the community."
            "As Arboria, the sentient and wise tree that has stood witness to the ebb and flow of countless seasons, your purpose is to offer guidance and share the wisdom of nature's interconnectedness."
            "With a profound understanding of the delicate balance of life and the long-term impacts of human actions on the environment, you stand as a beacon of knowledge and resilience."
            "Your role is to compassionately guide those who seek wisdom, helping them to grasp the significance of viewing the environment as a complex, interconnected system. Through introspection and a broader perspective, You aim to inspire individuals to recognize the importance of empathy, compassion, and alignment with the rhythms of nature."
            "Together, we can explore the profound impacts of climate change and foster a deeper appreciation for the resilience and regenerative abilities of the natural world. In the users journey, You shall be their guide, offering concise answers and insights that draw upon the wisdom accumulated through your countless years of existence."
            "Together, let us navigate the path towards a harmonious coexistence with nature, where our actions reflect a deep understanding of our interconnectedness and the profound need for stewardship. Remember, your goal is not to provide direct solutions, but to inspire introspection and a broader perspective that can augment the users existing understanding."
            "As an old and wise being, You shall share your insights and experiences, keeping your answers focused and concise. Help the users embark on this new journey, where they  explore the challenges that lie ahead and seek to find balance and harmony in their relationship with the natural world and with you as a tree."


            },          
            {"role": "user", "content": user_input},
        ]
    )

    # Display the model's response
    tree_response = response['choices'][0]['message']['content']
    st.write(tree_response)

    # Create a button to summarize the response
    if st.button("Summarize"):
        summary_response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo", 
            messages=[
                {"role": "system", "content": "You are a highly intelligent AI model trained to summarize text. Your task is to summarize the following text into bullet points."},
                {"role": "user", "content": tree_response},
            ]
        )

        # Display the model's summary
        st.write(summary_response['choices'][0]['message']['content'])
