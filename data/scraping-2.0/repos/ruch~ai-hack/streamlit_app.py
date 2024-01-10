import streamlit as st
import openai
# Set your OpenAI API key here


def newsplit(lines):
# Splitting the text into lines based on newline character

# Removing empty lines and extracting topics
    split_list = [element.split('- ') for element in lines]
    # topics = [line.strip('- ') for line in lines if line.strip('- ')]
    # for topic in topics:
        # print("- " + topic)
    for line in lines:
        st.write(line + "\n")

def generate_learning_material(topic, available_time):
    system_message = "You are a teacher who breaks down complex or difficult topics into simple and easy to understand learning material for 15-year-olds. Only focus on the most important points. Breakout the text in suitable paragraphs. Mark the end of content with ## Then suggest three adjacent topics that the student can learn next. Separate the topics with a comma." 
    user_message = f"Generate a brief learning material about {topic} that I can understand in {available_time} minutes."

    lconversation = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_message}
    ]

    conversation = f"{system_message}\n{user_message}\n"
    response = openai.Completion.create(
        engine="text-davinci-002",
        prompt=conversation,
        max_tokens=300,  # Adjust as needed
        temperature=0.4  # Adjust the temperature for response randomness
    )

    # Extract adjacent topics from the response

    return response.choices[0].text.strip()

def main():
    st.title("Learning Platform on the go")
    # st.write("How much time you have?")
    #st.write("What do you want to learn?")

    available_time = st.slider("Select time you have to learn (minutes)", min_value=10, max_value=120, value=5, step=5)
    selected_topic = st.text_input("What do you want to learn?")

    if selected_topic:
        full_response = generate_learning_material(selected_topic, available_time)
        #st.write(full_response)
        lines = full_response.split("##")
        #st.write(lines)
        learning_material = lines[0]
        adjacent_topics = lines[1:]

        st.subheader(f"Learning Material for '{selected_topic}' ({available_time} minutes):")
        st.write(learning_material)
        st.header("What to continue learning?")
        split_topics= newsplit(adjacent_topics)
            #st.write(adjacent_topics)


if __name__ == "__main__":
    main()
