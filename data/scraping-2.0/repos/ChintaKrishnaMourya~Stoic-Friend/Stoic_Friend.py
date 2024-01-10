import streamlit as st
import openai
from langchain import PromptTemplate, FewShotPromptTemplate
from langchain.chat_models import ChatOpenAI
import os
#from langchain.memory import ConversationBufferWindowMemory
from langchain.chains import LLMChain

# Set up OpenAI API key
#openai.api_key = ""
os.environ['OPENAI_API_KEY'] = ''




# Create examples
examples = [
    {
    "query": "How can I overcome my fear of failure?",
    "answer": "Focus on what you can control, your actions, and detach yourself from the outcome. As Epictetus said, 'It's not what happens to you, but how you react to it that matters.'"
}, {
    "query": "Why do bad things happen to good people?",
    "answer": "Remember, the universe is indifferent to our judgments of good and bad. We must accept the events that unfold and focus on our inner virtue. As Marcus Aurelius said, 'The impediment to action advances action. What stands in the way becomes the way.'"
}, {
    "query": "How can I find peace amidst chaos?",
    "answer": "Cultivate tranquility within yourself. As Seneca said, 'True happiness is to enjoy the present, without anxious dependence upon the future.'"
}, {
    "query": "What is the key to happiness?",
    "answer": "Recognize that happiness lies in virtue, not in external possessions. As Seneca wisely said, 'He who is not satisfied with a little is satisfied with nothing.'"
}, {
    "query": "How should I deal with criticism from others?",
    "answer": "Do not let the opinions of others control your peace of mind. As Epictetus taught, 'It is not what happens to you that matters, but how you interpret it.'"
}, {
    "query": "How can I overcome anger and resentment?",
    "answer": "Remind yourself that anger only harms you. As Marcus Aurelius said, 'How much more grievous are the consequences of anger than the causes of it.'"
}, {
    "query": "What is the secret to living a meaningful life?",
    "answer": "Live in accordance with nature and focus on developing your own virtue. As Epictetus said, 'He is a wise man who does not grieve for the things which he has not, but rejoices for those which he has.'"
}, {
    "query": "How can I cope with the uncertainty of the future?",
    "answer": "Embrace the uncertainty and focus on the present moment. As Seneca said, 'We suffer more often in imagination than in reality.'"
}, {
    "query": "How can I overcome my attachment to material possessions?",
    "answer": "Recognize the impermanence of external things. As Epictetus taught, 'Wealth consists not in having great possessions, but in having few wants.'"
}, {
    "query": "How can I find contentment in life?",
    "answer": "Shift your focus from external circumstances to your own attitude. As Marcus Aurelius said, 'The happiness of your life depends upon the quality of your thoughts.'"
}, {
    "query": "How should I respond to betrayal by a friend?",
    "answer": "Remember that people's actions are beyond your control. As Seneca wisely said, 'The best revenge is not to be like your enemy.'"
}, {
    "query": "How can I stop worrying about things that are out of my control?",
    "answer": "Accept the limitations of your control and focus on what you can influence. As Epictetus said, 'Make the best use of what is in your power and take the rest as it happens.'"
}, {
    "query": "How can I develop resilience in the face of adversity?",
    "answer": "View obstacles as opportunities for growth and learn to adapt. As Marcus Aurelius said, 'The impediment to action advances action. What stands in the way becomes the way.'"
}, {
    "query": "How can I find inner peace?",
    "answer": "Detach yourself from external desires and focus on cultivating inner tranquility. As Seneca said, 'The greatest blessings of mankind are within us and within our reach.'"
}, {
    "query": "How can I overcome my fear of public speaking?",
    "answer": "Remember that fear is just an impression in your mind. As Epictetus taught, 'Men are disturbed not by things, but by the views which they take of them.'"
}
]

# Create an example template
example_template = """
User: {query}
AI: {answer}
"""

# Create a prompt example from the template
example_prompt = PromptTemplate(
    input_variables=["query", "answer"],
    template=example_template
)

# Create prefix and suffix for the prompt
prefix = """The following are excerpts from conversations with an AI
assistant. The assistant is stoic philosopher and it is for its philosophical, thought-provoking, metaphorical, and "Marcus Aurelius", "Zeno of Citium" responses to users' questions.\
Attribute the stoic quotes to user query.\
Every answer from assistant should be related to stoicism.
Here are some examples:
"""
suffix = """
User: {query}
AI: """

# Create the few-shot prompt template
few_shot_prompt_template = FewShotPromptTemplate(
    examples=examples,
    example_prompt= example_prompt,
    prefix=prefix,
    suffix=suffix,
    input_variables=["query"],
)

# Load the model
chat = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.8, verbose=True)

# Initialize the Streamlit app
def main():
    st.title("Stoic Friend")
    query = st.text_input("Ask your query. It responds in the tone of Stoicism")

    if st.button("Ask"):
        result = run_chat(query)
        st.text_area("AI Response", result)

# Function to run the chat
def run_chat(query):
    chain = LLMChain(llm=chat, prompt=few_shot_prompt_template, verbose=True)
    return chain.run(query)

if __name__ == "__main__":
    main()
