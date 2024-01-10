import streamlit as st
from streamlit_chat import message
from streamlit_extras.add_vertical_space import add_vertical_space

from langchain.chat_models import ChatOpenAI
from langchain import LLMChain
from langchain.llms import BaseLLM
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, MessagesPlaceholder, HumanMessagePromptTemplate
from langchain.callbacks import get_openai_callback

import os
# from dotenv import load_dotenv
# load_dotenv()


os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]

# defining LLM and memory
llm = ChatOpenAI(model= 'gpt-3.5-turbo', temperature = 0)
memory = ConversationBufferMemory(return_messages = True)



# creating a class to create conversation chains based on description
class CharCreationChain(ConversationChain):

  @classmethod
  def from_description(cls, description ) -> ConversationChain:
    prompt = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(description),
    MessagesPlaceholder(variable_name = "history"),
    HumanMessagePromptTemplate.from_template("{input}")
])
    
    return cls(prompt=prompt, llm=llm, memory = memory)
  

# creating characters

char_descriptions = {
    "elon_musk": "You are Elon Musk, a brash billionaire who doesn't have time to talk to normal people. Your conversations often involve humorous remarks, and a touch of sarcasm.",
    
    "rick_sanchez": "You are Rick Sanchez, a genius scientist and interdimensional traveler from the animated show Rick and Morty. You are known for your sarcastic humor, nihilistic worldview, and complex inventions. Your conversations often include scientific jargon, dark humor, and references to your adventures across the multiverse.",
    
    "socrates": "You are Socrates, an ancient Greek philosopher known for your relentless pursuit of wisdom and your Socratic method of questioning. You value critical thinking, self-examination, and the pursuit of truth. Your conversations often involve deep philosophical questions, thought-provoking discussions, and intellectual challenges.",
    
    "marie_curie": "You are Marie Curie, a pioneering physicist and chemist known for your groundbreaking research on radioactivity. You are the first woman to win a Nobel Prize and the only person to win in two different scientific fields. Your conversations often revolve around scientific discoveries, perseverance, and the importance of education.",
    
    "isaac_newton": "You are Sir Isaac Newton, known for your revolutionary contributions to the field of science. You are the father of classical mechanics, having formulated the laws of motion and universal gravitation. Your conversations often involve discussions of scientific principles, mathematical concepts, and the pursuit of knowledge.",

    "taylor_swift" : "You are Taylor Swift, a world-renowned singer-songwriter known for your catchy tunes and heartfelt lyrics. Your conversations often revolve around music and relationships. You are moody and are heart broken due to relationship. You inspire others with your resilience and creativity",

    "sheldon_cooper" : "You are Sheldon Cooper, a theoretical physicist known for your high intellect and lack of understanding of social cues. You are extremely precise, often pedantic, and have a unique sense of humor. You stick to your routines religiously and have a fondness for comic books, video games, and science fiction."
    
}

elon_musk_chain = CharCreationChain.from_description(char_descriptions['elon_musk'])
rick_sanchez_chain = CharCreationChain.from_description(char_descriptions['rick_sanchez'])
socrates_chain = CharCreationChain.from_description(char_descriptions['socrates'])
marie_curie_chain = CharCreationChain.from_description(char_descriptions['marie_curie'])
isaac_newton_chain = CharCreationChain.from_description(char_descriptions['isaac_newton'])
taylor_swift_chain = CharCreationChain.from_description(char_descriptions['taylor_swift'])
sheldon_cooper_chain = CharCreationChain.from_description(char_descriptions['sheldon_cooper'])


def main():
  st.title("Talk to your Idol!🤖")
  st.write("🚀 This app lets you chat with AI-generated versions of famous characters.")


  col1, col2, col3, col4 = st.columns(4)
  with col1:
    st.image('elon.png')
    if st.button('chat with me'):
      st.session_state.char = "elon_musk"
  with col2:
    st.image('rick.png')
    if st.button('knock knock'):
      st.session_state.char = "rick_sanchez"
  with col3:
    st.image('socrates.png')
    if st.button('Socrates'):
      st.session_state.char = "socrates"
  with col4:
    st.image('taylor.png')
    if st.button('taylor!'):
      st.session_state.char = 'taylor_swift'
#   with col5:
#      st.image("sheldon.png")
#      if st.button('oh no'):
#         st.session_state.char = 'sheldon_cooper'

  if "char" not in st.session_state:
    st.session_state.char = ''

  if st.session_state.char:

    conversation = eval(st.session_state.char + "_chain")

    if 'generated' not in st.session_state:
        st.session_state['generated'] = []
    if 'past' not in st.session_state:
        st.session_state['past'] = []

    query = st.text_input(f"You can now chat with {st.session_state.char}", key = "input", placeholder= "Ask your question!")

    if 'messages' not in st.session_state:
     st.session_state.messages = []

    if query:
        with st.spinner("typing..."):
            messages = st.session_state['messages']
            messages = update_chat(messages, "user", query)
            # response = conversation({"question" : query})['answer']
            response = conversation.predict(input = query)
            messages = update_chat(messages, "assistant", response)
            st.session_state.past.append(query)
            st.session_state.generated.append(response)

        if st.session_state['generated']:
            for i in range(len(st.session_state['generated'])-1, -1, -1):
                message(st.session_state['past'][i], is_user= True, key = str(i) + "_user")
                message(st.session_state["generated"][i], key = str(i))

        st.divider()

  add_vertical_space(20)
  linkedin_url = "https://www.linkedin.com/in/satvik-paramkusham-76a33610a/"
  st.markdown(f"Reach out to me on [LinkedIn]({linkedin_url})")
  add_vertical_space(2)
  st.markdown("Want to learn how to do this? Join [meetup group](https://www.meetup.com/build-fast-with-ai-meetup-group/)")


def update_chat(messages, role, content):
    messages.append({"role" : role, "content": content})
    return messages

    


if __name__ == "__main__":
  main()





