#import tensorflow as tf
import openai
import pinecone
import streamlit as st
from time import sleep
from PIL import Image

title = '<p style="font-family:Calibri;color:#006A4E;font-size: 42px;">Budd-E</p>'
st.markdown(title,unsafe_allow_html=True)



if 'user_input' not in st.session_state:
      st.session_state['user_input'] = ''

if 'generated' not in st.session_state:
      st.session_state['generated'] = []
      
if 'past' not in st.session_state:
      st.session_state['past'] = []
      
if 'query' not in st.session_state:
      st.session_state['query'] = ''
      

# get secret vars
openai.api_key = st.secrets["OPENAI_API_KEY"]
pinecone_api_key = st.secrets["pinecone_api_key"]
my_environ = st.secrets["my_environ"]
# set embedding model 
index_name = st.secrets["index_name"]




def generate_response(prompt):
      completion = openai.Completion.create(
            engine = 'text-davinci-003',
            prompt=prompt,
            max_tokens = 1024,
            n=1,
            stop=None,
            temperature=0.1
      )
      message = completion['choices'][0]['text'].strip()
      return message


def retrieve_base(query):
      res = openai.Embedding.create(
            input = [query],
            model = "text-embedding-ada-002"
      )

      xq = res['data'][0]['embedding']

      res = index.query(xq,top_k=3,include_metadata=True)
      context = [
            x['metadata']['text'] for x in res['matches']
      ]

      prompt_start = (
            "Answer the question based on the context below. Act as a very knowledgeable, friendly, stoner.\n\n"+
            "Context:\n"
      )
      prompt_end = (
            f"\n\nQuestion: {query}\nAnswer:"
      )

      for i in range(0,len(context)):
            if len("\n\n---\n\n".join(context[:i])) >= 3750:
                  prompt = (
                        prompt_start + 
                        "\n\n---\n\n".join(context[:i-1])+
                        prompt_end
                  )
                  break
            elif i == len(context)-1:
                  prompt = (
                        prompt_start + 
                        "\n\n---\n\n".join(context)+
                        prompt_end
                  )
      return prompt 



      # initialize connection to pinecone 
pinecone.init(
      api_key=pinecone_api_key,
      environment=my_environ # may be different, check at app.pinecone
)
      

#  Send to pinecone 
index = pinecone.Index(index_name)


          

def clear_submit():
      st.session_state.query = st.session_state.input
      st.session_state.input = ''


query = st.text_area("Provide a prompt",key='input',on_change = clear_submit)

if st.session_state.query:
      st.session_state['user_input']=st.session_state.query
      with st.spinner("Loading"):
            output = generate_response(retrieve_base(st.session_state.user_input))

      st.session_state['past'].append(st.session_state.user_input)
      st.session_state['generated'].append(output)


if st.session_state.generated:
      for i in range(len(st.session_state['generated'])-1,-1,-1):
            container = st.container()
            col1, col2 = container.columns([4,20])
            with col1:
                  col1.image(Image.open('face.png').resize((50,29)) )
            with col2:
                  st.write(st.session_state['past'][i],is_user=True,key=str(i)+'_user')

            container = st.container()
            col1, col2 = container.columns([4,20])
            with col1:
                  col1.image(Image.open('weed.png').resize((40,40)) )
            with col2:
                  st.write(st.session_state["generated"][i],key=str(i))

            container.divider()
