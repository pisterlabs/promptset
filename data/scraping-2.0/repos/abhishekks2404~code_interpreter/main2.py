import os
import asyncio
import streamlit as st
from codeinterpreterapi import CodeInterpreterSession
import uuid
from dotenv import load_dotenv
from pydantic import BaseModel
from langchain.chat_models import ChatOpenAI
from langchain.schema import SystemMessage
from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, MessagesPlaceholder
from langchain.memory import ConversationBufferMemory
import openpyxl

# load_dotenv()


# deployment_name: str = "gpt-3.5-turbo-16k"
# openai_api_type: str = "azure"
# openai_api_base: str = os.getenv("OPENAI_ENDPOINT")
# openai_api_version: str = "2023-03-15-preview"
class File(BaseModel):
    name: str
    content: bytes

    @classmethod
    def from_path(cls, path: str):
        if not path.startswith("/"):
            path = f"./{path}"
        with open(path, "rb") as f:
            path = path.split("/")[-1]
            return cls(name=path, content=f.read())
        
    @classmethod
    def from_bytes(cls, name: str, content: bytes):
        return cls(name=name, content=content)

    @classmethod
    async def afrom_path(cls, path: str):
        return await asyncio.to_thread(cls.from_path, path)

    @classmethod
    def from_url(cls, url: str):
        import requests  # type: ignore

        r = requests.get(url)
        return cls(name=url.split("/")[-1], content=r.content)

    @classmethod
    async def afrom_url(cls, url: str):
        import aiohttp

        async with aiohttp.ClientSession() as session:
            async with session.get(url) as r:
                return cls(name=url.split("/")[-1], content=await r.read())

    def save(self, path: str):
        if not path.startswith("/"):
            path = f"./{path}"
        with open(path, "wb") as f:
            f.write(self.content)

    async def asave(self, path: str):
        await asyncio.to_thread(self.save, path)

    def get_image(self):
        try:
            from PIL import Image  # type: ignore
        except ImportError:
            print(
                "Please install it with "
                "`pip install 'codeinterpreterapi[image_support]'`"
                " to display images."
            )
            exit(1)

        from io import BytesIO

        img_io = BytesIO(self.content)
        img = Image.open(img_io)

        # Convert image to RGB if it's not
        if img.mode not in ("RGB", "L"):  # L is for greyscale images
            img = img.convert("RGB")

        return img

    def show_image(self):
        img = self.get_image()
        # Display the image
        try:
            # Try to get the IPython shell if available.
            shell = get_ipython().__class__.__name__  # type: ignore
            # If the shell is in a Jupyter notebook or similar.
            if shell == "ZMQInteractiveShell" or shell == "Shell":
                from IPython.display import display  # type: ignore

                display(img)
            else:
                img.show()
        except NameError:
            img.show()

    def __str__(self):
        return self.name

    def __repr__(self):
        return f"File(name={self.name})"

# openai_api_key: str = st.secrets["OPENAI_API_KEY"]
if "memory" not in st.session_state:
                        
    st.session_state.memory = ConversationBufferMemory()


# Set verbose mode to display more information
os.environ["VERBOSE"] = "True"
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": """Hello! How may I assist you?""" }]

if "images" not in st.session_state:
    st.session_state.images = []
images = []
def main():
    st.title("Code Interpreter")
    


    with st.sidebar:
        st.subheader("Upload File to Chat")
        uploaded_files = st.file_uploader("Choose a file", accept_multiple_files=True,type=[".jpg",".png",".xlsx",".csv",".txt",".py",".json"])

        uploaded_files_list = []
        for uploaded_file in uploaded_files:
            bytes_data = uploaded_file.read()
            uploaded_files_list.append(File.from_bytes(name=uploaded_file.name,
                                                content=bytes_data))
            
        if st.session_state.images:
            st.header("Extracted Images:")
            count = 1
            for image in st.session_state.images:
                
                st.image(image, caption=f'Image : {count}', use_column_width=True)
                count = count +1
            

                
    for message in st.session_state.messages:
        st.chat_message(message["role"]).markdown(message["content"])                
    # if(st.session_state.bool):
    #     if(st.session_state.name):
    #             with st.spinner("Thinking..."):

    #st.chat_message("assistant").markdown(f"""Hello! How may I assist you? """)
    
                    
 
    if prompt :=  st.chat_input("Ask me anything"):
            st.chat_message("user").markdown(prompt)
            st.session_state.messages.append({"role": "user", "content": prompt})
            st.session_state.memory.save_context({"input": prompt},{"output": ""})
            async def run_code_interpreter():
                with st.spinner("Thinking..."):
                
                    async with CodeInterpreterSession(model='gpt-3.5-turbo-16k') as session:
                        p = st.session_state.memory.load_memory_variables({prompt})
                        p=str(p)
                        print(p)
                        response = await session.generate_response(prompt + " = this is current prompt." +"\n"+ p +" = is the history", files=uploaded_files_list)


                        st.session_state.memory.save_context({"input": prompt},{"output": response.content})
                        # print(CodeInterpreterSession._history_backend)
                
                        
                        st.chat_message("assistant").markdown(response.content)
                
                    for file in response.files:
                    # Display the image content as bytes
                        st.image(file.content,use_column_width=True, caption=file.name)

                        st.session_state.images.append(file.content)
                        

                    
                    st.session_state.messages.append({"role": "assistant", "content": response.content})
            asyncio.run(run_code_interpreter())
        

if __name__ == "__main__":
    with open("style.css") as source_des:
        st.markdown(f"<style>{source_des.read()}</style>", unsafe_allow_html=True)

    main()