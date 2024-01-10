import streamlit as st
from langchain.callbacks import StreamlitCallbackHandler
from langchain.document_loaders import PagedPDFSplitter
from typing import List
import os
from dotenv import load_dotenv
import streamlit.components.v1 as components
import os
import torch
from torchvision import transforms, models
from torchvision.models.feature_extraction import create_feature_extractor
from PIL import Image
from deeplake.core.vectorstore.deeplake_vectorstore import VectorStore
import os
import numpy
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain.chains.query_constructor.base import AttributeInfo



load_dotenv()




model = models.resnet18(pretrained=True)

return_nodes = {
    "avgpool": "embedding"
}
model = create_feature_extractor(model, return_nodes=return_nodes)

model.eval()
model.to("cpu")


tform = transforms.Compose([
    transforms.Resize((224,224)), 
    transforms.ToTensor(),
    transforms.Lambda(lambda x: torch.cat([x, x, x], dim=0) if x.shape[0] == 1 else x),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])



batch_size_slider = st.sidebar.slider("Batch Size", min_value=1, max_value=100, value=5)
openai_model_name = st.sidebar.selectbox("OpenAI Model Name", ["gpt-3.5-turbo-0613", "gpt-3.5-turbo-16k", "gpt-4"])
openai_temperature = st.sidebar.slider("OpenAI Temperature", 0.0, 1.0, 0.7, 0.01)

def embedding_function(images, model = model, transform = tform, batch_size = batch_size_slider):
    """Creates a list of embeddings based on a list of image filenames. Images are processed in batches."""

    if isinstance(images, str):
        images = [images]

    #Proceess the embeddings in batches, but return everything as a single list
    embeddings = []
    for i in range(0, len(images), batch_size):
        batch = torch.stack([transform(Image.open(item)) for item in images[i:i+batch_size]])
        batch = batch.to("cpu")
        with torch.no_grad():
            embeddings+= model(batch)['embedding'][:,:,0,0].cpu().numpy().tolist()

    return embeddings

data_folder = 'documents/deeplake_images/common_objects'

image_fns = [os.path.join(data_folder, file) for file in os.listdir(data_folder) if os.path.splitext(file)[-1]=='.jpg']


# Uncomment the line after this if you wish to use OpenAI embeddings, but be aware
# This is likely very costly, so I reccomend using HuggingFace embeddings instead
# Deeplake is a library for working with very large datasets, and running all of those embeddings
# through openai is likely to be very expensive

# embedding_function = OpenAIEmbeddings(model='text-embedding-ada-002')
# embedding_function = HuggingFaceHubEmbeddings(model='hkunlp/instructor-xl')

vector_store_path = os.getenv("ACTIVELOOP_HUB_PATH")

vector_store = VectorStore(
    path = vector_store_path,
)


# db = DeepLake(dataset_path = vector_store_path, embedding=embedding_function, read_only=True)

# qa = RetrievalQA.from_chain_type(llm=OpenAIChat(model='gpt-3.5-turbo'), chain_type='stuff', retriever=db.as_retriever())


st.title("Deeplake :ocean: Image Similarity Search")
st.write("Connect your deeplake image dataset to a deeplake vectorstore and search for similar images.")
st.subheader("under construction :construction:")

st.write(f"This is your current dataset: {vector_store_path}")
save_path = "/documents/deeplake_images/user_images"
uploaded_files = st.file_uploader("Choose image(s) to upload to vectorstore:", accept_multiple_files=True)







if prompt := st.chat_input():
    st.chat_message("user").write(prompt)
    for uploaded_file in uploaded_files:
    
        bytes_data = uploaded_file.read()

        st.write("filename:", uploaded_file.name)
        st.write(bytes_data)

        file_name = os.path.join(save_path, uploaded_file.name)

        vector_store.add(image = image_fns,
                    filename = image_fns,
                    embedding_function = embedding_function, 
                    embedding_data = image_fns)

        st.success(f"Saved {uploaded_file.name} to {vector_store_path}")

    with st.chat_message("assistant"):
                st_callback = StreamlitCallbackHandler(st.container())

                st.write(prompt)
                # query_docs = db.similarity_search(query = prompt)
                image_path = 'documents/deeplake_images/user_images/reference_image.jpg'

                result = vector_store.search(embedding_data = [image_path], embedding_function = embedding_function)
                vector_image_response = Image.fromarray(result['image'][0])
                st.image(vector_image_response, caption='reference image', use_column_width=True)
                # st.write(query_docs)
            