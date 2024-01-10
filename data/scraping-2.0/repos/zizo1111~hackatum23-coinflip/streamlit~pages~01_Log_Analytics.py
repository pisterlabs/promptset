import streamlit as st
from qdrant_client.models import PointStruct
import time
st.set_page_config(page_title="Log Analyser", page_icon="👋", layout='wide')
from difflib import SequenceMatcher
import pandas as pd
import openai

if "file" not in st.session_state:
    st.error("Upload a file before proceeding")
#uploaded_file = st.file_uploader("Upload your Log file",type=['.txt',".out"])

num_of_lines_to_embedd = len(st.session_state.messages)-15000 #st.slider("Lines to Embedd", min_value=500, max_value=70000, value=5000, step=100)
num_of_chunks = st.slider("Lines to Embedd", min_value=10, max_value=1000, value=150, step=10)

lines = []
clean_lines = []
st.header('Log Analyser')
more_embed = []


st.subheader('File Information')

# Calculates embeddings for less
def calc_embed(inhalt):
        start = time.time()
        more_embed = st.session_state.model.encode(inhalt)
        end = time.time()
        return more_embed
        #st.write("Finished embeddings in " ,end-start)

# #Calculates Embeddings for all
def calculate_embeddings(to_encode, collection_name):
    with st.spinner("Calculating embeddings, please wait before proceeding"):
        start = time.time()
        embeddings = st.session_state.qdrant.calculate_embeddings(to_encode, num_of_lines_to_embedd, num_of_chunks)
        st.session_state.qdrant.populate_qdrant(collection_name,embeddings)
        st.session_state['embeddings']=embeddings
        # st.session_state["embeddings"] = st.session_state.model.encode(to_encode)
        end = time.time()
        st.write("Finished embeddings in " ,end-start)
        st.session_state["calculated"] = True


if "calculated" not in st.session_state:
    calculate_embeddings(st.session_state.messages, "my_collection")
elif(st.session_state.calculated == False):
    calculate_embeddings(st.session_state.messages , "my_collection")

                #st.write(res)


# def search_vector(query_vector, hit):
#         big_id = hit.id
#         inhalt = hit.payload["Inhalt"].split(";;;")
#         embed = calc_embed(inhalt)

#         st.session_state.client.upsert(
#                     collection_name="second_coll",
#             points=[
#                 PointStruct(
#                         id=idx,
#                         vector=vector.tolist(),
#                         payload={"Inhalt": inhalt[idx],"Line_Count": idx + big_id*num_of_chunks+1}
#                 )
#             for idx, vector in enumerate(embed)
#         ]
#         )
#         hits = st.session_state.client.search(
#         collection_name="second_coll",
#         query_vector=query_vector,
#         limit=20  # Return 10 closest points
#         )
#         return hits

# Main search from all for hits
def search_hits(collection_name, query_vector,query):
        with st.chat_message("fun_name"):

            openai.api_key = "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"
            hits = st.session_state.qdrant.search_collection(collection_name, query_vector, 5)
            top_logs = []
            log_str  = ""
            
            hits_logs = []
            idxs = []
            for hit in hits:
                st.header("Highest vector similiarity in lines: " + str(hit.id * num_of_chunks)+"-"+str(hit.id * num_of_chunks +num_of_chunks) + " with score " + str(hit.score))
                st.subheader("Top Results of this chunk")
                hits_0 = st.session_state.qdrant.search_vector(query_vector, hit,st.session_state.messages)
                hits_logs.append(hits_0)

                for hit_ in hits_0:
                    idx = hit_.id + hit.id*num_of_chunks
                    #idxs.append(idx)
                    top_logs.append(st.session_state.messages[idx])
                    if hit_.score < 0.5:
                        st.write(":red[Line:] " +str(idx) + " :blue[Message:] " +str(st.session_state.messages[idx]).replace("b'", "").replace("\\r","").replace("\\n",""))
                    else:
                        st.warning(":red[Line:] " +str(idx) + " :blue[Message:] " +str(st.session_state.messages[idx]).replace("b'", "").replace("\\r","").replace("\\n",""), icon="⚠️")
            
                log_str += " | ".join(top_logs)
            if len(log_str) > 4000 :
                 log_str = log_str[0:4000]

            sum_response = openai.Completion.create(
                engine = "text-davinci-003",
                prompt = "Summerize the following logs  LOGS : " + log_str,
                temperature = 0.6,
                max_tokens = 256,
            )

            response = openai.Completion.create(
                engine = "text-davinci-003",
                prompt = query + " LOGS : " + log_str,
                temperature = 0.8,
                max_tokens = 256,
            )
            st.subheader("GPT Response to query based on selected logs")
            st.write(response.choices[0]["text"])

            st.subheader("GPT Summary of these Logs")
            st.write(sum_response.choices[0]["text"])
 

            # for hits_0 in hits_logs:
            #      for hit_ in hits_0:
            #             if hit_.score < 0.5:
            #                 st.write(":red[Line:] " +str(idx) + " :blue[Message:] " +str(st.session_state.messages[idx]).replace("b'", "").replace("\\r","").replace("\\n",""))
            #             else:
            #                 st.warning(":red[Line:] " +str(idx) + " :blue[Message:] " +str(st.session_state.messages[idx]).replace("b'", "").replace("\\r","").replace("\\n",""), icon="⚠️")
            #print(len(log_str))
        
        #st.write(hits)

        # hits_0 = search_vector(query_vector, hit)
        # for hit in hits_0:
        #     idx = hit.payload["Line_Count"]
        #     if hit.score < 0.5:
        #         st.write(":red[Line:] " +str(idx) + " :blue[Message:] " +str(lines[idx]).replace("b'", "").replace("\\r","").replace("\\n",""))
        #     else:
        #         st.warning(":red[Line:] " +str(idx) + " :blue[Message:] " +str(lines[idx]).replace("b'", "").replace("\\r","").replace("\\n",""), icon="⚠️")


res = st.text_area(label = "Enter keywords youre interested in", value = "SSH connection error CMX")
st.session_state['user_query'] = res
query_vector = st.session_state.model.encode(res.replace("log",""))
if "embeddings" in st.session_state:
    search_hits("my_collection", query_vector,res)
