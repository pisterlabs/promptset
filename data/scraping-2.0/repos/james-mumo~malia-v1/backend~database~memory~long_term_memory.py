from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.memory import VectorStoreRetrieverMemory
from database.memory.short_term_memory import save_short_term_memory_to_json
from database.chat_history.chat_history import save_chat_dialogue_to_sql
from utils.tools import INFORMATION_ABOUT_JAY
import shutil

# For deleting vdb
from utils.config import VDB_DIR, SHORT_FULL_SUMMARY_FILE_PATH, LONG_FULL_SUMMARY_FILE_PATH
import os

# Load from vdb and create vector store memory
class GetVDB:
    def __init__(self):
        if os.path.exists(VDB_DIR):
            # If long memory exists, directly use it
            self.vdb = Chroma(persist_directory=VDB_DIR, embedding_function=OpenAIEmbeddings())
        else:
            # Else create a new one with primed knowledge
            self.vdb = Chroma.from_texts(INFORMATION_ABOUT_JAY, persist_directory=VDB_DIR, embedding=OpenAIEmbeddings())
        self.retriever = self.vdb.as_retriever(search_kwargs=dict(k=5))
        self.v_memory = VectorStoreRetrieverMemory(retriever=self.retriever)


def save_dialogues_to_vdb(dialogues, message_time):
    # Save the recent 2 messages to vdb
    [jay, malia] = dialogues
    [jay_time, malia_time] = message_time
    
    jay_message = f"""Jay: {jay}
Time of Record: {jay_time}
"""
    malia_message = f"""MALIA: {malia}
Time of Record: {malia_time}
"""

    try: 
        VDB.vdb.add_texts([jay_message, malia_message])
        VDB.vdb.persist()
        print("###Recent chat saved to vdb successfully!###")
        print()
    except Exception as e:
        print("!!!Faled to save recent chat vdb!!!")
        print(e)
        print()


def save_video_short_full_summary_to_vdb():
    # Store short full summary    
    with open(SHORT_FULL_SUMMARY_FILE_PATH, "r+", encoding="utf-8") as f:
        short_full_summary = f.read()
        try: 
            VDB.vdb.add_texts([short_full_summary])
            VDB.vdb.persist()
            print("###Short full sumamry saved to vdb successfully!###")
            print()
        except Exception as e:
            print("!!!Faled to save short full sumamry  vdb!!!")
            print(e)
            print()
            
        # Clear the file after storing  
        print("Clearing the short full summary")
        f.seek(0)
        f.truncate()
    
    
    
def save_video_long_full_summary_to_vdb():
    # Store long full summary
  
    with open(LONG_FULL_SUMMARY_FILE_PATH, "r+", encoding="utf-8") as f:
        long_full_summary = f.read()
        summary_list = long_full_summary.split("\n\n")
        
        try: 
            VDB.vdb.add_texts(summary_list)
            print("###All long full sumamry saved to vdb successfully!###")
            print()
            VDB.vdb.persist()
             # Clear the file after storing 
            print("Clearing the long full summary")
            f.seek(0)
            f.truncate()
        except Exception as e:
            print("!!!Faled to save long full sumamry vdb!!!")
            print(e)
            print()  
        
       
 

def delete_vdb():
    # clean vdb data
    try:
        VDB.vdb.delete_collection()
        VDB.vdb.persist()
        # Force remove the folder
        shutil.rmtree("database/vdb/")

        print("###Successfully delete vdb!###")
    except ValueError as e:
        print("!!!vdb collection does not exits!!!")
    except Exception as e:
        print(e)
    

    # Reset vdb
    try:
        # After remove long-term memory, create another new one with primed knowledge
        VDB.vdb = Chroma.from_texts(INFORMATION_ABOUT_JAY, persist_directory=VDB_DIR, embedding=OpenAIEmbeddings())
        VDB.retriever = VDB.vdb.as_retriever(search_kwargs=dict(k=5))
        VDB.v_memory = VectorStoreRetrieverMemory(retriever=VDB.retriever)
    except Exception as e:
        print(f"!!!Failed to create a new vdb!!!")
        print(e)


def persist_to_memory(message_buffer):
     
    user_message = message_buffer.jay_text 
    jay_time = message_buffer.jay_time 
    malia_response = message_buffer.malia_text
    malia_time = message_buffer.malia_time 
    
    short_term_memory = message_buffer.malia_messages
    moving_summary_buffer = message_buffer.malia_moving_summary_buffer
    
    
    # Save dialogues to vdb
    save_dialogues_to_vdb(
        dialogues=[user_message, malia_response], 
        message_time=[jay_time, malia_time]
    )

    # Save short-term memory to json
    save_short_term_memory_to_json(
        short_term_memory=short_term_memory, 
        moving_summary_buffer=moving_summary_buffer
    )
    
    # Save whole chat history to sqlite
    jay_message = ("Jay", user_message, jay_time)
    malia_message = ("MALIA", malia_response, malia_time)
    save_chat_dialogue_to_sql(message=jay_message)
    save_chat_dialogue_to_sql(message=malia_message)    
    
    
    # Store video summary to vdb if there is any
    if os.path.exists(LONG_FULL_SUMMARY_FILE_PATH) \
        and os.path.getsize(LONG_FULL_SUMMARY_FILE_PATH) != 0:
        save_video_long_full_summary_to_vdb()
    if os.path.exists(SHORT_FULL_SUMMARY_FILE_PATH) \
        and os.path.getsize(SHORT_FULL_SUMMARY_FILE_PATH) != 0:
        save_video_short_full_summary_to_vdb()



VDB = GetVDB()

if __name__ == '__main__':
    pass