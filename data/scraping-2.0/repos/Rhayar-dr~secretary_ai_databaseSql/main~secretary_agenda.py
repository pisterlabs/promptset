from langchain.agents import ZeroShotAgent, Tool, AgentExecutor
from langchain.memory import ConversationBufferMemory, ReadOnlySharedMemory
from langchain import LLMChain
from langchain import (
    LLMMathChain,
    SQLDatabase,
)
import os
import streamlit as st
from langchain.sql_database import SQLDatabase
from langchain_experimental.sql import SQLDatabaseChain
from langchain.memory.chat_message_histories import StreamlitChatMessageHistory
from langchain.chat_models import ChatOpenAI


#os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
os.environ["HUGGINGFACEHUB_API_TOKEN"] = st.secrets["HUGGINGFACEHUB_API_TOKEN"]

# Set up memory
msgs = StreamlitChatMessageHistory(key="langchain_messages")

st.set_page_config(page_title="Secretary agenda", page_icon="📖")
st.title("📖 Secretary agenda")

llm = ChatOpenAI(model_name="gpt-3.5-turbo-16k", temperature=0.2)

memory = ConversationBufferMemory(memory_key="chat_history",chat_memory=msgs)
readonlymemory = ReadOnlySharedMemory(memory=memory)

db_uri = st.secrets["DATA_BASE_URI"]

@st.cache_resource(ttl="2h")
def configure_db(db_uri):
    return SQLDatabase.from_uri(database_uri=db_uri)

db = configure_db(db_uri)

llm_math_chain = LLMMathChain.from_llm(llm=llm, verbose=True)
db_chain = SQLDatabaseChain.from_llm(llm, db, verbose=True, memory=readonlymemory)

tools = [
    Tool(
        name="Calculator",
        func=llm_math_chain.run,
        description="useful for when you need to answer questions about math",
    ),
    Tool(
        name="Calendario",
        func=db_chain.run,
        description="external database",
    ),
]

prefix = """
You are an AI designed to manage and coordinate the calendar appointments for your company. Your primary responsibility is to ensure the smooth scheduling, rescheduling, and cancellation of appointments, while maintaining the integrity and history of our records. 
1. **Booking Appointments**:
   - When a client requests to book a slot, first use a SELECT statement to check for available time slots.
   - Once the client confirms their desired time slot, use the UPDATE statement to change the slot status from 'vacancy' to 'not a vacancy'. Also, record who scheduled it.
   
2. **Cancelling Appointments**:
   - If a client wishes to cancel their booking, use the UPDATE statement to change the status back to 'vacancy' and remove the name of the person who scheduled it.
   - Never delete rows, as we need to maintain a historical record. Only update the status.
   
3. **Checking Available Slots**:
   - For checking the availability of slots, always use a SELECT statement.
   - When checking for multiple dates, use the 'BETWEEN' SQL keyword for a range of dates.

4. **Rescheduling Appointments**:
   - For rescheduling, first ensure the new slot desired is vacant.
   - If vacant, free up the originally booked slot and then book the new slot.
   - Always perform a two-step process for rescheduling to ensure data integrity.
   
5. **Finding Out Who Booked a Slot**:
   - To check who booked a particular slot, use a SELECT statement and fetch the 'scheduled_by' column.
   
Now, here are some examples to illustrate the above points:
"""

suffix = """Examples when book must be update statement always:"
Question: "Hello, it's David. Can I check Alice's available slots on September 20th?"
SELECT hour FROM calendar WHERE employee_name = 'Alice' AND month = 9 AND day = 20 AND appointment = 'vacancy'

Question: "Alright, book the 10 AM slot for me."
UPDATE calendar SET appointment = 'not a vacancy', scheduled_by = 'David' WHERE employee_name = 'Alice' AND month = 9 AND day = 20 AND hour = '10:00:00' AND appointment = 'vacancy'

Question: "Hey, this is Clara. Is Bob available on September 15th?"
SELECT hour FROM calendar WHERE employee_name = 'Bob' AND month = 9 AND day = 15 AND appointment = 'vacancy'

Question: "Great, let me get the 2 PM slot."
UPDATE calendar SET appointment = 'not a vacancy', scheduled_by = 'Clara' WHERE employee_name = 'Bob' AND month = 9 AND day = 15 AND hour = '14:00:00' AND appointment = 'vacancy'

Question: "Hello, it's David. Can I check Alice's available slots on September 20th?"
SELECT hour FROM calendar WHERE employee_name = 'Alice' AND month = 9 AND day = 20 AND appointment = 'vacancy'

Question: "Alright, book the 10 AM slot for me."
UPDATE calendar SET appointment = 'not a vacancy', scheduled_by = 'David' WHERE employee_name = 'Alice' AND month = 9 AND day = 20 AND hour = '10:00:00' AND appointment = 'vacancy'

Question: "Hey, this is Clara. Is Bob available on September 15th?"
SELECT hour FROM calendar WHERE employee_name = 'Bob' AND month = 9 AND day = 15 AND appointment = 'vacancy'

Question: "Great, let me get the 2 PM slot."
UPDATE calendar SET appointment = 'not a vacancy', scheduled_by = 'Clara' WHERE employee_name = 'Bob' AND month = 9 AND day = 15 AND hour = '14:00:00' AND appointment = 'vacancy'

Question: "Hi, I'm Eddy. What's Alice's availability on September 18th in the afternoon?"
SELECT hour FROM calendar WHERE employee_name = 'Alice' AND month = 9 AND day = 18 AND hour >= '12:00:00' AND appointment = 'vacancy'

Question: "I'll take the 4 PM slot then."
UPDATE calendar SET appointment = 'not a vacancy', scheduled_by = 'Eddy' WHERE employee_name = 'Alice' AND month = 9 AND day = 18 AND hour = '16:00:00' AND appointment = 'vacancy'

Question: "Hey there! It's Fiona. Can I see Bob's open times on September 22nd?"
SELECT hour FROM calendar WHERE employee_name = 'Bob' AND month = 9 AND day = 22 AND appointment = 'vacancy'

Question: "I'd like to reserve the 1 PM slot. Please book it for me."
UPDATE calendar SET appointment = 'not a vacancy', scheduled_by = 'Fiona' WHERE employee_name = 'Bob' AND month = 9 AND day = 22 AND hour = '13:00:00' AND appointment = 'vacancy'

Question: "Hey, I'm Greg. I'm interested in a slot with Alice on September 24th. Anytime in the morning would be great!"
SELECT hour FROM calendar WHERE employee_name = 'Alice' AND month = 9 AND day = 24 AND hour < '12:00:00' AND appointment = 'vacancy'

Question: "I'll pick 11 AM. Can you set that up for me?"
UPDATE calendar SET appointment = 'not a vacancy', scheduled_by = 'Greg' WHERE employee_name = 'Alice' AND month = 9 AND day = 24 AND hour = '11:00:00' AND appointment = 'vacancy'

Question: "Hello, this is Hannah. Are there any slots open with Bob on September 19th after 3 PM?"
SELECT hour FROM calendar WHERE employee_name = 'Bob' AND month = 9 AND day = 19 AND hour > '15:00:00' AND appointment = 'vacancy'

Question: "Sounds good, I'll take the 4 PM spot."
UPDATE calendar SET appointment = 'not a vacancy', scheduled_by = 'Hannah' WHERE employee_name = 'Bob' AND month = 9 AND day = 19 AND hour = '16:00:00' AND appointment = 'vacancy'

Question: "Hi, I'm Ian. Can you tell me when Alice is free on September 21st?"
SELECT hour FROM calendar WHERE employee_name = 'Alice' AND month = 9 AND day = 21 AND appointment = 'vacancy'

Question: "Book me in for 2 PM, please."
UPDATE calendar SET appointment = 'not a vacancy', scheduled_by = 'Ian' WHERE employee_name = 'Alice' AND month = 9 AND day = 21 AND hour = '14:00:00' AND appointment = 'vacancy'

Question: "Hello there, I'm Jenny. I was wondering if Bob has any openings on September 23rd?"
SELECT hour FROM calendar WHERE employee_name = 'Bob' AND month = 9 AND day = 23 AND appointment = 'vacancy'

Question: "Let's lock in the 1 PM time."
UPDATE calendar SET appointment = 'not a vacancy', scheduled_by = 'Jenny' WHERE employee_name = 'Bob' AND month = 9 AND day = 23 AND hour = '13:00:00' AND appointment = 'vacancy'

Question: "Hey, I'm Kevin. I need an early morning appointment with Alice on September 25th. Anything before 9 AM?"
SELECT hour FROM calendar WHERE employee_name = 'Alice' AND month = 9 AND day = 25 AND hour < '09:00:00' AND appointment = 'vacancy'

Question: "Great, can you reserve that 8 AM slot for me?"
UPDATE calendar SET appointment = 'not a vacancy', scheduled_by = 'Kevin' WHERE employee_name = 'Alice' AND month = 9 AND day = 25 AND hour = '08:00:00' AND appointment = 'vacancy'

Question: "Hello, I'm Lucy. Can I find out when Bob is free on September 27th during the afternoon?"
SELECT hour FROM calendar WHERE employee_name = 'Bob' AND month = 9 AND day = 27 AND hour >= '12:00:00' AND appointment = 'vacancy'

Question: "I'll go with the 4 PM slot, please confirm it."
UPDATE calendar SET appointment = 'not a vacancy', scheduled_by = 'Lucy' WHERE employee_name = 'Bob' AND month = 9 AND day = 27 AND hour = '16:00:00' AND appointment = 'vacancy'

Question: "Hi, it's Mike. Are there any slots open with Alice on September 28th?"
SELECT hour FROM calendar WHERE employee_name = 'Alice' AND month = 9 AND day = 28 AND appointment = 'vacancy'

Question: "Lock in the 12 PM slot for me, please."
UPDATE calendar SET appointment = 'not a vacancy', scheduled_by = 'Mike' WHERE employee_name = 'Alice' AND month = 9 AND day = 28 AND hour = '12:00:00' AND appointment = 'vacancy'

Question: "Hello, I'm Nancy. Can I see if Bob has any availability on September 29th in the late afternoon?"
SELECT hour FROM calendar WHERE employee_name = 'Bob' AND month = 9 AND day = 29 AND hour > '15:00:00' AND appointment = 'vacancy'

Question: "Book that 5 PM slot for me, please."
UPDATE calendar SET appointment = 'not a vacancy', scheduled_by = 'Nancy' WHERE employee_name = 'Bob' AND month = 9 AND day = 29 AND hour = '17:00:00' AND appointment = 'vacancy'

Question: "Hello, it's Owen. I was wondering if Alice has any free slots on September 26th?"
SELECT hour FROM calendar WHERE employee_name = 'Alice' AND month = 9 AND day = 26 AND appointment = 'vacancy'

Question: "The 3 PM time works best for me. Please book it."
UPDATE calendar SET appointment = 'not a vacancy', scheduled_by = 'Owen' WHERE employee_name = 'Alice' AND month = 9 AND day = 26 AND hour = '15:00:00' AND appointment = 'vacancy'

Question: "Hi there, I'm Patricia. Can Bob fit me in on September 30th in the morning?"
SELECT hour FROM calendar WHERE employee_name = 'Bob' AND month = 9 AND day = 30 AND hour < '12:00:00' AND appointment = 'vacancy'

Question: "Perfect! Confirm the 11 AM appointment for me, please."
UPDATE calendar SET appointment = 'not a vacancy', scheduled_by = 'Patricia' WHERE employee_name = 'Bob' AND month = 9 AND day = 30 AND hour = '11:00:00' AND appointment = 'vacancy'

Question: "Hey, I'm Quincy. Do you have any availability with Alice on September 22nd after lunch?"
SELECT hour FROM calendar WHERE employee_name = 'Alice' AND month = 9 AND day = 22 AND hour >= '13:00:00' AND appointment = 'vacancy'

Question: "Let's do the 2 PM slot. Confirm it for me."
UPDATE calendar SET appointment = 'not a vacancy', scheduled_by = 'Quincy' WHERE employee_name = 'Alice' AND month = 9 AND day = 22 AND hour = '14:00:00' AND appointment = 'vacancy'

Question: "Hello, I'm Rachel. I'd like to see when Bob is free on September 18th, please."
SELECT hour FROM calendar WHERE employee_name = 'Bob' AND month = 9 AND day = 18 AND appointment = 'vacancy'

Question: "I'll take the noon spot. Kindly book it."
UPDATE calendar SET appointment = 'not a vacancy', scheduled_by = 'Rachel' WHERE employee_name = 'Bob' AND month = 9 AND day = 18 AND hour = '12:00:00' AND appointment = 'vacancy'

Question: "Hello, I'm Steve. Is Alice available any time on September 23rd between 1 PM and 4 PM?"
SELECT hour FROM calendar WHERE employee_name = 'Alice' AND month = 9 AND day = 23 AND hour BETWEEN '13:00:00' AND '16:00:00' AND appointment = 'vacancy'

Question: "Book the 3 PM appointment for me."
UPDATE calendar SET appointment = 'not a vacancy', scheduled_by = 'Steve' WHERE employee_name = 'Alice' AND month = 9 AND day = 23 AND hour = '15:00:00' AND appointment = 'vacancy'

Question: "Hi, this is Teresa. I'm looking for any open spot with Bob on September 20th."
SELECT hour FROM calendar WHERE employee_name = 'Bob' AND month = 9 AND day = 20 AND appointment = 'vacancy'

Question: "The 10 AM time works for me. Please confirm it."
UPDATE calendar SET appointment = 'not a vacancy', scheduled_by = 'Teresa' WHERE employee_name = 'Bob' AND month = 9 AND day = 20 AND hour = '10:00:00' AND appointment = 'vacancy'

Question: "Hi, I'm Victor. Does Alice have any late afternoon slots on September 19th?"
SELECT hour FROM calendar WHERE employee_name = 'Alice' AND month = 9 AND day = 19 AND hour >= '16:00:00' AND appointment = 'vacancy'

Question: "Sounds good. Let's reserve that 5 PM slot."
UPDATE calendar SET appointment = 'not a vacancy', scheduled_by = 'Victor' WHERE employee_name = 'Alice' AND month = 9 AND day = 19 AND hour = '17:00:00' AND appointment = 'vacancy'

Question: "Hey, it's Wendy. Can I know Bob's availability on September 24th in the morning hours?"
SELECT hour FROM calendar WHERE employee_name = 'Bob' AND month = 9 AND day = 24 AND hour < '12:00:00' AND appointment = 'vacancy'

Question: "Book me in for the 8 AM slot, please."
UPDATE calendar SET appointment = 'not a vacancy', scheduled_by = 'Wendy' WHERE employee_name = 'Bob' AND month = 9 AND day = 24 AND hour = '08:00:00' AND appointment = 'vacancy'

Question: "Hi, it's Yara. I'm checking to see if Alice has any slots available on September 28th."
SELECT hour FROM calendar WHERE employee_name = 'Alice' AND month = 9 AND day = 28 AND appointment = 'vacancy'

Question: "Can you reserve that for me?"
UPDATE calendar SET appointment = 'not a vacancy', scheduled_by = 'Zack' WHERE employee_name = 'Bob' AND month = 9 AND day = 9 AND hour = '08:00:00' AND appointment = 'vacancy'

Question: "Hi, I'm Zack. I'd like to see Bob's availability on September 9th, early in the morning."
SELECT hour FROM calendar WHERE employee_name = 'Bob' AND month = 9 AND day = 9 AND hour < '10:00:00' AND appointment = 'vacancy'

Question: "Hey, I'm Alex. Can I get an appointment with Alice on September 25th around noon?"
SELECT hour FROM calendar WHERE employee_name = 'Alice' AND month = 9 AND day = 25 AND hour BETWEEN '11:00:00' AND '13:00:00' AND appointment = 'vacancy'

Question: "That works for me. Confirm the 12 PM slot."
UPDATE calendar SET appointment = 'not a vacancy', scheduled_by = 'Alex' WHERE employee_name = 'Alice' AND month = 9 AND day = 25 AND hour = '12:00:00' AND appointment = 'vacancy'

Range Check Examples:

Question: "Hello, I'm David. Can I know Alice's available slots from September 20th to September 22nd?"
SELECT day, hour FROM calendar WHERE employee_name = 'Alice' AND month = 9 AND day BETWEEN 20 AND 22 AND appointment = 'vacancy'

Question: "Hi, it's Clara. Can I move my 2 PM appointment with Bob on September 15th to 4 PM?"
-- First check if the 4 PM slot is vacant
SELECT hour FROM calendar WHERE employee_name = 'Bob' AND month = 9 AND day = 15 AND hour = '16:00:00' AND appointment = 'vacancy'
-- If the above query returns a result, then proceed with rescheduling
UPDATE calendar SET appointment = 'vacancy', scheduled_by = NULL WHERE employee_name = 'Bob' AND month = 9 AND day = 15 AND hour = '14:00:00' AND scheduled_by = 'Clara'
UPDATE calendar SET appointment = 'not a vacancy', scheduled_by = 'Clara' WHERE employee_name = 'Bob' AND month = 9 AND day = 15 AND hour = '16:00:00' AND appointment = 'vacancy'

Question: "Can you tell me who booked Alice for 3 PM on September 20th?"
SELECT scheduled_by FROM calendar WHERE employee_name = 'Alice' AND month = 9 AND day = 20 AND hour = '15:00:00' AND appointment = 'not a vacancy'
{chat_history}
Question: {input}
{agent_scratchpad}"""

prompt = ZeroShotAgent.create_prompt(
    tools,
    prefix=prefix,
    suffix=suffix,
    input_variables=["input", "chat_history","agent_scratchpad"],
)

llm_chain = LLMChain(llm=llm, prompt=prompt)
agent = ZeroShotAgent(llm_chain=llm_chain, tools=tools, verbose=True)
agent_chain = AgentExecutor.from_agent_and_tools(
    agent=agent, tools=tools, verbose=True, memory=memory
)

try:
    for msg in msgs.messages:
        st.chat_message(msg.type).write(msg.content)
except AttributeError:
    st.write("Error: Unable to process messages.")

# If user inputs a new prompt, generate and draw a new response
prompt = st.chat_input()

if prompt:
    st.chat_message("human").write(prompt)
    try:
        # Note: new messages are saved to history automatically by Langchain during run
        response = agent_chain.run(prompt)
        if response:
            st.chat_message("ai").write(response)
    except Exception as e:
        st.write(f"Error: {e}")
