import constants
import os
os.environ["OPENAI_API_KEY"] = constants.openapi_key


from llama_index import GPTVectorStoreIndex, QuestionAnswerPrompt
from langchain.chains.conversation.memory import ConversationBufferMemory
from langchain.chat_models import ChatOpenAI
from llama_index.langchain_helpers.agents import LlamaToolkit, create_llama_chat_agent, IndexToolConfig
from langchain.chat_models import ChatOpenAI
import time, pickle
import mysql.connector as ms
import re, ast

# Load indices from disk
index_set = {}
datas = ['About', 'Career', 'Clubs', 'FAQ', 'First Degree Programmes', 'Higher Degree Programmes', 'Phd Programmes', 'Visa Information', 'Wiki', 'Events', 'Courses','Links']
for d in datas:
    cur_index = GPTVectorStoreIndex.load_from_disk(f'{d}.json')
    index_set[d] = cur_index

index_summaries = [
                    "Simple description About Bits Pilani Dubai extracted from the BPDC Website including Mission, Vision, Policy and a general overview great to answer questions like what is bits pilani",
                    "All about careers and placements at BPDC, in depth information about the Practice School (PS) program at BPDC, also talks about the thesis alternative to practice school for FD, HD and PhD",
                    "Details on all the clubs, associations and chapters at BPDC, with names of the Presidents, chairpersons, and faculty in charge of each club or association including clubs such as as Sports, Scientific Associations, Cultural Activites, MAD (social and environmental club making a difference), public speaking and literary, dance club (groove), drama club (paribasha), art club (shades), music club (trebel), chimera, design club, fashion club (allure), photography club (reflexions), quiz club (flummoxed), supernova (astronomy), wall street club, ieee, acm and sae chapters, Association of Electronics Engineers (AOEE), American Institute of Chemical Engineers (AIChE), American Society of Mechanical Engineers (ASME), Center for Innovation, Incubation and Entrepreneurship (CIIE), Intelligent Flying Object for Reconnaissance Club (Team IFOR), Microsoft Tech Club, Skyline, WIE, SWE, Editorial Board",
                    "A great index in case none of the other indexes are appropriate. Frequently asked questions about BPDC related to Admissions, Fees, Hostel, Visa, and other including transfers, visa, costs, dress code, insurance, prospects, BPDC Library, WebOPAC portal, textbooks, parents, clinic, Question Papers. And Events such as BSF, IceBreakers, College timings etc.",
                    "Details on all the First Degree (FD), Bachelor of Engineering (B.E.) programmes at BPDC, their fees, eligibility, scholarships, concessions, special info, detailed writeups on each program. Also talks about minor programs and the structure of the program itself",
                    "Details on all the Higher Degree (FD), Master of Engineering (M.E.) and M.B.A. programmes at BPDC, their fees, eligibility, scholarships, concessions, special info, detailed writeups on each program",
                    "Details on the PHD program at BPDC its eligibility and general information",
                    "Details about UAE Residence Visa which is required to study at BPDC, how to apply and get this visa",
                    "Overview of Bits Pilani Dubai Campus and extract from the Wikipedia. Has information on the director, chancellor, vice chancellor, campus size and location, campus affiliations, overview, history, campus and DIAC (Dubai International Academic City), Hostels, Labs, Departments, Practice School (PS 1 AND 2), Events, DIAC Residential Halls, and notable alumni",
                    "Details about most annual BPDC events such as Jashn, Sparks, BITS Sports Festival (BSF), Icebreakers, Technofest & Enginuity, STEM Day, Spectrum, Artex, Dandiya Night, Recharge, Inter Year Sports Tournament, Synergy, Ethnic Day, Diro’s Tea Party, Convocation, BITS Innovation Challenge",
                    "Details about all the lectures, lectorial, practical, course requirements, attendance requirements for the courses offered for the FD programmes at BPDC with course codes such as MATH F111, BITS F225 etc. does NOT contain all the courses available",
                    "Links to important documents such as the academic calendar, FD General timetable, guide for giving feedback, applying for projects, widthdrawing from courses, enrolling/registering for courses/classes, semester abroad program, facilities services and help, projects list.",
                ]

# define toolkit
index_configs = []
i = 0
for y in datas:
    tool_config = IndexToolConfig(
        index=index_set[y], 
        name=f"Vector Index {y}",
        description=index_summaries[i],
        tool_kwargs={"return_direct": True, "return_sources": True},
    )
    index_configs.append(tool_config)
    i += 1

toolkit = LlamaToolkit(
    index_configs=index_configs
)

QA_PROMPT_TMPL = (
    "Context information is below. \n"
    "---------------------\n"
    "{context_str}"
    "\n---------------------\n"
    "Given the context information and common sense, and chat history but not prior knowledge"
    "answer the question. If you don't know the answer, reply 'I don't know': {query_str}\n"
)
QA_PROMPT = QuestionAnswerPrompt(QA_PROMPT_TMPL)

'''
create table chat(chat_id int auto_increment, fname varchar(30), feedback_count int default 0, primary key(chat_id));
create table conversation(conversation_id int auto_increment, chat_id int, user_message text, bot_message text, primary key(conversation_id), foreign key (chat_id) references chat(chat_id) on delete cascade);
'''
def create_cursor():
    mydb = ms.connect(host='localhost', user='root', password=constants.mysqlpassword, database="chatbot", autocommit=True)
    cursor = mydb.cursor(dictionary=True, buffered=True)
    return mydb, cursor

def return_chain(memory):
    llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo")
    agent_chain = create_llama_chat_agent(
        toolkit,
        llm,
        memory=memory,
        verbose=True,
        text_qa_template=QA_PROMPT
    )
    return agent_chain

def create_chain():
    memory = ConversationBufferMemory(memory_key="chat_history")
    agc = return_chain(memory)
    fname = f"{time.time()}.pkl"
    with open("memorychains/"+fname,"wb") as f:
        pickle.dump(agc.memory,f)
    mydb, cursor = create_cursor()
    cursor.execute("insert into chat(fname) values(%s)",(fname,))
    cursor.execute("select LAST_INSERT_ID() as chat_id")
    data = cursor.fetchone()
    chat_id = data["chat_id"]
    cursor.close()
    mydb.close()
    return agc, chat_id

def save_chain(chain, chat_id):
    mydb, cursor = create_cursor()
    cursor.execute("select fname from chat where chat_id=%s", (chat_id,))
    data = cursor.fetchone()
    fname = data["fname"]
    cursor.close()
    mydb.close()
    with open("memorychains/"+fname,"wb") as f:
        pickle.dump(chain.memory,f)

def load_chain(chat_id):
    mydb, cursor = create_cursor()
    cursor.execute("select fname from chat where chat_id=%s", (chat_id,))
    data = cursor.fetchone()
    fname = data["fname"]
    cursor.close()
    mydb.close()
    with open("memorychains/"+fname,"rb") as f:
        memory = pickle.load(f)
    return return_chain(memory)

def none_parser(dataDict):
    for d in dataDict:
        if dataDict[d] == 'None':
            dataDict[d] = None
    return dataDict

def log_feedback(chat_id):
    mydb, cursor = create_cursor()
    cursor.execute("update chat set feedback_count=feedback_count+1 where chat_id=%s",(chat_id,))
    cursor.close()
    mydb.close()
    return True

def return_output(message, chain, chat_id):
    simplification = re.compile(re.escape('bpdc'), re.IGNORECASE)
    message = simplification.sub('Bits Pilani Dubai Campus', message)
    
    try:
        message_response = chain.run(message)
    except Exception as e:
        print(e)
        return "Sorry, something went wrong!"
    save_chain(chain, chat_id)
    if message_response[0] == "{":
        message_response = ast.literal_eval(message_response)
    if type(message_response) == dict:
        message_response = message_response["answer"]
    
    mydb, cursor = create_cursor()
    cursor.execute("insert into conversation(chat_id, user_message, bot_message) values(%s,%s,%s)",(chat_id,message,message_response))
    cursor.close()
    mydb.close()

    return message_response

def get_chats():
    mydb, cursor = create_cursor()
    cursor.execute("select distinct chat_id, feedback_count from chat natural join conversation order by feedback_count desc")
    data = cursor.fetchall()
    cursor.close()
    mydb.close()
    return data

def get_conversation(chat_id):
    mydb, cursor = create_cursor()
    cursor.execute("select * from conversation where chat_id=%s",(chat_id,))
    data = cursor.fetchall()
    cursor.close()
    mydb.close()
    return data