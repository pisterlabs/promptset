from db_init import query_by_id, record_chat_db, update_chat_db
import os
import json
from util import chatgpt_wrapper, chatgpt_wrapper_16k
from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate
)
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
import logging
import uuid
from datetime import datetime


def get_full_info(event):
    database_folder = "database"
    patient_db_path = os.path.join(database_folder, "patient.db")
    patient_json = query_by_id(patient_db_path, "patients", event["patient"])[0]
    full_event = event.copy()
    full_event["patient"] = {k: patient_json[k] for k in ["name", "age", "BMI"]}
    plan_db_path = os.path.join(database_folder, "plan.db")
    plan_json = query_by_id(plan_db_path, "plans", event["plan"])[0]
    full_event["plan"] = plan_json["plan_text"]
    measure_db_path = os.path.join(database_folder, "measure.db")
    measure_json = query_by_id(measure_db_path, "measures", event["measure"])[0]
    full_event["measure"] = measure_json["measure_text"]
    return full_event


def create_context_template(event):
    database_folder = "database"
    prompt_db_path = os.path.join(database_folder, "prompt.db")
    # using version v1 of the prompt, 'id=1'
    sys_template = query_by_id(prompt_db_path, "prompts", 1)[0]["prompt_text"]
    result = chatgpt_wrapper(sys_template, event)
    return result


def agreement_checker(prompt_context, history):
    database_folder = "database"
    prompt_db_path = os.path.join(database_folder, "prompt.db")
    sys_template = query_by_id(prompt_db_path, "prompts", 6)[0]["prompt_text"]
    sys_template = sys_template.format(prompt_context=prompt_context)
    history_str = "\n".join(history)
    result = chatgpt_wrapper_16k(sys_template, history_str)  # gpt 16k model
    try:
        output = json.loads(result)
    except json.JSONDecodeError:
        logging.info("The response was not valid JSON.")
        non_reason = """
        {
        "agreement": false,
        "measure": null,
        "confidence_score": 0
        }
        """
        output = json.loads(non_reason)
    return output


def measure_giver(event, chat_history):
    full_event = get_full_info(event)
    prompt_context = create_context_template(full_event)

    database_folder = "database"
    prompt_db_path = os.path.join(database_folder, "prompt.db")
    sys_template = query_by_id(prompt_db_path, "prompts", 5)[0]["prompt_text"]
    measure = [
  {
    "id": 18,
    "measure_sum": "Do-Abend schwimmen",
    "measure_text": "Am Do-Abend schwimmen gehen"
  },
  {
    "id": 19,
    "measure_sum": "Aqua-Fit anschliessen",
    "measure_text": "sich einer Aqua-Fit anschliessen, wo andere, übergewichtigeMenschen auch teilnehmen"
  }
]
    measure = str(measure)
    sys_template = sys_template.format(prompt_context=prompt_context, measure=measure)
    prompt = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(sys_template, validate_template=False),
        MessagesPlaceholder(variable_name="history"),
        HumanMessagePromptTemplate.from_template("{input}")
    ])
    llm = ChatOpenAI(
        model_name="gpt-3.5-turbo-16k",
        temperature=0,
        streaming=True,
        callbacks=[StreamingStdOutCallbackHandler()]
    )
    memory = ConversationBufferMemory(return_messages=True)
    conversation = ConversationChain(memory=memory, prompt=prompt, llm=llm)

    # record chat history
    chat_db_path = os.path.join(database_folder, "conversation.db")
    chat_id = str(uuid.uuid4())
    now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    usr_id = event["patient"]
    chat_type = "measure giver"
    language = "en"
    conversation_sum = f"measure giver at {now} with {usr_id}"
    record_chat_db(chat_db_path, chat_id, usr_id, now, chat_type, language, conversation_sum, chat_history)

    resp = conversation(str(chat_history))
    chat_history.append(f"physician: {resp['response']}")
    update_chat_db(chat_db_path, chat_id, chat_history)

    while True:
        usr_prompt = input("\ntype input here: \n")
        chat_history.append(f"patient: {usr_prompt}")
        logging.info(chat_history)
        agreement = agreement_checker(prompt_context, chat_history)
        logging.info(agreement)
        if agreement["confidence_score"] < 95:
            resp = conversation(usr_prompt)
            chat_history.append(f"physician: {resp['response']}")
            update_chat_db(chat_db_path, chat_id, chat_history)
        else:
            print(memory.json())
            update_chat_db(chat_db_path, chat_id, chat_history)
            return agreement


if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s',
                        level=logging.INFO,
                        # datefmt='%d-%b-%y %H:%M:%S'
                        )
    load_dotenv()
    reason_json = {
        "patient": 12,
        "plan": 27,
        "measure": 18,
        "reason": {
            'reason_found': True,
            'reason': 'The patient is not comfortable in public swimming pools.',
            # 'confidence_score': 80  #TODO: better workaround
        }
    }
    history = ["patient: Hi",
               "physician: Hello! How are you today? I hope you're doing well. "
               "I noticed that you didn't complete the swimming session on Thursday evening as planned. "
               "Can you tell me what happened?",
               "patient: I was tired.",
               "physician: I understand that feeling tired can make it difficult to stick to your planned activities. "
               "It's impkortant to listen to your body and give it the rest it needs. "
               "However, it's also important to find a balance between rest and exercise "
               "to maintain a healthy lifestyle. "
               "Is there anything specific that made you feel more tired than usual on Thursday?",
               "patient: I don't want to go swimming",
               "physician: I see. It's completely normal to have days "
               "where you don't feel like doing a particular activity. "
               "Is there a specific reason why you didn't feel like going swimming on Thursday? "
               "Understanding your reasons can help me provide you with better support and guidance.",
               "patient: I am not comfortable in public swimming pools"]
    measure_stat = measure_giver(reason_json, history)
    print(measure_stat)
