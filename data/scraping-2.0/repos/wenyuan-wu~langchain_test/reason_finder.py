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
    sys_template = query_by_id(prompt_db_path, "prompts", 2)[0]["prompt_text"]
    result = chatgpt_wrapper(sys_template, event)
    return result


def reason_extractor(prompt_context, history):
    database_folder = "database"
    prompt_db_path = os.path.join(database_folder, "prompt.db")
    sys_template = query_by_id(prompt_db_path, "prompts", 3)[0]["prompt_text"]
    sys_template = sys_template.format(prompt_context=prompt_context)
    history_str = "\n".join(history)
    result = chatgpt_wrapper_16k(sys_template, history_str)  # gpt 16k model
    try:
        output = json.loads(result)
    except json.JSONDecodeError:
        logging.info("The response was not valid JSON.")
        non_reason = """
        {
        "reason_found": false,
        "reason": null,
        "confidence_score": 0
        }
        """
        output = json.loads(non_reason)
    return output


def reason_finder(event):
    full_event = get_full_info(event)
    prompt_context = create_context_template(full_event)

    database_folder = "database"
    prompt_db_path = os.path.join(database_folder, "prompt.db")
    sys_template = query_by_id(prompt_db_path, "prompts", 4)[0]["prompt_text"]
    sys_template = sys_template.format(prompt_context=prompt_context)
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
    chat_type = "reason finder"
    language = "en"
    conversation_sum = f"reason finder at {now} with {usr_id}"
    history = []
    record_chat_db(chat_db_path, chat_id, usr_id, now, chat_type, language, conversation_sum, history)

    while True:
        usr_prompt = input("\ntype input here: \n")
        history.append(f"patient: {usr_prompt}")
        logging.info(history)
        reason = reason_extractor(prompt_context, history)
        logging.info(reason)
        if reason["confidence_score"] < 95:
            resp = conversation(usr_prompt)
            history.append(f"physician: {resp['response']}")
            update_chat_db(chat_db_path, chat_id, history)
        else:
            print(memory.json())
            update_chat_db(chat_db_path, chat_id, history)
            return reason


if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s',
                        level=logging.INFO,
                        # datefmt='%d-%b-%y %H:%M:%S'
                        )
    load_dotenv()
    event_json = {
        "patient": 12,
        "plan": 27,
        "measure": 18,
        "completed": False
    }
    reason_stat = reason_finder(event_json)
    print(reason_stat)
