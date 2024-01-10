from .models import *
from sqlalchemy.orm import Session
from .schemas import *
from .resources import *
import openai
import json
import datetime

client = openai.OpenAI(api_key=OPENAI_API_KEY)

def get_prompts(db: Session):
    return db.query(Prompts).order_by(Prompts.prompt_id).all()


def get_messages_by_client_id(client_id: ChatMessageID, db: Session):
    return db.query(
        Chat_messages
    ).filter(Chat_messages.client_id == client_id.client_id).order_by(Chat_messages.chat_message_id).all()


def get_gpt_answer(client_id: ChatMessageID, db: Session):
    full_chat = get_messages_by_client_id(client_id, db)
    if full_chat:
        sys_prompts = [(pmt.role, pmt.message) for pmt in get_prompts(db)]
        user_chat = [(chat.message, chat.is_bot) for chat in full_chat]
        
        if user_chat[-1][1]:
            return {'message': user_chat[-1][0]}
        
        gpt_prompts=[
            *[{'role': pmt[0], 'content': pmt[1]} for pmt in sys_prompts],
            *[{'role': 'assistant' if msg[1] else 'user', 'content': msg[0]} for msg in user_chat]
        ]
        
        gpt_answer = client.chat.completions.create(
            model="gpt-4",
            messages = gpt_prompts
        )
        
        return_gpt_answer = gpt_answer.choices[0].message.content


        post_chat_messages(ChatMessage(client_id=client_id.client_id, message=return_gpt_answer, is_bot=True), db)
        return {'message': return_gpt_answer}
    else:
        post_chat_messages(ChatMessage(client_id=client_id.client_id, message=FIRST_MESSAGE, is_bot=True), db)
        
        return {'message': FIRST_MESSAGE}

def add_arrangement(arrangement: Arrangement, db: Session):
    existing_record = db.query(Arrangements).filter((Arrangements.client_id == arrangement.client_id)).one_or_none()
    
    if existing_record:
        for key, value in arrangement.dict().items():
            setattr(existing_record, key, value)
        db.commit()
        db.refresh(existing_record)
        return existing_record
    else:
        new_record = Arrangements(**arrangement.dict())
        db.add(new_record)
        db.commit()
        db.refresh(new_record)
        return new_record

def post_chat_messages(message: ChatMessage, db: Session):
    chat_message = Chat_messages(**message.dict()) 
    db.add(chat_message)
    db.commit()

    lead = db.query(Leads).filter(Leads.tg == chat_message.client_id).one_or_none()
    if lead and not lead.ls_start:
        lead.ls_start = datetime.datetime.now()
        db.commit()
    if not lead:
        lead = Leads(tg=chat_message.client_id, ls_start=datetime.datetime.now())
        db.add(lead)
        db.commit()


    if check_if_last_message(chat_message.message):
        output = parse_gpt_answer(chat_message.message)
        
        
        not_string = 'client_id: ' + str(chat_message.client_id) + '\n'
        for key, value in output.items(): 
            not_string += f'{key}: {value}\n'

        

        finished_client_chat = get_messages_by_client_id(ChatMessageID(client_id=chat_message.client_id), db)
        with open(f'logs_{chat_message.client_id}.txt', 'w+', encoding='utf-8') as f:
            for chat in finished_client_chat:
                if chat.is_bot:
                    f.write(f'Bot: {chat.message}\n\n')
                else:
                    f.write(f'{chat.client_id}: {chat.message}\n\n')
        
        output['time'] = output['time'].replace('МСК','')
        output['time'] = output['time'].replace('по','')
        output['time'] = output['time'].replace('Московскому времени','')

        date = output['date'].split('.')
        time = output['time'].split(':')
        dt = datetime.datetime(int(date[2]), int(date[1]), int(date[0]), int(time[0]), int(time[1]))
        add_arrangement(Arrangement(client_id=chat_message.client_id, platform=output['platform'], datetime=dt), db)

        send_notification_to_hr_chat(not_string)
        send_logs_to_hr_chat(f'logs_{chat_message.client_id}.txt') 
        add_to_leads(output)
        lead.ls_agree = datetime.datetime.now()
        db.commit()
    if not chat_message.is_bot:    
        lead.ls_answer = datetime.datetime.now()
    db.commit()
    db.refresh(chat_message)
    db.refresh(lead)
    return chat_message


def get_notification_message(client_id: ChatMessageID, db: Session):
    arrangement = db.query(Arrangements).filter(Arrangements.client_id == client_id.client_id).one_or_none()
    if arrangement:
        not_message = NOTIFICATION_MESSAGE + arrangement.datetime.strftime('%H:%M')
        post_chat_messages(ChatMessage(client_id=client_id.client_id, message=not_message, is_bot=True), db)
        return {'message': not_message}
    else:
        return {'message': 'У вас нет запланированных встреч'}

def get_no_answer_message(db: Session):
    return {'message': NO_ANSWER_MESSAGE}


def truncate_chat_messages(client: ChatMessage = None, db: Session = None):
    if client:
        rows_to_delete = db.query(Chat_messages).filter(Chat_messages.client_id == client.client_id)
        rows_to_delete.delete(synchronize_session=False)
    else:
        rows_to_delete = db.query(Chat_messages)
        rows_to_delete.delete(synchronize_session=False)
    db.commit()