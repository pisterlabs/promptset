from datetime import datetime, timedelta

import tiktoken

from db.engine import Database
from db import schema
from sqlalchemy import update
from config import OpenAI


class DbCommands(Database):
    def __init__(self):
        Database.__init__(self)

    def create_user(self, id, nickname):
        session = self.maker()
        is_user = session.query(schema.Users).filter_by(id=id).first()
        if not is_user:
            print('попали в юзерс')
            user = schema.Users(id=id, nickname=nickname, premium=False)
            session.add(user)
            session.commit()
            session.close()
            session = self.maker()
            messages = schema.Messages(user_id=id, message=None, state=0)
            session.add(messages)
            session.commit()
            session.close()

    def delete_message(self, user_id):
        session = self.maker()
        session.query(schema.Messages).filter(schema.Messages.user_id == user_id).update(
            {"message": None})
        session.commit()
        session.close()

    def add_message(self, id, text=None, role=None, username=None, cut_response=False):
        print(f"in add_message method, user_id: {id}, role: {role}")
        messages = [
            {"role": "assistant", "content": f"{OpenAI.promt}"}
        ]
        session = self.maker()
        user = session.query(schema.Users).filter_by(id=id).first()
        if user:
            messages_from_db = session.query(schema.Messages).filter_by(user_id=id).first()
            if cut_response:
                # кейс с ошибкой - отрезаем старый ответ
                update_messages = update(schema.Messages).where(schema.Messages.user_id == id).values(
                    message=messages_from_db.message[1:])
                session.execute(update_messages)
                session.commit()
                response = messages_from_db.message[1:]
                session.close()
                return response
            if messages_from_db.message is None:
                messages.append({"role": role, "content": text})
                update_message = update(schema.Messages).where(schema.Messages.user_id == id).values(message=messages)
                session.execute(update_message)
                session.commit()
                session.close()
                return messages
            else:
                new_messages = messages_from_db.message + [{'role': role, 'content': text}]
                update_messages = update(schema.Messages).where(schema.Messages.user_id == id).values(
                    message=new_messages)
                session.execute(update_messages)
                session.commit()
                # смотрим кол-во токенов
                token_len = num_tokens_from_string(str(new_messages))
                # если колличество токенов близко к максимальному - отрезаем 1 старый ответ
                if token_len >= 1500:
                    print("token_len:", token_len)
                    update_messages = update(schema.Messages).where(schema.Messages.user_id == id).values(
                        message=messages_from_db.message[1:])
                    session.execute(update_messages)
                    session.commit()
                    new_messages = new_messages[1:]
                session.close()
                return new_messages
        else:
            self.create_user(id, username)
            messages.append({"role": role, "content": text})
            update_message = update(schema.Messages).where(schema.Messages.user_id == id).values(message=messages)
            session.execute(update_message)
            session.commit()
            session.close()
            return messages

    def add_type_of_relationship(self, id, state):
        print(state)
        session = self.maker()
        user = session.query(schema.Users).filter_by(id=id).first()
        if user:
            message = session.query(schema.Messages).filter_by(user_id=id).first()
            if message:
                update_status = update(schema.Messages).where(schema.Messages.user_id == id).values(state=state)
                session.execute(update_status)
                session.commit()
                session.close()
            else:
                pass
        else:
            pass

    def empty_message(self, id):
        session = self.maker()
        message = session.query(schema.Messages).filter_by(user_id=id).first()
        if message is None:
            messages = schema.Messages(user_id=id, message=None, state=0)
            session.add(messages)
            session.commit()
        session.close()

    def set_message_state_to_default(self, id):
        session = self.maker()
        update_status = update(schema.Messages).where(schema.Messages.user_id == id).values(state=0)
        session.execute(update_status)
        session.commit()
        session.close()

    def get_message_state(self, id):
        session = self.maker()
        message = session.query(schema.Messages).filter_by(user_id=id).first()
        if message:
            return message.state
        return 0

    def is_attempt_expire(self, id):
        session = self.maker()
        user = session.query(schema.Users).filter_by(id=id).first()
        if user:
            if user.premium:
                return True
            elif user.attempt >= 5:
                return False
            else:
                attempt = user.attempt + 1
                update_status = update(schema.Users).where(schema.Users.id == id).values(attempt=attempt)
                session.execute(update_status)
                session.commit()
                session.close()
                return True
        # если пользователя нету, то ок
        return True

    def check_premium_expire_for_all_users(self):
        session = self.maker()
        users = session.query(schema.Users).all()
        for user in users:
            if user.premium:
                current_time = datetime.now()
                day_expire = user.subscribe_expire_day
                if current_time > day_expire:
                    update_premium = update(schema.Users).where(schema.Users.id == user.id).values(
                        premium=False,
                        subscribe_expire_day=None)
                    session.execute(update_premium)
                    session.commit()
                    session.close()

    def is_user_premium(self, id):
        session = self.maker()
        user = session.query(schema.Users).filter_by(id=id).first()
        if user:
            if user.premium:
                return True
        return False

    def set_user_premium(self, id, cost):
        days_by_cost = {
            "149.00": 7,
            "399.00": 30,
            "1999.00": 365
        }
        days = days_by_cost[cost]
        subscribe_expire_day = datetime.now() + timedelta(days=days)
        session = self.maker()
        user = session.query(schema.Users).filter_by(id=id).first()
        if user:
            premium = update(schema.Users).where(schema.Users.id == id).values(
                premium=True,
                subscribe_expire_day=subscribe_expire_day)
            session.execute(premium)
            session.commit()
            session.close()
            return subscribe_expire_day


def num_tokens_from_string(string: str) -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
    num_tokens = len(encoding.encode(string))
    return num_tokens


db = DbCommands()
if __name__ == '__main__':
    db.create_user('1', 'loh')
