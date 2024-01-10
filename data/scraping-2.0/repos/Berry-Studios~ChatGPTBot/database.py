###
# ChatGPT discord bot
# Author: @Anthony01M
# License: GPL-3.0
###
from sqlalchemy import create_engine, Column, Integer, String
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from cryptography.fernet import Fernet
import os
from dotenv import load_dotenv
import openai

load_dotenv()

engine = create_engine(os.getenv("HOST"), echo=True)
Base = declarative_base()

class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True)
    discord_id = Column(Integer)
    api_key = Column(String(255))
    encryption_key = Column(String(255))

Base.metadata.create_all(engine)
Session = sessionmaker(bind=engine)
session = Session()

# Function to encrypt data
def encrypt_data(data: bytes) -> bytes:
    encryption_key = Fernet.generate_key()
    cipher_suite = Fernet(encryption_key)
    cipher_text = cipher_suite.encrypt(data)
    return cipher_text, encryption_key

# Function to decrypt data
def decrypt_data(cipher_text: bytes, encryption_key: bytes) -> bytes:
    cipher_suite = Fernet(encryption_key)
    plain_text = cipher_suite.decrypt(cipher_text)
    return plain_text

def add_user(discord_id, api_key):
    if not is_valid_api_key(api_key):
        return False
    if get_user(discord_id) is not None:
        return update_api_key(discord_id, api_key)
    cipher_text, encryption_key = encrypt_data(api_key.encode())
    user = User(discord_id=discord_id, api_key=cipher_text, encryption_key=encryption_key)
    session.add(user)
    session.commit()

def get_user(discord_id):
    try:
        user = session.query(User).filter_by(discord_id=discord_id).first()
        return user
    except:
        return None

def get_api_key(discord_id):
    user = get_user(discord_id)
    if user is not None:
        encryption_key = user.encryption_key
        cipher_text = user.api_key
        return decrypt_data(cipher_text, encryption_key).decode()
    else:
        return None

def update_api_key(discord_id, api_key):
    if not is_valid_api_key(api_key):
        return False
    user = get_user(discord_id)
    if user is not None:
        cipher_text, encryption_key = encrypt_data(api_key.encode())
        user.api_key = cipher_text
        user.encryption_key = encryption_key
        session.commit()
        return "updated"

def delete_user(discord_id):
    user = get_user(discord_id)
    if user is not None:
        session.delete(user)
        session.commit()
    else:
        return False

def is_valid_api_key(api_key):
    try:
        openai.api_key = api_key
        openai.Model.list()
        return True
    except:
        return False