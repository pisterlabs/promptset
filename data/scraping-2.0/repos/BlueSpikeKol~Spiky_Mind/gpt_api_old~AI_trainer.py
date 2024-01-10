import mysql.connector
import openai
import uuid
from utils import config_retrieval

config=config_retrieval.ConfigManager()
openai.api_key = config.openai.api_key


def training_dataset_creation(prompt, answer, AI_name):
    mydb = mysql.connector.connect(
        host="localhost",
        user="root",
        password="Q144bughL0?Y@JFYxPA0",
        database="externalmemorydb"
    )

    mycursor = mydb.cursor()

    id = str(uuid.uuid4())
    sql = "INSERT INTO AI_Responses (id, prompt, output, AI_name) VALUES (%s, %s, %s, %s)"
    val = (id, prompt, answer, AI_name)

    mycursor.execute(sql, val)

    mydb.commit()

    print(mycursor.rowcount, "record inserted.")