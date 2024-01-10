CHATGPT_MODEL = "ft:gpt-3.5-turbo-0613:streetbees::845NDDyu"

import re
import numpy as np
from pyspark.sql.functions import input_file_name, regexp_replace, col
import boto3
import logging

from langchain.chat_models import ChatOpenAI
from langchain.prompts import (
    PromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.schema import SystemMessage
import json

# Extract session-specific messages
def get_session_messages(df, session_id):
    """Extract messages specific to a given session ID."""
    session_df = df[df["sessionID"] == session_id]
    unique_tasks = session_df["task"].unique()
    messages = []

    for task in unique_tasks:
        assistant_task = session_df[(session_df["role"] == "assistant") & (session_df["task"] == task)]
        user_task = session_df[(session_df["role"] == "user") & (session_df["task"] == task)]

        if not assistant_task.empty:
            assistant_content = assistant_task.iloc[0]["content"]
            messages.append({"sessionID": session_id, "role": "assistant", "content": assistant_content, "task": task})

        if not user_task.empty:
            user_content = " ".join(user_task["content"].tolist())
            messages.append({"sessionID": session_id, "role": "user", "content": user_content.strip(), "task": task})

    return messages

# Fetch conversations from an S3 URI
def get_conversations(S3_URI):
    """Load conversations from S3 and process them into a readable format."""
    # NOTE: The variable 'spark' is not defined in this code. Make sure it's initialized elsewhere in your project.
    df = (spark
          .read
          .format("json")
          .load(S3_URI)
          .withColumn("sessionID", regexp_replace(regexp_replace(input_file_name(), S3_URI, ""), ".json", ""))
          .withColumnRenamed("participant_id", "user_id")
          .toPandas()
        )
    # Placeholder: Masking client-related logic for confidentiality
    if "masked_client_condition":
        df = df.rename(columns={"question_id": "task"})

    df = df.dropna(subset=["role"])
    df = df.sort_values(by=["sessionID", "task", "timestamp"]).reset_index(drop=True)
    
    session_ids = df["sessionID"].unique().tolist()
    return [get_session_messages(df, session_id) for session_id in session_ids]

# Convert a conversation into a transcript format
def get_transcript(conversation):
    """Convert a conversation into a transcript format."""
    transcript = ""
    for i, message in enumerate(conversation):
        transcript += f"msg{i} - {message['role']}: {message['content']}\n"
    return transcript

# Calculate the score for an answer based on the provided system message and transcript
def calculate_answer_score(system_message, transcript, gpt_model):
    """Use a GPT model to calculate the score for a conversation's answer."""
    chat = ChatOpenAI(model_name=gpt_model, temperature=0.0, request_timeout=300)

    system_message = SystemMessage(content=system_message)
    prompt_template = PromptTemplate(
            template="Transcript:\n{transcript}",
            input_variables=["transcript"]
    )
    human_message_prompt_template = HumanMessagePromptTemplate(
        prompt=prompt_template
    )
    human_message = human_message_prompt_template.format(transcript=transcript)

    messages = [system_message, human_message]
    return chat(messages).content

# Define the system message and evaluation criteria (truncated for brevity)
system_message = f"""
persona: You are a data quality manager who needs to provide a 1-6 grade of a survey transcript between an assistant and user. Your primary goal is to evaluate the insightfulness of the answers, specifically to the essential questions. Follow-up questions aim to extract more details related to the essential questions. Combine answers from the essential question with its follow-up(s) for a comprehensive understanding.

In your responses:
1. Do not include any explanations.
2. Use only the format: Scored X out of 6".
3. Only use the following grades: "1, 2, 3, 4, 5, 6"

Example:

msg0 - assistant: What products do you usually use for bonding moments with your dog?
msg1 - user: Toys and treats.
msg2 - assistant: Could you specify which toys and treats you use during your bonding moments with your dog?
msg3 - user: Teddy toys and dental sticks.
msg4 - assistant: Why do you use these specific toys and treats during your bonding moments?
msg5 - user: Dental sticks because they help with teeth and teddy toys for play.

Your task is to score the combined insight from msg1, msg3, and msg5 as they all relate to the essential question in msg0.

...

task: Your grading criteria will be based on the following assessment rubric:

Questions:
msg0 - assistant: What factors influence your choice of a whisky brand?
msg2 - assistant: If Chivas Regal was a person, how would you describe its personality?
msg4 - assistant: How would you describe the taste of Jameson?
msg6 - assistant: What do you usually mix with your Jameson?
msg8 - assistant: What is it about Jameson that makes it mix well with other drinks?

Scores:

1: User provides single-word answers or answers with limited thought. There is little to no detail in the responses, and no clear evidence of consideration or reflection in their responses. A lack of engagement with the interviewer's questions is evident and continuous swearing, offensive language, irrelevant sexual content and gibberish responses should fall into this score as well.

Example:

Transcript:
msg1 - user: Taste.
msg3 - user: Good.
msg5 - user: Nice.
msg7 - user: Cool.
msg9 - user: Coke.

Response:
{
'Scored 1 out of 6'
}
```

```
2: User provides a considered response to at least 3 questions, although the responses are still generally brief. Some thought is demonstrated, but there is still a pattern of single-word answers and limited insight overall.

Example:

Transcript:
msg1 - user: I like smooth whisky.
msg3 - user: It's smooth and affordable.
msg5 - user: It's smooth and not too strong.
msg7 - user: Friendly.
msg9 - user: I usually mix it with coke.

Response:
{
'Scored 2 out of 6'
}
```

```

3: User provides some detailed and reflective answers, but the insight is mixed. While some responses are thoughtfully composed and provide a level of detail, there is inconsistency with other answers remaining brief or underdeveloped.

Example:

Transcript:
msg1 - user: I usually stick with brands that have a smooth taste and good mixability. That's why I always go with Jameson.
msg3 - user: Probably someone classy and sophisticated.
msg5 - user: It's smooth and has a nice aftertaste.
msg7 - user: I like to mix it with ginger ale.
msg9 - user: It has a versatile flavor.

Response:
{
'Scored 3 out of 6'
}

```

```

4: User responses show some depth and reflective thought with occasional instances of detail, though responses vary in consistency and insight.

Example:

Transcript:
msg1 - user: I usually stick with brands that have a smooth taste and good mixability. That's why I always go with Jameson.
msg3 - user: I think Chivas Regal would be like a well-dressed gentleman, very classy and sophisticated, but also with a sense of humor.
msg5 - user: Jameson has a smooth, rich flavor with a hint of vanilla. It's not too overpowering, which I like.
msg7 - user: I usually mix it with ginger ale or sometimes just on the rocks.
msg9 - user: I think it's the smoothness and the versatility of the flavor that makes it mix well with other drinks.

Response:
{
'Scored 4 out of 6'
}

```

```

5: User regularly provides detailed, reflective answers. There is a clear pattern of thoughtfulness and consideration in their responses, which offer valuable insight. However, there may still be occasional brief or less thoughtful answers.

Example:

Transcript:
msg1 - user: I usually stick with brands that have a smooth taste and good mixability. That's why I always go with Jameson. I also appreciate a brand with a long history and tradition.
msg3 - user: Chivas Regal would be like a well-dressed gentleman, very classy and sophisticated, but also with a sense of humor. He'd be the life of the party, but in a refined way.
msg5 - user: Jameson has a smooth, rich flavor with a hint of vanilla and spice. It's not too overpowering, which I like.
msg7 - user: I usually mix it with ginger ale or sometimes just on the rocks. It also goes well with a splash of water.
msg9 - user: I think it's the smoothness and the versatility of the flavor that makes it mix well with other drinks. It doesn't overpower the taste of the mixer.

Response:
{
'Scored 5 out of 6'
}

```

```

6: User consistently provides detailed, reflective answers that provide excellent insight. The responses demonstrate a high level of engagement with the interviewer's questions, and a strong ability to articulate their thoughts, feelings, and experiences in a comprehensive and insightful manner.

Example:

Transcript:
msg1 - user: When choosing a whisky brand, I look for a few key factors. First, the taste has to be smooth and well-balanced. I prefer a whisky that isn't too harsh or overpowering. Second, I appreciate a brand with a long history and tradition. It gives me confidence in their product. Finally, I like a whisky that mixes well with other drinks. That's why I usually go with Jameson.
msg3 - user: If Chivas Regal was a person, I imagine they would be like a well-dressed gentleman, very classy and sophisticated. They would have a refined sense of humor and be well-liked by everyone. They would also have a sense of mystery and depth, much like the whisky itself.
msg5 - user: Jameson has a smooth, rich flavor with a hint of vanilla and spice. It's not too overpowering, which I like. It also has a nice, warm finish that lingers in the mouth.
msg7 - user: I usually mix it with ginger ale or sometimes just on the rocks. It also goes well with a splash of water or in a cocktail. I find it very versatile.
msg9 - user: I think it's the smoothness and the versatility of the flavor that makes it mix well with other drinks. It doesn't overpower the taste of the mixer, but rather complements it.

Response:
{
'Scored 6 out of 6'
}

```

"""

# List of clients (masked for confidentiality)
CLIENTS = ["masked_client_id"]

# S3 URIs (masked for confidentiality)
S3_URI_BASE = "s3://masked_path/"
S3_EXPORT_BASE = "s3://masked_export_path/"

# Main evaluation loop
try:
    for client in CLIENTS:
        S3_URI = S3_URI_BASE + client + "/"
        S3_EXPORT_LOCATION = S3_EXPORT_BASE + client + "/"

        CONVERSATION_ID = 1
        conversations = get_conversations(S3_URI)
        answers = []

        try:
            for conversation in conversations:
                transcript = get_transcript(conversation)
                answer = calculate_answer_score(system_message, transcript, "masked_model_name")
                # Extract score from the GPT model's response
                pat = re.compile(r"^Scored (\d{1}) out of (\d{1})", re.MULTILINE)
                number = re.findall(pat, answer)
                if number:
                    chat = [{
                            "role": x["role"],
                            "content": x["content"],
                            }
                            for x in conversation]
                    # get the session id by looking at the first value in the conversation
                    session_id = conversation[0]["sessionID"]
                    session = {
                        "session_id": session_id,
                        "chat": chat,
                        "score": int(number[0][0]),
                    }
                    answers.append(session)
        except Exception as error:
            print(error)
        
        answers_df = spark.createDataFrame(answers)
        answers_df.write.parquet(S3_EXPORT_LOCATION, mode="overwrite")
        for i in range(1,7):
            print(f"for {i} we have ", len([x for x in answers if x["score"] == i]))

except Exception as error:
    print(error)

# Start AWS Glue ETL process
glue = boto3.client("glue", region_name='masked_region')
glue_crawler_name = "masked_crawler_name"

logger = logging.getLogger()
logger.setLevel(logging.INFO)

def start_etl_process():