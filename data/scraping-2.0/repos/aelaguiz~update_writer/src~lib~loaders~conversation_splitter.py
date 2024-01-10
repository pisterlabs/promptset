import sys
import os
import typing
from typing import Optional, Union, Literal, AbstractSet, Collection, Any, List
from langchain.text_splitter import TextSplitter
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.output_parsers.json import SimpleJsonOutputParser
from langchain.output_parsers import PydanticOutputParser
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_core.pydantic_v1 import BaseModel, Field, validator
from langchain.callbacks import OpenAICallbackHandler
from langchain_core.documents import Document
from operator import itemgetter
import json
from .. import lib_docdb, lc_logger
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)  # for exponential backoff



import logging


"""
I think the steps should be:

1. Review the messages and extract all unique conversations, parciipants and when and who started the conversation
2. Classify messages one by one into the conversation they most likely fit into including
"""

topic_prompt_template = """# Discord message conversation classification

Your task is to analyze a list of Discord messages and identify unique conversations. For each conversation, extract the following information:

1. Conversation Topic: Determine the main topic or subject of the conversation.
2. Participants: List all users who have participated in the conversation.
3. Initial Message: Identify the first message that started the conversation, including who said it and when.

Consider the content of the messages, the participants involved, and the flow of the conversation to distinguish between different topics. Ignore time gaps between messages, as conversations can be ongoing over extended periods.

Analyze the messages and output the extracted conversations with the required details.

## Guidelines for conversation topics

1. Topic Consistency: Group messages into a conversation if they discuss the same main topic.
2. Response Chain: Include a message in a conversation if it directly responds to or follows up on an earlier message.
3. Extended Time Window: In addition to the contextual markets you can also use a 2-hour window to determine if messages belong to the same conversation when other methods fail.
4. Natural Conversation Flow: Identify the start and end of conversations based on the natural flow of messages.
5. Multiple Conversations: Recognize and categorize simultaneous conversations based on topic relevance.


## Examples

### Example 1:

Example Input:
```
1: UserA (01/01/2024, 08:15 PM): Hey everyone, how about a game night this Saturday?
2: UserB (01/01/2024, 08:20 PM): Sounds fun, I'm in!
3: UserC (01/01/2024, 08:45 PM): Saturday works for me!
```

Expected Output:
```json
{{
    "conversations": [
        {{
            "topic": "Game Night Planning",
            "participants": ["UserA", "UserB", "UserC"],
            "first_message": 1
        }}
    ]
}}
```

### Example 2:
Example Input:
```
4: UserX (01/01/2024, 10:00 AM): Has anyone seen the latest space documentary?
5: UserY (01/01/2024, 10:05 AM): Yes, watched it last night. It's mind-blowing!
6: UserZ (01/01/2024, 10:15 AM): I think our project needs a new approach.
7: UserX (01/01/2024, 10:20 AM): Totally agree on the documentary. What did you think of the Mars segment, UserY?
8: UserA (01/01/2024, 10:30 AM): @UserZ, I'm open to suggestions. What are you thinking?
9: UserY (01/01/2024, 10:35 AM): Mars segment was the best. Also, @UserZ, are you proposing a complete overhaul?
10: UserZ (01/01/2024, 10:40 AM): Not a complete overhaul, but significant changes. Let's discuss this afternoon.
11: UserB (01/01/2024, 10:45 AM): I missed the documentary. Can anyone summarize it?
12: UserA (01/01/2024, 10:50 AM): Looking forward to the meeting, @UserZ. We definitely need fresh ideas.
```

Expected Output:
```json
{{
    "conversations": [
        {{
            "topic": "Space Documentary Discussion",
            "participants": ["UserX", "UserY", "UserB"],
            "first_message": 4
        }},
        {{
            "topic": "Project Strategy Discussion",
            "participants": ["UserZ", "UserA", "UserY"],
            "first_message": 6
        }}
    ]
}}```

# INPUT MESSAGES:
```
{input_messages}
```

**Note**: Respond with the output json and nothing else
"""


message_classify_prompt_template = """# Discord message classification

Given a list of new Discord messages and the identified conversation topics from Step 1, your task now is to classify each of these new messages into the most relevant existing conversation. Use the following criteria for your classification:

1. Message Content: Compare the content of each new message to the topics of existing conversations.
2. Participants: Consider if the sender of the new message is already a participant in an existing conversation.
3. Mentions and Context: Look for direct mentions (@username) and contextual hints in the message that might link it to an existing conversation.

## Guidelines for Classification

1. Topic Relevance: Assign a message to a conversation where its content closely aligns with the identified topic.
2. Existing Participants: If a message is from a user already participating in a conversation, it's likely to belong to that conversation.
3. Conversational Flow: Use any indications of ongoing dialogue, like direct responses or follow-up questions, to classify messages.

## Example

Input Messages:
```
13: UserX (01/01/2024, 10:00 AM): Has anyone seen the latest space documentary?
14: UserZ (01/01/2024, 10:15 AM): I think our project needs a new approach.
15: UserY (01/01/2024, 10:05 AM): Yes, watched it last night. It's mind-blowing!
16: UserA (01/01/2024, 08:15 PM): Hey everyone, how about a game night this Saturday?
17: UserB (01/01/2024, 08:20 PM): Sounds fun, I'm in!
18: UserC (01/01/2024, 08:45 PM): Saturday works for me!
19: UserD (01/01/2024, 09:00 PM): What time are we starting the game night?

```

Conversations:
```json
{{
    "conversations": [
        {{
            "topic": "Space Documentary Discussion",
            "participants": ["UserX", "UserY"],
            "first_message": 13
        }},
        {{
            "topic": "Game Night Planning",
            "participants": ["UserA", "UserB", "UserC"],
            "first_message": 16
        }},
        {{
            "topic": "Project Strategy Discussion",
            "participants": ["UserZ"],
            "first_message": 14
        }}
    ]
}}
```

Expected Output:
```json
{{
    "conversation_messages": [
        {{
            "topic": "Space Documentary Discussion",
            "participants": ["UserX", "UserY"],
            "messages": [
                13,
                15
            ]
        }},
        {{
            "topic": "Game Night Planning",
            "participants": ["UserA", "UserB", "UserC"],
            "messages": [
                16,
                17,
                18,
                19
            ]
        }},
        {{
            "topic": "Project Strategy Discussion",
            "participants": ["UserZ"],
            "messages": [
                14
            ]
        }}
    ]
}}
```

# INPUT MESSAGES:
```
{input_messages}
```

# EXISTING CONVERSATIONS:
```json
{conversations}
```


**Note**: Respond with the output json and nothing else
"""

from pydantic import BaseModel, Field
from typing import List
from datetime import datetime

class Message(BaseModel):
    user: str
    timestamp: datetime = Field(..., example="01/01/2024, 10:00 AM")
    message: str

class Conversation(BaseModel):
    topic: str
    participants: List[str]
    first_message: Optional[int] = None  # Reference to the message number
    messages: Optional[List[int]] = None  # List of message numbers


class ConversationData(BaseModel):
    conversation_messages: List[Conversation]




from datetime import timedelta
import traceback

import tiktoken

# Function to count the number of tokens in a message
def count_tokens(message: str, encoding) -> int:
    return len(encoding.encode(message))

# Updated function to split messages into chunks based on time intervals and token limits
def split_into_time_chunks(messages, interval_minutes=60, max_tokens=2048):
    if not messages:
        return []

    # Load the encoding for token counting
    encoding = tiktoken.get_encoding("cl100k_base")  # Replace with the appropriate encoding

    messages = sorted(messages, key=lambda x: x['timestamp'])

    chunks = []
    current_chunk = []
    current_token_count = 0
    chunk_start_time = messages[0]['timestamp']

    for message in messages:
        message_token_count = count_tokens(message['message'], encoding)

        if (message['timestamp'] < chunk_start_time + timedelta(minutes=interval_minutes) and 
            current_token_count + message_token_count <= max_tokens):
            current_chunk.append(message)
            current_token_count += message_token_count
        else:
            chunks.append(current_chunk)
            current_chunk = [message]
            current_token_count = message_token_count
            chunk_start_time = message['timestamp']

    if current_chunk:
        chunks.append(current_chunk)

    return chunks



def format_message_for_json(message):
    """Formats a message dictionary for JSON serialization, converting datetime to string."""
    formatted_message = message.copy()
    formatted_message['timestamp'] = message['timestamp'].strftime("%Y-%m-%d %H:%M:%S")
    return formatted_message


def format_message_for_prompt(message):
    """
    Formats a message dictionary for input into the prompt.

    Args:
        message (Dict): A dictionary representing the message.
                        Expected keys: 'number', 'user', 'timestamp', 'message'.
    
    Returns:
        str: The formatted message string.
    """
    # Example format: "13: UserX (01/01/2024, 10:00 AM): Has anyone seen the latest space documentary?"
    timestamp = message['timestamp'].strftime("%d/%m/%Y, %I:%M %p") if isinstance(message['timestamp'], datetime) else message['timestamp']
    formatted_message = f"{message['number']}: {message['user']} ({timestamp}): \"{message['message']}\""
    return formatted_message

@retry(wait=wait_random_exponential(min=1, max=30), stop=stop_after_attempt(2))
def batch_classify_with_backoff(message_classify_chain, *args, **kwargs):
    return message_classify_chain.batch(*args, **kwargs)

def split_conversations(messages: List[str]) -> List[str]:
    logger = logging.getLogger(__name__)
    logger.debug(f"Splitting conversation on {len(messages)} messages")

    message_lookup = {msg['number']: msg for msg in messages}

    time_chunks = split_into_time_chunks(messages, interval_minutes=60*24*2)
    logger.debug(f"Split into {len(time_chunks)} time chunks")

    total_msgs = 0
    for idx, chunk in enumerate(time_chunks):
        logger.debug(f"Chunk {idx+1}: {len(chunk)} messages")
        # for msg in chunk:
        #     logger.debug(f"\t\t{msg['timestamp']} {msg['user']}: {msg['message']}")
        #     logger.debug(json.dumps(format_message_for_json(msg)))
        # logger.debug(f"\tFirst message: {chunk[0]}")
        # logger.debug(f"\tLast message: {chunk[-1]}")
        total_msgs += len(chunk)

    logger.debug(f"Counted {total_msgs} messages in {len(time_chunks)} chunks")

    llm = lib_docdb.get_json_llm()
    lmd = lc_logger.LlmDebugHandler()
    oaic = OpenAICallbackHandler()

    topic_prompt = PromptTemplate.from_template(topic_prompt_template)
    message_classify_prompt = PromptTemplate.from_template(message_classify_prompt_template)

    def inspect_obj(obj):
        logger.debug(f"Inspecting object: {obj}")

        return obj

    topic_chain = (
        topic_prompt
        | llm
        | StrOutputParser()
        # | PydanticOutputParser(pydantic_object=ConversationData)
    )
    
    message_classify_chain = (
        {
            "conversations": topic_chain,
            "input_messages": itemgetter("input_messages")
        }
        | message_classify_prompt
        | llm
        | PydanticOutputParser(pydantic_object=ConversationData)
    )

    # overall_chain =  (
    #     topic_chain
    #     | message_classify_chain
    #     | PydanticOutputParser(pydantic_object=ConversationData)
    # )

    tc = time_chunks

    batches = [{"input_messages": "\n".join(format_message_for_prompt(m) for m in chunk)} for chunk in tc]
    # logger.debug(batches[0]["input_messages"])
    # res = message_classify_chain.batch(batches, config={'callbacks': [lmd]})

    # for idx, val in enumerate(res):
    #     logger.debug(f"Result[{idx}]")
    #     logger.debug("    " + json.dumps(json.loads(val.json()), indent=4))

    batch_size = 5
    batches_of_batches = [batches[i:i + batch_size] for i in range(0, len(batches), batch_size)]

    for idx, batch_of_batches in enumerate(batches_of_batches):
        try:
            logger.debug(f"Processing batch {idx+1}/{len(batches_of_batches)} containing {len(batch_of_batches)} batches")
            res = batch_classify_with_backoff(message_classify_chain, batch_of_batches, config={'callbacks': [oaic, lmd]})
            logger.debug(oaic)
            for batch in res:
                logger.debug(f"Result: ")
                logger.debug(f"     {batch}")
                # logger.debug("    " + json.dumps(json.loads(res.json()), indent=4))
                for conversation in batch.conversation_messages:
                    logger.debug(f"Conversation: {conversation.topic}")
                    logger.debug(f"    Participants: {conversation.participants}")
                    logger.debug(f"    Messages: ")
                    chats = []
                    for msg_num in conversation.messages:
                        msg = message_lookup[msg_num]
                        logger.debug(f"        {format_message_for_prompt(msg)}")
                        chats.append(msg)

                    return_doc = {
                        'topic': conversation.topic,
                        'participants': conversation.participants,
                        'messages': chats
                    }
                    yield return_doc
        except Exception as e:
            traceback_str = traceback.format_exc()
            logger.error(f"Error processing batch {idx}: {e}")
            logger.error(f"Batch contained {len(batch_of_batches[idx])} batches")

            # Create a dictionary to store the exception information
            exception_info = {
                "exception": str(e),
                "traceback": traceback_str,
                "batch_of_batches": batch_of_batches
            }

            # Create the "fails" directory if it doesn't exist
            fails_directory = "fails"
            os.makedirs(fails_directory, exist_ok=True)

            # Generate a unique filename
            filename = f"exception_{idx}.json"
            filepath = os.path.join(fails_directory, filename)

            # Save the exception information into a JSON file
            with open(filepath, "w") as file:
                json.dump(exception_info, file, indent=4)

            logger.error(f"Exception information saved to: {filepath}")
            raise e