import asyncio
import json
from pathlib import Path
from typing import Dict, Any

from langchain import PromptTemplate, ConversationChain
from langchain.callbacks import StdOutCallbackHandler
from langchain.chat_models import ChatOpenAI

from jonbot import logger
from jonbot.backend.data_layer.analysis.get_chats import get_chats
from jonbot.backend.data_layer.models.discord_stuff.discord_chat_document import DiscordChatDocument

_DEFAULT_SUMMARY_TEMPLATE = """
Write a concise summary of the following:
"{text}"
CONCISE SUMMARY:"""
SUMMARY_TEMPLATE = PromptTemplate.from_template(_DEFAULT_SUMMARY_TEMPLATE)


async def summarize_tree(chats: Dict[str, DiscordChatDocument],
                         save_path: str) -> Dict[str, Any]:
    topics_by_chat = {}
    tasks = []
    try:
        for chat_number, chat_item in enumerate(chats.items()):
            chat_id, chat = chat_item
            if chat_number > 5:
                break
            await extract_topics_from_chat(chat, chat_id, save_path, topics_by_chat)

        await asyncio.gather(*tasks)
    except Exception as e:
        logger.exception(e)
        raise e
    finally:
        return topics_by_chat


async def extract_topics_from_chat(chat: DiscordChatDocument,
                                   chat_id: int,
                                   save_path: str,
                                   topics_by_chat: Dict[str, Any]):
    print(f"\n=====================================\n"
          f"There are {len(chat.messages)} messages in this chat."
          f"\n_____________________________________\n")
    llm = ChatOpenAI(temperature=0,
                     model_name="gpt-3.5-turbo",
                     callbacks=[StdOutCallbackHandler()],
                     streaming=True)
    llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo-16k")
    llm_chain = LLMChain(llm=llm, prompt=prompt)
    conversation_chain = ConversationChain(
        llm=llm,
        verbose=True,
        prompt=TOPIC_CONVERSATION_TEMPLATE,
        memory=topic_memory
    )
    human_ai_pairs = get_human_ai_message_pairs(chat)
    topics = []
    for human_message, ai_response in human_ai_pairs:
        # inputs = {"input": human_message}
        # outputs = {"output": ai_response}
        # topic_memory.save_context(inputs=inputs,
        #                           outputs=outputs)
        # topics.append(topic_memory.load_memory_variables(inputs={"input": human_message}, ))
        # topic_memory.save_context(inputs=inputs,
        #                           outputs=outputs)
        input_string = f"Human: {human_message}\nAI: {ai_response}"
        conversation_chain.run(input=input_string)
    json_filename = f"{chat_id}_extracted_topics.json"
    json_path = Path(save_path) / json_filename
    with open(str(json_path), "w", encoding="utf-8") as file:
        json.dump(topic_memory.entity_store.store, file, indent=4)
    print(f"Saved conversation topic memory to {json_path}")
    topics_by_chat[chat_id] = topic_memory.entity_store.store


if __name__ == "__main__":
    database_name = "classbot_database"
    server_id = 1150736235430686720
    chats_in = asyncio.run(get_chats(database_name=database_name,
                                     query={"server_id": server_id}))
    save_path = Path().home() / "syncthing_folders" / "jon_main_syncthing" / "jonbot_data"
    save_path.mkdir(parents=True, exist_ok=True)
    topics_by_chat = asyncio.run(summarize_tree(chats=chats_in,
                                                save_path=str(save_path)))

    json_filename_out = f"_all_chats_extracted_topics.json"
    json_path_out = Path(save_path) / json_filename_out

    with open(str(json_path_out), "w", encoding="utf-8") as file:
        json.dump(topics_by_chat, file, indent=4)
