import asyncio
import json
from pathlib import Path
from typing import Dict, Any, Tuple

import numpy as np
from langchain.embeddings import OpenAIEmbeddings
from langchain.schema import Document
from langchain.vectorstores import Chroma

from jonbot.backend.data_layer.analysis.get_chats import get_chats
from jonbot.backend.data_layer.models.discord_stuff.discord_chat_document import DiscordChatDocument
from jonbot.backend.data_layer.visualize_data.plot_vector_clusters_3d import visualize_clusters_3d


async def create_vector_store(chats: Dict[str, DiscordChatDocument],
                              collection_name: str,
                              persistence_directory: str) -> Tuple[Chroma, Dict[str, Any], Dict[str, Any]]:
    print("Creating vector store from {collection_name} collection with {len(all_entries)} entries")
    document_tree = {}
    document_tree["speakers"] = {}
    document_tree["categories"] = {}
    document_tree["channels"] = {}
    document_tree["chats"] = {}
    document_tree["couplets"] = {}
    all_documents_list = []
    all_speaker_documents_list = []
    all_channel_documents_list = []
    all_category_documents_list = []
    all_chat_documents_list = []
    all_couplet_documents_list = []
    all_documents_list.extend(all_speaker_documents_list)
    all_documents_list.extend(all_channel_documents_list)
    all_documents_list.extend(all_category_documents_list)
    all_documents_list.extend(all_chat_documents_list)
    all_documents_list.extend(all_couplet_documents_list)

    word_count = {}
    word_count["bot"] = 0
    word_count["human"] = 0
    word_count["speakers"] = {}

    for chat_id, chat in chats.items():
        # get the first non-bot speaker

        context_route = chat.context_route
        chat_document = Document(page_content=chat.as_text,
                                 metadata={"source": chat.jump_url,
                                           "type": "chat",
                                           "chat_id": chat.thread_id,
                                           "context_description": chat.context_description,
                                           **context_route.as_flat_dict,
                                           }
                                 )
        # save as chat
        document_tree["chats"][chat_id] = chat_document
        all_chat_documents_list.append(chat_document)

        # append to channel document
        if context_route.channel.id not in document_tree["channels"]:
            channel_document = Document(page_content="",
                                        metadata={
                                            "source": f"channel `{context_route.channel.name}` id:{context_route.channel.id}",
                                            "type": "channel",
                                            "server_id": context_route.server.id,
                                            "server_name": context_route.server.name,
                                            "category_id": context_route.category.id,
                                            "category_name": context_route.category.name,
                                        }
                                        )
            document_tree["channels"][context_route.channel.id] = channel_document
            all_channel_documents_list.append(
                channel_document)  # relying on pass-by-reference weirdness to keep this up to date

        document_tree["channels"][context_route.channel.id].page_content += f"_____\n\n{chat.as_text}_____\n\n"

        # save each couplet
        for couplet_number, couplet in enumerate(chat.couplets):

            if not couplet.human_message or not couplet.ai_message or couplet.as_text == "":
                continue

            speaker_id = couplet.human_message.author_id
            if speaker_id != 0:
                if speaker_id not in document_tree["speakers"]:
                    word_count["speakers"][speaker_id] = {}
                    word_count["speakers"][speaker_id]["bot"] = 0
                    word_count["speakers"][speaker_id]["human"] = 0

                    speaker_document = Document(page_content="",
                                                metadata={"speaker_id": speaker_id,
                                                          "source": f"speaker_id: {speaker_id}",
                                                          "type": "speaker",
                                                          }
                                                )
                    document_tree["speakers"][speaker_id] = speaker_document
                    all_speaker_documents_list.append(
                        speaker_document)  # relying on pass-by-reference weirdness to keep this up to date

                document_tree["speakers"][speaker_id].page_content += f"_____\n\n{couplet.as_text}_____\n\n"
                word_count["speakers"][speaker_id]["bot"] += len(
                    couplet.ai_message.content.replace("\n", "").split(' '))
                word_count["speakers"][speaker_id]["human"] += len(
                    couplet.human_message.content.replace("\n", "").split(' '))

                word_count["bot"] += len(couplet.ai_message.content.replace("\n", "").split(' '))
                word_count["human"] += len(couplet.human_message.content.replace("\n", "").split(' '))
                couplet_document = Document(page_content=couplet.as_text,
                                            metadata={"source": couplet.human_message.jump_url,
                                                      "type": "couplet",
                                                      "couplet_number": couplet_number,
                                                      "chat_id": chat.thread_id,
                                                      "speaker_id": speaker_id,
                                                      "context_description": chat.context_description,
                                                      **chat.context_route.as_flat_dict,
                                                      }
                                            )
                document_tree["couplets"][couplet_number] = couplet_document
                all_couplet_documents_list.append(couplet_document)

    for channel_id, channel_document in document_tree["channels"].items():
        if channel_document.page_content == "":
            del document_tree["channels"][channel_id]
            continue
        if channel_document.metadata["category_id"] not in document_tree["categories"]:
            category_document = Document(page_content="",
                                         metadata={
                                             "source": f"category `{channel_document.metadata['category_name']}` id:{channel_document.metadata['category_id']}",
                                             "type": "category",
                                             "server_id":
                                                 channel_document.metadata[
                                                     "server_id"],
                                             "server_name":
                                                 channel_document.metadata[
                                                     "server_name"],
                                         }
                                         )
            document_tree["categories"][channel_document.metadata["category_id"]] = category_document
            all_category_documents_list.append(
                category_document)  # relying on pass-by-reference weirdness to keep this up to date

        document_tree["categories"][channel_document.metadata[
            "category_id"]].page_content += f"____________\n\n{channel_document.page_content}___________\n\n"

    vector_store = Chroma(
        embedding_function=OpenAIEmbeddings(),
        collection_name=collection_name,
        persist_directory=persistence_directory,
    )
    # await vector_store.aadd_documents(all_documents_list)

    await vector_store.aadd_documents(all_couplet_documents_list)

    # await vector_store.aadd_documents(all_chat_documents_list)

    def recurse_tree(document_tree) -> Dict[str, Any]:
        document_tree_dict = {}
        for key, document in document_tree.items():
            if isinstance(document, Document):
                document_tree_dict[key] = document.dict()
            else:
                document_tree_dict[key] = recurse_tree(document)
        return document_tree_dict

    document_tree_dict = recurse_tree(document_tree)

    return vector_store, document_tree_dict, word_count


async def plot_vectorstore_data(vector_store: Chroma):
    collection = vector_store._collection.get(include=["embeddings", "documents", "metadatas"])

    labels = []
    for metadata in collection["metadatas"]:
        labels.append(f"{metadata['channel_name']}")
    embeddings_npy = np.asarray(collection["embeddings"])

    visualize_clusters_3d(embeddings=embeddings_npy,
                          text_contents=collection["documents"],
                          metadatas=collection["metadatas"])



async def get_or_create_vectorstore(chroma_persistence_directory: str,
                                    database_name: str,
                                    server_id: int,
                                    chroma_collection_name: str,
                                    document_tree_json_name="document_tree.json",
                                    word_count_json_name="word_counts.json") -> Chroma:

    if Path(chroma_persistence_directory).exists():
        vector_store = Chroma(persist_directory=chroma_persistence_directory,
                              embedding_function=OpenAIEmbeddings(),
                              collection_name=chroma_collection_name)
    else:
        chats_out = await get_chats(database_name=database_name,
                                    query={"server_id": server_id})
        chat_documents = {key: DiscordChatDocument.from_dict(chat_dict) for key, chat_dict in chats_out.items()}
        vector_store, document_tree_dict, word_counts = await create_vector_store(chats=chat_documents,
                                                                                  collection_name=chroma_collection_name,
                                                                                  persistence_directory=chroma_persistence_directory)
    return vector_store


if __name__ == "__main__":
    database_name_in = "classbot_database"
    server_id_outer = 1150736235430686720

    vector_store_outer = asyncio.run(get_or_create_vectorstore(chroma_collection_name="classbot_vector_store",
                                                               chroma_persistence_directory="classbot_chroma_persistence",
                                                               server_id=server_id_outer,
                                                               database_name=database_name_in
                                                               )
                                     )
    relevant_documents = vector_store_outer.similarity_search("tell me about the center of mass", k=4)

    for document in relevant_documents:
        print(f"______________________\n\n"
              f"{document}\n\n"
              f"______________________\n\n")

    asyncio.run(plot_vectorstore_data(vector_store_outer))
