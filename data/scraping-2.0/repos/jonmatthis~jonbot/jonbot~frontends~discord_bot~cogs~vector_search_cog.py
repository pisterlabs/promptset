import json
import uuid
from typing import TYPE_CHECKING, Optional

import discord
from langchain.vectorstores.chroma import Chroma

from jonbot.backend.data_layer.vector_embeddings.create_vector_store import get_or_create_vectorstore

if TYPE_CHECKING:
    from jonbot.frontends.discord_bot.discord_bot import MyDiscordBot


class VectorSearchCog(discord.Cog):

    def __init__(self,
                 bot: "MyDiscordBot",
                 database_name: str,
                 persistence_directory: str, ):
        super().__init__()
        self.bot = bot
        self.persistence_directory = persistence_directory
        self.vector_store: Optional[Chroma] = None
        self.database_name = database_name

    @discord.slash_command(name="vector_search",
                           description="Does a vector search of the whole server using the given query")
    @discord.option(
        name="query",
        description="The query to search for - will return content that is 'similar' to this query",
        input_type=str,
        required=True,
    )
    @discord.option(
        name="number_of_results",
        description="The initial message to send to the bot",
        input_type=str,
        required=False,
    )
    async def vector_search(self,
                            ctx: discord.ApplicationContext,
                            query: str,
                            number_of_results: int = 4,
                            ) -> None:
        thing = await ctx.send(f"Searching for {query}...")

        vector_store = await get_or_create_vectorstore(chroma_collection_name=f"vector_store_{ctx.guild.id}",
                                                       chroma_persistence_directory=self.persistence_directory,
                                                       server_id=ctx.guild.id,
                                                       database_name=self.database_name)

        relevant_documents = vector_store.similarity_search(query=query, number_of_results=number_of_results)

        chunk_to_send = "Relevant documents:\n\n___________\n\n"
        for document in relevant_documents:
            chunk_to_send += f"{document.metadata['server_name']}/{document.metadata['channel_name']}/{document.metadata['thread_name']} - {document.metadata['source']}\n\n"

        # send relevant documents as json
        docs_as_json = json.dumps([document.dict() for document in relevant_documents], indent=4)
        filename = f"relevant_documents_{uuid.uuid1()}.json"
        with open(filename, 'w', encoding="utf-8") as file:
            file.write(docs_as_json)

        await thing.edit(content=chunk_to_send, file=discord.File(filename))
