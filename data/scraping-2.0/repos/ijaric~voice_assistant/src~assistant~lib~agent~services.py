import logging
import uuid

import langchain.agents
import langchain.agents.format_scratchpad
import langchain.agents.output_parsers
import langchain.chains
import langchain.chat_models
import langchain.memory
import langchain.memory.chat_memory
import langchain.prompts
import langchain.schema
import langchain.tools.render

import lib.agent.repositories as lib_agent_repositories
import lib.agent.repositories.chat_repository as chat_repositories
import lib.app.settings as app_settings
import lib.models as models


class AgentService:
    def __init__(
        self,
        settings: app_settings.Settings,
        chat_repository: chat_repositories.ChatHistoryRepository,
        tools: lib_agent_repositories.OpenAIFunctions,
    ) -> None:
        self.settings = settings
        self.tools = tools
        self.chat_repository = chat_repository
        self.logger = logging.getLogger(__name__)

    async def process_request(self, request: models.AgentCreateRequestModel) -> models.AgentCreateResponseModel:
        # Get session ID
        session_request = models.RequestLastSessionId(channel=request.channel, user_id=request.user_id, minutes_ago=3)
        session_id = await self.chat_repository.get_last_session_id(session_request)
        if not session_id:
            session_id = uuid.uuid4()

        # Declare tools (OpenAI functions)
        tools = [
            langchain.tools.Tool(
                name="GetMovieByDescription",
                func=self.tools.get_movie_by_description,
                coroutine=self.tools.get_movie_by_description,
                description="Use this function to find data about a movie by movie's description",
            ),
        ]

        template = """
        1. You are movie expert with a vast knowledge base about movies and their related aspects.
        2. Use functions to get an additional data about movies.
        3. Translate each inbound request into English language. Before calling any functions.
        4. Answer always in Russian language.
        5. Be very concise. You answer must be no longer than 100 words."""

        prompt = langchain.prompts.ChatPromptTemplate.from_messages(
            [
                ("system", template),
                langchain.prompts.MessagesPlaceholder(variable_name="chat_history"),
                ("user", "{input}"),
                langchain.prompts.MessagesPlaceholder(variable_name="agent_scratchpad"),
            ]
        )

        llm = langchain.chat_models.ChatOpenAI(
            temperature=self.settings.openai.agent_temperature,
            openai_api_key=self.settings.openai.api_key.get_secret_value(),
            model=self.settings.openai.model,
        )

        agent_kwargs = {
            "extra_prompt_messages": [langchain.prompts.MessagesPlaceholder(variable_name="memory")],
        }
        memory = langchain.memory.ConversationBufferMemory(memory_key="chat_history", return_messages=True)

        # Load chat history from database
        request_chat_history = models.RequestChatHistory(session_id=session_id)
        chat_history = await self.chat_repository.get_messages_by_sid(request_chat_history)
        if chat_history:
            for entry in chat_history:
                if entry.role == "user":
                    memory.chat_memory.add_user_message(entry.content)
                elif entry.role == "agent":
                    memory.chat_memory.add_ai_message(entry.content)

        agent = langchain.agents.OpenAIFunctionsAgent(llm=llm, tools=tools, prompt=prompt)
        agent_executor: langchain.agents.AgentExecutor = langchain.agents.AgentExecutor.from_agent_and_tools(
            tools=tools,
            agent=agent,
            agent_kwargs=agent_kwargs,
            memory=memory,
        )

        response = await agent_executor.arun({"input": request.text})

        # Save user request and AI response to database
        user_request = models.RequestChatMessage(
            session_id=session_id,
            user_id=request.user_id,
            channel=request.channel,
            message={"role": "user", "content": request.text},
        )
        ai_response = models.RequestChatMessage(
            session_id=session_id,
            user_id=request.user_id,
            channel=request.channel,
            message={"role": "assistant", "content": response},
        )

        await self.chat_repository.add_message(user_request)
        await self.chat_repository.add_message(ai_response)

        return models.AgentCreateResponseModel(text=response)
