from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.agents import initialize_agent, Tool, AgentType
from application.tools.tool import get_information
from decouple import config


def lookup(name: str) -> str:
    llm = ChatOpenAI(api_key=config("OPENAI_API_KEY"), temperature=0, model_name="gpt-3.5-turbo-16k")

    template = """
    Verilen tam ad: {name_of_person} ile bir Google araması yapın ve:
    1. LinkedIn gibi profesyonel profilleri bulun.
    2. Instagram gibi sosyal medya profillerini arayın.
    3. En alakalı veya ilginç bulguları kısa bir şekilde özetleyin.
    """

    tools_for_agent = [
        Tool(
            name="Crawl Google 4 for altinbas university gdsc",
            func=get_information,
            description="It is useful when searching by person's name.",
        )
    ]

    agent = initialize_agent(
        tools=tools_for_agent,
        llm=llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
    )

    # verbose = True diyorum bu ajanın daha önceki süreçleri loglamasını ve muhakeme yeteneği kazanmasını sağlayacak.

    prompt_template = PromptTemplate(
        template=template, input_variables=["name_of_person"]
    )

    community_event_url = agent.run(prompt_template.format_prompt(name_of_person=name))

    return community_event_url
