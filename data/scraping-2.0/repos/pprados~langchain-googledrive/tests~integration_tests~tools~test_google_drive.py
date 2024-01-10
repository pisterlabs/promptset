from langchain_googledrive.tools.google_drive.tool import GoogleDriveSearchTool
from langchain_googledrive.utilities.google_drive import GoogleDriveAPIWrapper


def test_google_drive() -> None:
    folder_id = "root"

    # By default, search only in the filename.
    tool = GoogleDriveSearchTool(
        api_wrapper=GoogleDriveAPIWrapper(
            folder_id=folder_id,
            num_results=2,
            template="gdrive-query-in-folder",  # Search in the body of documents
        )
    )
    import logging

    logging.basicConfig(level=logging.INFO)
    print(tool.run("machine learning"))
    print(tool.description)

    # %%
    from langchain import OpenAI
    from langchain.agents import AgentType, initialize_agent

    llm = OpenAI(temperature=0)
    agent = initialize_agent(
        tools=[tool],
        llm=llm,
        agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
    )
    print(agent.run("Search in google drive, who is 'Yann LeCun' ?"))
