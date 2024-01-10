from langchain.prompts import (
    AIMessagePromptTemplate,
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)

from txt2cmd.filer import read

new_script = ChatPromptTemplate.from_messages(
    [
        SystemMessagePromptTemplate.from_template(read("txt2cmd/templates/chat/system.txt")),
        HumanMessagePromptTemplate.from_template(read("txt2cmd/templates/chat/human_create.txt")),
    ]
)
update_script = ChatPromptTemplate.from_messages(
    [
        SystemMessagePromptTemplate.from_template(read("txt2cmd/templates/chat/system.txt")),
        HumanMessagePromptTemplate.from_template(read("txt2cmd/templates/chat/human_update.txt")),
        AIMessagePromptTemplate.from_template("How would you like to modify the {language} script?"),
        HumanMessagePromptTemplate.from_template("{user_prompt}"),
    ]
)
