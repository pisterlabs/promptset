import sys
from langchain.chat_models import ChatOpenAI
from langchain import LLMChain
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    AIMessagePromptTemplate,
)
chat = ChatOpenAI(temperature=0)

system_message_template = "Question: Convert pronouns such as `he` to a neutral variable `${{p.she}}`. Take note of capitalization.\n\nCode for the pronouns:\n```\nlet p = {{\n  sdesc: 'this character', // Bob\n  Sdesc: 'This character', // Bob\n  sdescs: \"this character's\",\n  Sdescs: \"This character's\",\n  him: 'him/her',\n  Him: 'Him/her',\n  his: 'his/her',\n  His: 'His/her',\n  she: 'he/she',\n  She: 'He/she',\n  herself: 'him/herself',\n  Herself: 'His/herself',\n  hers: 'his/hers',\n  Hers: 'His/Hers'\n}};\n```\n\n"
system_message_prompt = SystemMessagePromptTemplate.from_template(system_message_template)

user_message_example = HumanMessagePromptTemplate.from_template("Genderize: She is constantly building bridges between two sides. She never takes sides and always tries to find the good in everyone. Her optimism and positivity has made her popular.")
assistant_message_example = AIMessagePromptTemplate.from_template("${{p.She}} is constantly building bridges between two sides. ${{p.She}} never takes sides and always tries to find the good in everyone. ${{p.His}} optimism and positivity has made ${{p.him}} popular.")

user_message_template = "Genderize: {text}"
user_message_prompt = HumanMessagePromptTemplate.from_template(user_message_template)

chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, user_message_example, assistant_message_example, user_message_prompt])

chain = LLMChain(llm=chat, prompt=chat_prompt)

if __name__ == "__main__":
    # Assuming the prompt is the second argument when running the script in zsh
    # Example usage: python genderize.py "She is constantly building bridges between two sides. She never takes sides and always tries to find the good in everyone. Her optimism and positivity has made her popular."
    text = sys.argv[1]
    response = chain.run(text=text)
    print(response)
