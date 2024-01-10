from langchain.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)

from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain

import sys
import time

WINDOW = 50

def translate(text, input="English", output="Portuguese"):
    template="You are a helpful assistant that translates from {input_language} to {output_language}. Translate without changing the original format."
    system_message_prompt = SystemMessagePromptTemplate.from_template(template)

    human_template="{text}"
    human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)
    chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])

    chain = LLMChain(llm=ChatOpenAI(temperature=0, model_name='gpt-3.5-turbo'), prompt=chat_prompt)

    # response = chain.run({"input_language": "Spanish", "output_language": "Portuguese", "text" :"""
    # 1
    # 00:01:32,885 --> 00:01:36,388
    # <i>Cuando era niño, mi maestro me preguntó</i>

    # 2
    # 00:01:36,388 --> 00:01:40,976
    # <i>cuáles serían las tres cosas que salvaría
    # si mi casa se incendiara.</i>

    # 3
    # 00:01:43,854 --> 00:01:50,277
    # <i>Le respondí: "Mi cuaderno de dibujo,
    # mi álbum de AC/DC y a mi gato Enojón".</i>

    # 4
    # 00:01:57,910 --> 00:02:00,412
    # <i>No mencioné a mis papás ni a mi hermana.</i>

    # 5
    # 00:02:01,288 --> 00:02:03,498
    # <i>Los otros niños sí lo hicieron.</i>

    # 6
    # 00:02:04,791 --> 00:02:06,793
    # <i>¿Eso me hace una mala persona?</i>

    # 7
    # 00:02:11,840 --> 00:02:13,467
    # <i>Mi gato se murió,</i>
    # """})

    response = chain.run({"input_language": input, "output_language": output, "text" : text})
    return response



if __name__ == '__main__':
    if len(sys.argv) > 1:
        fname = sys.argv[1]
    else:
        fname = "test.srt" 
    with open(fname, encoding='utf-8-sig') as file:
        content = file.readlines()
    try:
        translated_text = ""
        while len(content) > 0:
            chunk = []
            while len(content) > 0:
                line = content.pop(0)
                chunk.append(line)
                if line.strip().isnumeric():
                    if len(chunk) > WINDOW:
                        content.insert(0, line)
                        chunk.pop()
                        break
            time.sleep(5)
            text = ''.join(chunk)
            response = translate(text, input="Spanish")
            translated_text += f"{response}\n\n"
    except Exception as ex:
        print(ex)

print(translated_text)
