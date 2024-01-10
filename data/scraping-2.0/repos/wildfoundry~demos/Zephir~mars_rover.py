from langchain.llms import Ollama
from langchain.prompts import PromptTemplate
import json


def get_sensor_data():
    # implement here image captioning mechanism for a cameras
    sensor_data_no_mineral = "Front camera: rocks\nRight camera: clear\nLeft camera: crater\nRear camera: clear"
    sensor_data_mineral = "Front camera: clear\nRight camera: mineral\nLeft camera: crater\nRear camera: rocks"
    sensor_data_mineral_hidden = "Front camera: inaccessible rocks, mineral after it\nRight camera: rocks\nLeft camera: clear\nRear camera: clear"
    sensor_data_return_to_base = "Front camera: clear\nRight camera: base\nLeft camera: crater\nRear camera: inaccessible rocks."

    state = 'exploration'

    return sensor_data_no_mineral, state


def move(llm_output):
    direction = json.loads(llm_output)["direction"]
    # here implement Mars rover movement logic
    print(f"Moving to {direction} direction.")


llm = Ollama(
    model="zephyr:7b-beta-q5_1",
    temperature=0,
)

with open('prompts/exploration.prompt', 'r') as file:
    prompt_template = file.read()
analyzer_prompt = PromptTemplate.from_template(prompt_template)

with open('prompts/mineral_gathering.prompt', 'r') as file:
    prompt_template = file.read()
gathering_prompt = PromptTemplate.from_template(prompt_template)

with open('prompts/return.prompt', 'r') as file:
    prompt_template = file.read()
return_to_base_prompt = PromptTemplate.from_template(prompt_template)


if __name__ == '__main__':
    sensor_data, state = get_sensor_data()

    if state == 'exploration':
        chain = analyzer_prompt | llm
    elif state == 'mineral_gathering':
        chain = gathering_prompt | llm
    else:   # state == 'return_to_base'
        chain = return_to_base_prompt | llm

    result = chain.invoke({"sensor_data": sensor_data})
    print(result)

    move(result)
