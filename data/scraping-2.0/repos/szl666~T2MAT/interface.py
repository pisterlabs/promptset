import re
import os
import yaml
import openai


def get_key_words(text):
    openai.api_key = "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"
    model_engine = "text-davinci-003"
    prompt = f"Distill material properties from this sentence in the format: property: value/value range(If the range exists, it should be formatted in '[a, b] value unit', possible property list:[bulk modulus, band gap, Ehull]) :{text}"
    completions = openai.Completion.create(
        engine=model_engine,
        prompt=prompt,
        max_tokens=1024,
        n=1,
        stop=None,
        temperature=0.5,
    )
    return completions.choices[0].text


def write_input(message):
    with open('input_tem.yaml', 'r') as f:
        params = yaml.safe_load(f)
    possi_property_list = ['Bulk modulus', 'Band gap']
    property_dict = {}
    for prop in possi_property_list:
        matches = re.findall(f"{prop}: \[.*?\]", message)[0].strip(f'{prop}: ')
        property_dict[prop] = eval(matches)
    params['property'] = list(property_dict.keys())
    params['property_range'] = list(property_dict.values())
    with open("input.yaml", "w") as f:
        yaml.dump(params, f)


def main():
    program_name = "ONEGA"
    print("=" * 40)
    print("=" * 40)
    print('\n')
    print(" " * 12 + "WELCOME TO " + program_name)
    print('\n')
    print("=" * 40)
    print("=" * 40)
    text = input("请输入您的材料设计需求：")
    # text = '生成一批体模量在100-200GPa之间并且带隙在0到1eV之间的材料'
    message = get_key_words(text)
    print(f"请确认您的需求: {message}")
    input()
    print("开始逆向设计")
    write_input(message)
    # settings = read_yaml('input.yaml')
    # run_onega(settings)
    # os.system('run_onega_con')


if __name__ == '__main__':
    main()
