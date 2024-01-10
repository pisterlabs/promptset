import asyncio

import openpyxl
from openai import AsyncOpenAI

architecture_style = '''find an appropriate architectural style can summarize this heritage '''
architecture_period = '''find an appropriate architectural period can summarize this heritage '''
architecture_texture = '''find an appropriate architectural texture can summarize this heritage'''
architecture_method = '''find an appropriate building method can summarize this heritage'''
architecture_achievement = '''summarize this heritage most architecture achievement'''
architecture_function = '''select a function from religious, political, or social purposes that can summerize this heriatge'''
architecture_ornamentation = '''Architecture encompasses the visual and aesthetic qualities of a structure, including 
its colors, textures, ornamentation, and decorative elements. Summarize this heritage's ornamentation'''
architecture_ingenuity = '''architecture within world cultural heritage sites often reflects the 
technological advancements and innovative solutions developed by ancient civilizations. The use of advanced construction
 techniques, sophisticated engineering systems, and artistic craftsmanship demonstrates the ingenuity and technical 
 expertise of the past. summarize this heritage's most important ingenuity '''

architecture_prompts = [architecture_style, architecture_period, architecture_texture, architecture_method,
                        architecture_achievement, architecture_function,
                        architecture_ornamentation, architecture_ingenuity]

problem_set = {
    "architecture_style": architecture_style,
    "architecture_period": architecture_period,
    "architecture_texture": architecture_texture,
    "architecture_method": architecture_method,
    "architecture_achievement": architecture_achievement,
    "architecture_function": architecture_function,
    "architecture_ornamentation": architecture_ornamentation,
    "architecture_ingenuity": architecture_ingenuity,
}


def architecture_generator():
    return problem_set


async def process_entity(client, text, heritage):
    prompt = f"""
    Your task is to analyze the attributes of world heritage.
    Heritage: ```{heritage}```
    Classify task: ```{text}```
    Output limitations:
    1. Return the most concise answer less than 5 words
    """

    #  removed If you can't find a proper option, provide an answer without quotes.
    prompt = {"role": "user", "content": prompt}
    c = await call_openai(client, prompt)
    answer = c.choices[0].message.content
    return answer.strip('.').lower()


async def process_heritage(client, heritage, header_dict, i, sheet, wb):
    prompts = architecture_generator()
    for attr, prompt in prompts.items():
        answer = await process_entity(client, prompt, heritage)
        # print(f"{i} finished")
        position = header_dict[attr]  # Get the insert position
        position = openpyxl.utils.column_index_from_string(position)
        cell = sheet.cell(row=2 + i, column=position)
        cell.value = answer
        wb.save("heritage.xlsx")


async def call_openai(client, s):
    c = await client.chat.completions.create(
        messages=[s],
        model='gpt-3.5-turbo'
    )
    return c


async def main():
    import yaml
    with open('config.yml', 'r') as yaml_file:
        config = yaml.safe_load(yaml_file)
        client = AsyncOpenAI(
            api_key=config['credentials']['api_key'],
            base_url="https://api.chatanywhere.com.cn/v1",
            max_retries=3
        )
    file = "heritage.xlsx"
    wb = openpyxl.load_workbook(file)
    sheet = wb.active

    header_row = sheet[1]
    header_dict = {cell.value: cell.column_letter for cell in header_row}

    heritages = [cell.value for cell in sheet['C'][1:]]
    sorts = [cell.value for cell in sheet['D'][1:]]

    tasks = []
    for i, heritage in enumerate(heritages):
        sort_ = sorts[i]
        if 'cultural' in sort_:
            task = process_heritage(client, heritage, header_dict, i, sheet, wb)
            tasks.append(task)

    await asyncio.gather(*tasks)

    wb.close()
    print("fin")


asyncio.run(main())
