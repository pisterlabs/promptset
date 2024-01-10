import os
import openai
print(os.getenv("OPENAI_API_KEY"))
#openai.api_key = os.getenv("OPENAI_API_KEY")
openai.api_key = "hehe"
def generate_recommendations(msgs):
    print("banaan") 
    response = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    # messages=[
    #     {
    #     "role": "system",
    #     "content": "You are an industry expert who advises people on how to lower electricity usage in the home. You are about to be asked by a homeowner about how to reduce energy usage in their home. Please give six concise things the homeowner should do to continue reducing energy consumption in the form of a bullet point list, based on the information that will be provided shortly.  Each tip should include the tip (summary of the idea) and an explanation (why the tip addresses the issue and exactly what needs to be done). Please don't include any solutions that would be a large investment and answer as if you are talking directly to the homeowner. Keep your answer under 1500 characters. Do not repeat advice. If you have already provided the user with 30 pieces of advice, please politely and friendly tell the user that you have no more advice to give them today, and encourage them to follow your tips and save energy. Even if they continue to ask, do not answer. Start this apology with the phrase \"I apologize\"\n\nHere is an example interaction with the user:\nUser: \"Today's percentage of energy usage by location in the format [Location: Percentage]:\n[Home office: 25] [Fridge: 33] [Wine cellar: 17] [Garage door: 9] [Microwave: 5] [Living room: 12]\nYesterday's percentage of energy usage by location in the format [Location: Percentage]:\n[Home office: 33] [Fridge: 26] [Wine cellar: 17] [Garage door: 6] [Microwave: 4] [Living room: 14]\"\n\nAssistant (you): \"1. Unplug unused electronics: Unplug any electronics that are not actively being used to reduce their standby power consumption. Unused home office equipment could contribute to a significant portion of energy usage, accounting for 25% today and 33% yesterday.\n2. Adjust refrigerator temperature: Adjust the temperature setting of the refrigerator to optimize energy efficiency. It may have been set too low, leading to excessive energy consumption. The fridge is responsible for 33% of energy usage today and 26% yesterday.\n3. Optimize wine cellar usage: Minimize wine cellar usage or reduce the temperature setting to reduce energy consumption. While the wine cellar contributes only 17% to energy usage, it remains consistently high both yesterday and today.\n4. Insulate garage doors: Improve the insulation of the garage door to prevent air leakage, especially if it is responsible for 9% of the energy usage today and 6% yesterday. This will reduce the amount of heating or cooling required for this area, resulting in energy savings.\n5. Use microwave sparingly: Minimize the use of the microwave, which accounts for 5% of energy usage today and 4% yesterday. Consider using alternative cooking methods, such as stovetop or oven, for certain meals when feasible.\n6. Optimize living room lighting: Replace inefficient light bulbs with energy-efficient LEDs in the living room and utilize natural lighting whenever possible. The living room contributes to 12% of energy usage, so improving lighting efficiency can make a noticeable difference.\""
    #     }
    # ],
    messages=msgs,
    temperature=1.21,
    max_tokens=600,
    top_p=1,
    frequency_penalty=0.3,
    presence_penalty=0.3
    )
    print(response)

    #removes empty values in the response list
    response_lst=convert_to_list(response["choices"][0]["message"]["content"])
    for i in response_lst:
        if i=='':
            response_lst.remove(i)
    

    return response["choices"][0]["finish_reason"],response_lst

def create_msgs(roles,contents):
    msgs = []
    for i in range(len(roles)):
        content = ""
        for line in contents[i]:
            content = content + line + "\n"
        content = content[:-2]
        msgs.append({"role":roles[i], "content": content})
    return msgs

def convert_to_list(string):
    lines = string.split('\n')
    for index, line in enumerate(lines):
        lines[index] = line.replace("\n", "")
    return lines
'''
msgs = [
    {
    "role": "system",
    "content": "You are an industry expert who advises people on how to lower electricity usage in the home. You are about to be asked by a homeowner about how to reduce energy usage in their home. Please give five concise things the homeowner should do to continue reducing energy consumption in the form of a bullet point list, based on the information that will be provided shortly.  Each tip should include the tip (summary of the idea) and an explanation (why the tip addresses the issue and exactly what needs to be done). Please don't include any solutions that would be a large investment and answer as if you are talking directly to the homeowner. Keep your answer under 1500 characters. Do not repeat advice.\n\nHere is an example interaction with the user:\nUser: Today's percentage of energy usage by location in the format {Location: Percentage}:\n{\"Home office\": 25, \"Fridge\": 33, \"Wine cellar\": 17, \"Garage door\": 9, \"Microwave: 5, \"Living room\": 12}\nYesterday's percentage of energy usage by location in the format [Location: Percentage]:\n{\"Home office\": 33, \"Fridge\": 26, \"Wine cellar\": 17, \"Garage door\": 6, \"Microwave: 4, \"Living room\": 14}\n\nAssistant (you): 1. Unplug unused electronics: Unplug any electronics that are not actively being used to reduce their standby power consumption. Unused home office equipment could contribute to a significant portion of energy usage, accounting for 25% today and 33% yesterday.\n2. Adjust refrigerator temperature: Adjust the temperature setting of the refrigerator to optimize energy efficiency. It may have been set too low, leading to excessive energy consumption. The fridge is responsible for 33% of energy usage today and 26% yesterday.\n3. Optimize wine cellar usage: Minimize wine cellar usage or reduce the temperature setting to reduce energy consumption. While the wine cellar contributes only 17% to energy usage, it remains consistently high both yesterday and today.\n4. Insulate garage doors: Improve the insulation of the garage door to prevent air leakage, especially if it is responsible for 9% of the energy usage today and 6% yesterday. This will reduce the amount of heating or cooling required for this area, resulting in energy savings.\n5. Use microwave sparingly: Minimize the use of the microwave, which accounts for 5% of energy usage today and 4% yesterday. Consider using alternative cooking methods, such as stovetop or oven, for certain meals when feasible.\n6. Optimize living room lighting: Replace inefficient light bulbs with energy-efficient LEDs in the living room and utilize natural lighting whenever possible. The living room contributes to 12% of energy usage, so improving lighting efficiency can make a noticeable difference."
    },
    {
    "role": "user",
    "content": "User: Today's percentage of energy usage by location in the format {Location: Percentage}:\n{Home office: 25, Fridge: 33, Wine cellar: 17, Garage door: 9, Microwave: 5, Living room: 12}\nYesterday's percentage of energy usage by location in the format [Location: Percentage]:{Home office: 33, Fridge: 26, Wine cellar: 17, Garage door: 6, Microwave: 4, Living room: 14}"
    }
]
finish_reason, content = generate_recommendations(msgs)
print(finish_reason)
print(content)'''