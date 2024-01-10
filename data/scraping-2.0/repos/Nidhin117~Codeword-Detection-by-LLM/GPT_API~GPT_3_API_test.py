import openai
import json

file = open("API_Key.txt",'r')
API_KEY=file.readline()
openai.api_key = API_KEY
model_name = "gpt-3.5-turbo"
# list models
#models = openai.Model.list()
#print(models)
fewshot_msg=[
        {"role": "system", "content": "You are an investigator looking for code word for drugs hidden in sentences. Learn from the given examples and answer with a No if code word is not present. If code word is present, identify code word and what it refers to"},
        {"role": "user", "content": "Lol, that shit is probably so stepped on you can't even call it coke anymore."},
        {"role": "assistant", "content": "Present: Yes, Code word : coke, Code word meaning : Cocaine"},
        {"role": "user", "content": "No one would resist a pot of soup"},
        {"role": "assistant", "content": "Present: No,"}
    ]
#
#Add space in the end after question mark
# task_sentences=f"""
# 1)My cousin did the same and when the legalized pot in dc they really started cracking down in virginia and maryland. Present: ?
# 2)for all vendors of coke it seems pretty obvious that it is not as pure as they market it. Present: ?
# 3)i understand this is to get more customers but imo its bullshit. Present: ?
# 4)is there any tests that show the real purity of the grass vendors sell. Present: ?
# 5)i know whit doc got caught with 50 peruvian,but is there tests of other vendors product. Present: ? 
# """
task_sentences = "for all vendors of coke it seems pretty obvious that it is not as pure as they market it. ?"
task_msg= {"role": "user", "content":task_sentences }
fewshot_msg.append(task_msg)

print(fewshot_msg)
# json_string = json.dumps(task_msg)
# print(json_string)
response = openai.ChatCompletion.create(
    model=model_name,
    messages=fewshot_msg,
    temperature=0,
)

#print(response)
print(response["choices"][0]["message"]["content"])