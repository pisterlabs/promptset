import os
print(os.getcwd())
basename = os.path.basename(os.getcwd())
if basename == 'receipt_parse':
    os.chdir(os.path.dirname(os.getcwd())) # this is important we have to change the working directory back one
elif 'ICS-438' in basename:
    os.chdir(os.path.join(os.getcwd(), 'receiptParse')) # 
print(os.getcwd())


import os
from glob import glob
import json

#### MODEL ####
from langchain.chat_models import ChatOpenAI
# jnicolowathawiiAPIkey = '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'
os.environ['OPENAI_API_KEY'] = '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'
model = ChatOpenAI(model='gpt-3.5-turbo')

#### Prompt ####
from langchain.prompts import PromptTemplate
from langchain.prompts.few_shot import FewShotPromptTemplate

# get examples
examples = []
jsonFiles = glob(os.path.join(os.path.join('data', 'receipts', 'json', 'actual'), '*.json'))
for jsonFile in jsonFiles:
    baseFn = os.path.basename(jsonFile.replace('.json', ''))
    txtFile = glob(os.path.join('data', 'receipts', 'text', f'{baseFn}.txt'))[0]
    with open(jsonFile, 'r') as f: JSONobj = f.read()
    with open(txtFile, 'r') as f: rawRecieptText = f.read()
    exampleDict = {
        "rawRecieptText": rawRecieptText,
        "JSONobj":JSONobj.replace('{', '{{{{').replace('}', '}}}}')
        }
    examples.append(exampleDict)


for promptTemplateFile in glob(os.path.join('receipt_parse', 'prompt_templates', 'prompt_template_*.txt')):
  print(promptTemplateFile)
  if promptTemplateFile.endswith('1.txt'): continue # already processed reciepts with this prompt
  with open(promptTemplateFile, 'r') as f: promptTemplate = f.read()
  example_prompt = PromptTemplate(input_variables=["rawRecieptText", "JSONobj"], 
                              template=promptTemplate)

  prompt = FewShotPromptTemplate(
      examples=[examples[0]],
      example_prompt=example_prompt,
      suffix="Get JSON for this:\n{input}",
      input_variables=["input"]
  )

#   print(prompt.format(input="recieptTxt"))



  #### Create Chain ####
  chain = prompt | model # how to pass the prompt to the model (pipe prompt to model)

  #### Run inference on reciepts ####
  promptName = os.path.basename(promptTemplateFile).split('.txt')[0].replace('_', '')#.split('_')[-1] e.g. prompt_template_1
  recieptFiles = glob(os.path.join('data', 'receipts', 'text', '*.txt'))
  for recieptFn in recieptFiles:
    print(recieptFn)
    saveJson = os.path.join('data', 'receipts', 'json', f'{os.path.basename(recieptFn).split(".tx")[0]}_{promptName}.json')
    if os.path.exists(saveJson):continue
    with open(recieptFn, 'r') as f: recieptTxt = f.read()
    response = chain.invoke({'input':recieptTxt})
    # print(response)
    try:
        data_dict = json.loads(response.content)
    except(json.decoder.JSONDecodeError):
        print(response.content)
    else:
        with open(saveJson, 'w') as f: json.dump(data_dict, f)
