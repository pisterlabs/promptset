#sentiment dataset + reasoning generation by large model to guide small model Part1 data preprocessing by Yuanyi Hu
from datasets import load_dataset
#load the raw sentiment dataset
data = load_dataset('zeroshot/twitter-financial-news-sentiment')
#rename the dataset
data = data.rename_column("text", "input")
data = data.rename_column("label", "output")
#if need to load the data then uncomment the following codes:
#from datasets import load_from_disk
#data = load_from_disk('C:/Users/Hu Yuanyi/Desktop/capstone_dataset')

from IPython.display import clear_output
import openai
api_key = input()
openai.api_key = api_key
#hide the personal info
clear_output()

def prompt_convert_func(prompt):
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=prompt,
        max_tokens=50
    )
    reasoning_output = response.choices[0].text
    return reasoning_output

#an example about how to use it:
#Define the prompt (use this template for sentiment analysis, otherwise it might not work well based on prev experiments)
prompt_example = ('''Please explain why the ouput is 0, 1, or 2
         (0 means 'negative' emotion, 1 means 'neutral' emotion, 2 means 'positive' emotion by reading
         the following tweets about marktet trend:''' + '<input>' + str(data['train'][6]['input']) + '<output>'
         + str(data['train'][6]['output']))
#remove any newline characters since we will put it into a new column
prompt_convert_func(prompt_example).replace("\n", "")

#now repeat the approach for raw dataset:
data_size = len(data['train'])
#use the following code if you want to use a part of data
#data_size = 1000

train_instruction = []
for i in range(len(data['train'].select(list(range(0, data_size))))):
    prompt_tmp = ('''Please explain why the ouput is 0, 1, or 2
         (0 means 'negative' emotion, 1 means 'neutral' emotion, 2 means 'positive' emotion by reading
         the following tweets about marktet trend:''' + '<input>' + str(data['train'][i]['input']) + '<output>'
         + str(data['train'][i]['output']))
    train_instruction.append(prompt_convert_func(prompt_tmp).replace("\n", ""))
    if i % 100 == 0:
        print("The current iteration is", i)
#merge them together:
sub_train = data['train'].select(list(range(0, data_size)))
sub_train = sub_train.add_column("instruction", train_instruction)

#repeat for validation dataset
validation_instruction = []
for i in range(len(data['validation'].select(list(range(0, data_size))))):
    prompt_tmp = ('''Please explain why the ouput is 0, 1, or 2
         (0 means 'negative' emotion, 1 means 'neutral' emotion, 2 means 'positive' emotion by reading
         the following tweets about marktet trend:''' + '<input>' + str(data['validation'][i]['input']) + '<output>'
         + str(data['validation'][i]['output']))
    validation_instruction.append(prompt_convert_func(prompt_tmp).replace("\n", ""))
    if i % 100 == 0:
        print("The current iteration is", i)

#merge them together:
sub_validation = data['validation'].select(list(range(0, data_size)))
sub_validation = sub_validation.add_column("instruction", validation_instruction)

from datasets.dataset_dict import DatasetDict
from datasets import Dataset

d = {'train':sub_train,
     'validation':sub_validation
     }

result_dataset = DatasetDict(d)

result_dataset

data = result_dataset
#save the preprocessed dataset:
data.save_to_disk('C:/Users/Hu Yuanyi/Desktop/capstone_dataset')
