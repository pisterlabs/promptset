import os
import openai
from ratelimit import limits, sleep_and_retry

from Defense.LMDefense import LMDefense
from utils import read_data,write_data
from tqdm import tqdm 

openai.api_key = os.getenv("OPENAI_API_KEY")

class GPT3Defense(LMDefense):
    def __init__(self, data_path,data_purity) -> None:
        super().__init__()
        self.data_path = data_path
        self.data_purity = data_purity
        self.train_data = read_data(os.path.join(data_path,data_purity),'train')
        self.dev_data = read_data(os.path.join(data_path,data_purity),'dev')
        self.test_data = read_data(os.path.join(data_path,data_purity),'test')

        self.train_defend_data_path = (os.path.join(self.data_path,'gpt3defend_'+data_purity),'train.tsv')
        self.dev_defend_data_path = (os.path.join(self.data_path,'gpt3defend_'+data_purity),'dev.tsv')
        self.test_defend_data_path = (os.path.join(self.data_path,'gpt3defend_'+data_purity),'test.tsv')


    def paraphrase_defend(self):
        self.train_data = self.paraphrase_dataset(self.train_data)
        # self.dev_data = self.paraphrase_dataset(self.dev_data)
        # self.test_data = self.paraphrase_dataset(self.test_data)

        write_data(self.train_defend_data_path[0],self.train_defend_data_path[1],self.train_data)
        write_data(self.dev_defend_data_path[0],self.dev_defend_data_path[1],self.dev_data)
        write_data(self.test_defend_data_path[0],self.test_defend_data_path[1],self.test_data)       

    def paraphrase_dataset(self,dataset):
        for i in tqdm(range(0,len(dataset),20)):
            prompts = ["Paraphrase: "+dt[0] for dt in dataset[i:i+20]]
            outputs = self.call_openai_gpt(prompts=prompts)
            for k,choice in enumerate(outputs['choices']):
                text = choice['text'].replace("\n", "")
                dataset[i+k] = (text, dataset[i+k][1])
        return dataset

    @sleep_and_retry
    @limits(calls=30, period=65)
    def call_openai_gpt(self,prompts):
        # model_name = "text-davinci-003"
        # model_name = "text-ada-001"
        model_name = "text-babbage-001"
        outputs = openai.Completion.create(
            model=model_name,
            prompt=prompts,
            temperature=0.7,
            max_tokens=50
        )
        return outputs
        

            

