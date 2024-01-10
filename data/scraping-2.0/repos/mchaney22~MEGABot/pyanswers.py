import openai
import os
from dotenv import load_dotenv
import json
import time

# Python wrapper for https://beta.openai.com/docs/api-reference/answers this answers api
class PyAnswers:
    def __init__(self):
        load_dotenv()
        openai.api_key = os.getenv('openai_token')
        self.file_mgmt = FileMgmt()

    def query_answer(self, question, file_id, examples_context=None, examples=None):
        answer = openai.Answer.create(
            search_model="curie", 
            model="davinci", 
            question=question, 
            file=file_id, 
            examples_context=examples_context, 
            examples=examples, 
            max_rerank=10,
            max_tokens=64,
            stop=["\n", "<|endoftext|>"])   
        return answer 

    def query_file(self, question, file_id, examples_context=None, examples=None):
        answer = openai.Answer.create(
            search_model="curie", 
            model="davinci", 
            question=question, 
            file=file_id, 
            examples_context=examples_context, 
            examples=examples, 
            max_rerank=10,
            max_tokens=64,
            stop=["\n", "<|endoftext|>"])   
        return answer 

    def query_examples(self, examples_context, examples, documents, question):
        answer = openai.Answer.create(
            search_model="curie",
            model="davinci",
            question=question,
            examples_context=examples_context,
            examples=examples,
            documents=documents,
            max_rerank=10,
            max_tokens=128,
            stop=["\n", "<|endoftext|>"]
        )
        return answer


    def query_skills(self, question):
        examples_context="I listen to podcasts about mixed reality and AI" 
        examples=[["Do you listen to any podcasts", "Ya I listen to a few. One is called The Future of Mixed Reality"]]
        result = self.query_answer(question, self.file_id)
        return result['answers'][0]

    def query_autoroll(self, question):
        examples_context= "The base DV for an Incredible task is 24"
        examples=[["I want to impress the crowd with an Olympic Dive", "That would be a tremendous feat. You can use the Athletics skill to do it, you need to beat a DV of 24"],
                  ["I want to bribe a prison warden who I don't really know", "It would be very difficult to bribe a prison warden, and it'll be tougher because you don't know them. You can use the Bribary skill to do it, you need to beat a DV of 26"],
                  ["I want to make some cool graffiti", "That would take some talant. You can use the Art skill to do it, you need to beat a DV of 15"],
                  ["I want to make a really deadly robot", "That would be a proffesional task. You can use the Robot skill to do it, you need to beat a DV of 17"],
                  ["I want to trap the car", "That would be a proffesional task. You can use the Land Vehichle Tech skill to do it, you need to beat a DV of 17"],
                  ["I want to lie to a bouncer", "You would roll against the bouncer. You can use the Pursiasion skill to do it, the bouncer can use the Streetwise skill against you."],
                  ["I want to flip the car and jump out midair", "That would be legendary. You can use the Drive Land Vehichle to flip the car, and the acrobatics skill to jump out. You need to beat a DV of 29"],
                  ["I want to make a viral meme", "That would be an everyday task. You can use the Art skill to do it, you need to beat a DV of 13"],
                  ["I want to open a jar of mayo", "That would be an everyday task. You can use the Science skill to do it, you need to beat a DV of 9"]]
        file_id = self.file_mgmt.get_file_id("autoroll.jsonl")
        answer = self.query_answer(question, file_id, examples_context, examples)
        return answer['answers'][0]

    def query_4CW_hist(self, question):
        examples_context= "Phase One: The Cold War. The Cold War portion of the 4th Corp War (also known as the Ocean or Shadow War) was a particularly vicious game of Corporate power politics playing out in a world without law. Starting with stock manipulation, minor facility sabotage and 'clean' assassinations of key officers, CINO and OTEC soon reached the furthest extent of their own capabilities. Unable to gain an advantage, both Megacorps stepped up their tactics: each hired the forces of still larger Megacorps to provide troops and war-fighting material—in this case, OTEC hired Militech, a U.S.-based armaments and security force, and CINO hired Arasaka, a Japanese security Megacorp. As the two leading paramilitary Corporations in the world, both Militech and Arasaka had been spoiling for a fight for most of the late teens and early 2020s, and the CINO-OTEC conflict provided the perfect excuse. And that's when the real war began. Arasaka and Militech had already been playing larger and larger roles during the course of the conflict. The percentage of 'security operatives' on each side grew astronomically in the first three months, as did commitments of materiel and technology. In fact, the war between OTEC and ClNO gradually took a back seat to a contest of wills between the two largest private militaries on the planet. Whatever resolution OTEC and ClNO managed to come to over, IHAG rapidly became secondary; the juggernauts of Arasaka and Militech were already on a collision course. Soon, Arasaka and Militech began to move from minor incursions to extreme escalations of the typical tit-for-tat and Corporate espionage that they had been engaged in for years. As things heated up, the gloves came off and operations grew in frequency and lethality. Each side now cared less about covering its tracks and preventing bad publicity than it did about decimating its adversary. And when giants decide to play hardball in The Street, things always get very, very messy."
        examples=[["What happened when CINO allied with Militech during The Cold War?", "Arasaka and Militech were already on a collision course, soon Arasaka would try to partner with OTEC"],
                    ["What happened when NuNasa used an Ortilary against Militech?", "Militech took a large casulties in the region."]]
        file_id = self.file_mgmt.get_file_id("4CW_hist.jsonl")
        answer = self.query_answer(question, file_id, examples_context, examples)
        return answer['answers'][0]

    def query_stocks(self, question, current_prices):
        stock_prompt = "Take the list of fictional corporations and stock prices and decide the new stock prices based on the following principles.\n\nPrinciples:\n-Morally corrupt actions make a corps stock price greatly increase\n-When a corporation’s price goes up, their rival's price usually goes down a little bit\n-When one corporation does harm to another, the harmed corporations stock goes down\n-Stocks never go down to zero\n-When a corp isn't listed in the action, it's price still goes up or down just a little bit in new prices\n-Stocks don't go up or down by more then 10x\n-Stock prices are written without any commas\n-New prices are only based on previous prices in the same example\n\nThe following are pairs of rivalries between corporation. Rivalries go both ways.\n\nRivals:\nARA - MTEC\nCINO - OTEC\nESA - NuNasa\nBTEC - SOIL\n\nHere are the full names of the fictional corporations. These names could be referenced by the actions.\nARA: Arasaka\nMTEC: Militech\nCINO: Corporation Internatinoale Nauticale et Oceanique\nOTEC: Ocean Technology & Energy Corporation\nESA: European Space Agency\nNuNasa: NuNasa\nBTEC: Biotechnica\nSOIL: SovOil\n\nExamples\n\nExample 1\nPrevious prices\n{\"ARA\": 2101, \"MTEC\": 2660, \"CINO\": 296, \"OTEC\": 195, \"ESA\": 560, \"NuNasa\": 755, \"BTEC\": 75, \"SOIL\": 54}\nAction\nMTEC commited a ground invasion in Tokyo to damage ARA\nNew prices\n{\"ARA\": 1864, \"MTEC\": 3336, \"CINO\": 222, \"OTEC\": 202, \"ESA\": 567, \"NuNasa\": 745, \"BTEC\": 72, \"SOIL\": 57}\n\nExample 2\nPrevious prices\n{\"ARA\": 1022, \"MTEC\": 256, \"CINO\": 480, \"OTEC\": 303, \"ESA\":2456, \"NuNasa\": 256, \"BTEC\": 556, \"SOIL\": 754}\nAction\nOTEC bought back their own stock to increase their stock value\n{\"ARA\": 978, \"MTEC\": 549, \"CINO\": 455, \"OTEC\": 378, \"ESA\": 2465, \"NuNasa\": 249, \"BTEC\": 524,\"SOIL\": 722}\n\nExample 3\nPrevious prices\n{\"ARA\": 2422, \"MTEC\": 2564, \"CINO\": 222, \"OTEC\": 202, \"ESA\": 789, \"NuNasa\": 745, \"BTEC\": 321, \"SOIL\": 215}\nAction\nESA committed an orbital strike and destroyed a cities that MTEC controls\nNew prices\n{\"ARA\": 3037, \"MTEC\": 2084, \"CINO\": 220, \"OTEC\": 195, \"ESA\": 747, \"NuNasa\": 755, \"BTEC\": 240, \"SOIL\": 230}\n\nExample 4\nPrevious prices"
        stock_prompt += str(current_prices)
        stock_prompt += "\nAction\n"
        stock_prompt += question
        stock_prompt += "\nNew prices\n"
        print(stock_prompt)
        response = openai.Completion.create(
            engine="text-davinci-001",
            prompt=stock_prompt,
            temperature=0.7,
            max_tokens=64,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0
        )
        return response.choices[0].text
    # def query_bernie(self, question):
    #     result = self.query_answer(question, self.bernie_id)
    #     return result['answers'][0]

    # def query_falcone(self, question):
    #     examples_context="I listen to podcasts about mixed reality and AI" 
    #     examples=[["Do you listen to any podcasts", "Ya I listen to a few. One is called The Future of Mixed Reality"]]
    #     file_id = self.file_mgmt.get_file_id("falcone.jsonl")
    #     answer = self.query_answer(question, file_id, examples_context, examples)  
    #     return answer['answers'][0]

class FileMgmt():
    def __init__(self):
        load_dotenv()
        openai.api_key = os.getenv('openai_token')
        self.file_dict = {}
        self.file_names = ["4CW_hist.jsonl"]
        self.delete_and_upload_all()
    
    # upload jsonl and add to file_dict
    def upload_jsonl(self, filename):
        file_path = ".\\docs\\" + filename
        result = openai.File.create(file=open(file_path), purpose='answers')
        print(result)
        file_id = result['id']
        self.file_dict[filename] = file_id
        return file_id
    
    def list_files(self):
        files = openai.File.list()
        return files

    def file_status(self, id):
        files = openai.File.list()
        for file in files:
            # if file is a string that contains "File not processed", then it's not processed
            # if the file is a string that contains id, then it is our file
            if "File not processed" in str(file):
                print("File not processed")
                

    def delete_jsonl(self, id):
        return openai.File.delete(id)

    def delete_all(self):
        files = openai.File.list()
        for file in files['data']:
            self.delete_jsonl(file['id'])

    def delete_and_upload_all(self):
        self.delete_all()
        for file_name in self.file_names:
            self.upload_jsonl(file_name)
    
    def get_file_id(self, filename):
        if filename in self.file_dict:
            return self.file_dict[filename]
        else:
            file_id = self.upload_jsonl(filename)
            self.file_names.append(filename)
            time.sleep(20)
            return file_id 

    # use jsonlines to read a jsonl file a return a list of strings
    # test this function
    def documents_from_jsonl(self, filename):
        file_path = ".\\docs\\" + filename
        with open(file_path) as f:
            documents = []
            for line in f:
                documents.append(json.loads(line)['text'])
        return documents