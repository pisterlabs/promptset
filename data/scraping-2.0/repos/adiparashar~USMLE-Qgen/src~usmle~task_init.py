from typing import List
import pandas as pd
from InstructorEmbedding import INSTRUCTOR
from src.utils import Prompt
import json
from sklearn.metrics.pairwise import cosine_similarity
from src.prompt_lib.prompt_lib.backends import openai_api
import re
import numpy as np
import ast
from pathlib import Path

class UsmleQgenTaskInit(Prompt):
    def __init__(self, prompt_examples: str, engine: str) -> None:
        super().__init__(
            question_prefix="",
            answer_prefix="Context: ",
            intra_example_sep="\n\n",
            inter_example_sep="\n\n###\n\n",
        )
        self.engine = engine
        self.setup_prompt_from_examples_file(prompt_examples)

    def setup_prompt_from_examples_file(self, examples_path: str) -> str:
        TEMPLATE = """Clinical Note: {clinical_note}

Topic: {topic}

Keypoint: {keypoint}

Context: {context}

Question: {question}

Correct answer: {correct_answer}

Distractor options: {distractor_options}
"""     

        SIM_TEMPLATE = """
Context: {context}

Question: {question}

Options: {options}

Correct Answer : {correct_answer}
"""
        with open(examples_path) as f:
            dictt = json.load(f)
        examples_df = pd.DataFrame.from_dict(dictt, orient="index").T
        prompt = []
        for _, row in examples_df.iterrows():
            prompt.append(TEMPLATE.format(clinical_note=row["clinical_note"], topic=row["topic"],keypoint=row["keypoint"],context=row["context"],question=row["question"] ,correct_answer=row["correct_answer"],distractor_options=row["distractor_options"]))
        instruction = "An example of a USMLE type question generated from a given clinical note, topic and keypoint: "
        self.prompt = self.inter_example_sep.join(prompt)
        self.prompt = instruction + self.prompt + self.inter_example_sep
    
    def make_query(self, clinical_note: str,keypoint:str,topic:str) -> str:
        clinical_note_prefix="Clinical Note: "
        keypoint_prefix="Keypoint: "
        topic_prefix="Topic: "
        print(type(clinical_note))
        similar_qtns = self.retrieve_sim_questions(clinical_note, topic, 3)
        sim_instruction = "USMLE context based questions with their correct answers and options: \n"
        query = f"""{clinical_note_prefix}{clinical_note}{self.intra_example_sep}{topic_prefix}{topic}{self.intra_example_sep}{keypoint_prefix}{keypoint}{self.intra_example_sep}"""
        #query = f"{sim_instruction}{similar_qtns}{self.prompt}{query}{self.intra_example_sep}"
        query = f"{sim_instruction}{similar_qtns}{query}{self.intra_example_sep}"
        return query
    def retrieve_sim_questions(self,clinical_note:str, topic:str, k:int):
        SIM_TEMPLATE = """
Context and Question: {question}

Correct answer: {correct_answer}

Distractor options: {distractor_options}


"""
        model = INSTRUCTOR('hkunlp/instructor-large')
        instruction = "Represent the following clinical note for clustering, for retrieving USMLE questions related to the topic '{topic}': "
        embedding = model.encode([[instruction.format(topic=topic),clinical_note]])
        inputs_file_path = "data/US/US_qbank_embed.csv"
        qbank_df = pd.read_csv(inputs_file_path)
        qbank_df['embedding'] = qbank_df.apply(lambda x: self.get_embed_array(x),axis=1)
        qbank_df['cos_sim'] = qbank_df.apply(lambda x: cosine_similarity((x['embedding']),embedding),axis =1)
        qbank_df = qbank_df.sort_values(by=['cos_sim'], ascending=False)
        qbank_topk = qbank_df[:k].to_dict('records')
        sim_prompt = []
        for item in qbank_topk:
            item.pop('embedding')
            item.pop('cos_sim')
            sim_prompt.append(SIM_TEMPLATE.format(question=item["question"] ,correct_answer=item["correct_answer"],distractor_options=item["distractor_options"])) 
        sim_k = {'clinical_note' : clinical_note,'topic':topic}
        sim_k['similar_questions'] = qbank_topk
        with open("output.txt", "a+") as text_file:
            text_file.write(str(sim_k))
            text_file.close()
        sim_example_prompt = self.inter_example_sep.join(sim_prompt) + self.inter_example_sep
        print(sim_example_prompt)
        return sim_example_prompt
    def get_embed_array(self,row):
        x = row['embedding']
        x = re.sub("\[\s+", "[", x.strip())
        x = re.sub("\s+", ",", x.strip())
        x = ast.literal_eval(x)
        return x
    def __call__(self, clinical_note: str,keypoint:str,topic:str) -> str:
        generation_query = self.make_query(clinical_note,keypoint,topic) + "\nGenerate the context, question, correct answer and distractor options(each separately) from the clinical note, keypoint and the topic.\n"
        #print(generation_query)
        output = openai_api.OpenaiAPIWrapper.call(
            prompt=generation_query,
            engine=self.engine,
            max_tokens=2000,
            stop_token="###",
            temperature=0.7,
        )

        generated_content = openai_api.OpenaiAPIWrapper.get_first_response(output)
        #print(f"gen content: {generated_content}")
        generated_content = 'Context: ' + generated_content.split(self.answer_prefix)[1].replace("#", "").strip()
        context = re.search(r'Context:(.*?)\n\n', generated_content,flags=re.IGNORECASE).group(1)
        question = re.search(r'Question:(.*?)\n\n', generated_content,flags=re.IGNORECASE).group(1)
        correct_answer = re.search(r'correct answer:(.*?)\n\n', generated_content.lower(),flags=re.IGNORECASE).group(1)
        distractor_options = re.search(r'distractor options:(.*)', generated_content.replace('\n',' ').lower(),flags=re.IGNORECASE).group(1)
        return context.strip(),question.strip(),correct_answer.strip(),distractor_options.strip()


if __name__ == "__main__":
    task_init = UsmleQgenTaskInit(
        prompt_examples="data/prompt/usmle/init.jsonl",
        engine="gpt-3.5-turbo"
    )
    
    #print(task_init.prompt)
    cn = "An 84-year-old female with a past medical history of hypertension presented with weakness, dry cough, and shortness of breath for four days. The patient had received two doses of the COVID vaccine, with the second dose in March 2021. In the ER, her vital signs were BP 133/93, HR 103 bpm, RR 22 breaths/min, oxygen saturation of 96% on 40 L per minute of supplemental oxygen via high-flow nasal cannula, and afebrile. Laboratory assessment is in Table. Nasopharyngeal swab for SARS-CoV-2 RNA was positive. Chest X-ray on admission shows worsening right pleural effusion with new opacity obscuring the lower two-third of the right lung and a new pleural-based opacity in the left upper lobe Figure. CT chest with contrast shows large right pleural effusion and associated right basilar consolidation and abdominal ascites. The patient was admitted to the telemetry unit and started on methylprednisolone, piperacillin-tazobactam, remdesivir, and baricitinib. The patient clinically deteriorated on Day 2 and was transferred to the intensive care unit for thoracentesis and possible intubation. Thoracentesis removed 1.95 L of bloody, serosanguineous fluid obtained, with partial resolution of the effusion Figure. On Day 3, the patient developed septic shock, florid renal failure, and lethargy and was started on intravenous fluids and norepinephrine drip. Chest X-ray showed near-complete opacification of bilateral lung fields and subsequently was intubated. On Day 4, tense ascites were noted and the patient underwent paracentesis, which removed 4.25 L of bloody, serosanguinous fluid. Renal replacement therapy started. The patient was deemed to have a guarded prognosis with multiorgan failure."
    topic = "underlying processes/pathways"
    keypoint = "pathophysiology of sepsis"

    task_init.__call__(clinical_note=cn,keypoint=keypoint,topic=topic)
    # print(task_init.make_query(["a", "b", "c"]))