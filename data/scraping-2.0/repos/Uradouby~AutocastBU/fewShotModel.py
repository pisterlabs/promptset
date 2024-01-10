import openai
import numpy as np
import sklearn
from sklearn.metrics import pairwise
from sklearn import preprocessing
from difflib import SequenceMatcher
from dataprocess import get_questions, split_questions_by_type
import Levenshtein

openai.api_key="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"
codex_engine = "code-davinci-002"
few_shot_max_tokens = 256
engine_temperature = 0
engine_topP = 0

embedding_model = "text-embedding-ada-002"
embedding_encoding = "cl100k_base"  # this the encoding for text-embedding-ada-002
max_tokens = 8000  # the maximum for text-embedding-ada-002 is 8191

similarity_cnt=3  #the number of similar questions


def get_embedding(text, model="text-embedding-ada-002"):
    text = text.replace("\n", " ")
    return openai.Embedding.create(input=[text], model=model)['data'][0]['embedding']

def lev_dis(a, b):
    return Levenshtein.distance(a,b)


class fewShotModel():

    def __init__(self, questions):
        self.questions = questions
        '''
        self.Q_embeddings = []
        for q in self.questions:
            self.Q_embeddings.append(get_embedding(q['question'], model=embedding_model))
        '''
        self.similar = []
        self.input = []
        self.output = []

    def predict(self, q):
        input = ''
        distance = []

        '''
        embedding_question = get_embedding(q)
        for embedding in self.Q_embeddings:
            distance.append(sklearn.metrics.pairwise.cosine_distances(X=embedding_question, Y=embedding))
        '''
        for qs in self.questions:
            distance.append(lev_dis(qs["question"], q["question"]))
        print(distance)
        arg_distance = np.argsort(distance)
        print(arg_distance)
        for i in range(similarity_cnt):
            index = arg_distance[i]
            input = input+str(self.questions[index]['question'])+str(self.questions[index]['answer'])+'\n\n'
        input=input+str(q['question'])
        print(input)
        few_shot_output = openai.Completion.create(engine=codex_engine,
                                                   prompt=input,
                                                   max_tokens=few_shot_max_tokens,
                                                   temperature=engine_temperature,
                                                   top_p=engine_topP)['choices'][0]['text']
        #embedding_ans = get_embedding(few_shot_output)
        res = []
        for i in range(len(q['choices'])):
            res.append(lev_dis(q['choices'][i], few_shot_output))
            '''
            embedding_choice=get_embedding(q['choices'][i])
            res.append(1-sklearn.metrics.pairwise.cosine_distances(X=embedding_choice, Y=embedding_ans))
            '''
        res=np.array(res)
        normalized_res = preprocessing.normalize([res])
        return normalized_res[0]

    def eval(self):
        loss = 0
        accuracy = 0
        for q in self.questions:
            predict = self.predict(q)
            ground_true = ord(q['answer']) - ord('A')
            ans = np.argmax(predict)
            if ans == ground_true:
                accuracy += 1
            for i in range(len(predict)):
                if i == ground_true:
                    loss += (1-predict[i])**2
                else:
                    loss += predict[i]**2

        loss = loss/len(self.questions)
        accuracy = accuracy/len(self.questions)

        return loss, accuracy


questions = get_questions()
mc_qs, num_qs, tf_qs = split_questions_by_type(questions)

fewShot_model = fewShotModel(questions=mc_qs)
loss, accuracy = fewShot_model.eval()






