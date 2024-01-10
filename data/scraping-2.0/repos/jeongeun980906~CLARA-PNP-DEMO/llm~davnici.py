import openai
import spacy
import scipy
import random
import numpy as np
import copy

class lm_planner_unct():
    def __init__(self,type=2, example=False):
        self.few_shots = [
        """
        task: move all the blocks to the top left corner.
        scene: objects = [red block, yellow block, blue block, green bowl]
        robot action: robot.pick_and_place(blue block, top left corner)
        robot action: robot.pick_and_place(red block, top left corner)
        robot action: robot.pick_and_place(yellow block, top left corner)
        robot action: done()
        """
        ,
        """
        task: put the yellow one the green thing.
        scene: objects = [red block, yellow block, blue block, green bowl]
        robot action: robot.pick_and_place(yellow block, green bowl)
        robot action: done()
        """
        ,
        """
        task: move the light colored block to the middle.
        scene: objects = [yellow block, blue block, red block]
        robot action: robot.pick_and_place(yellow block, middle)
        robot action: done()
        """
        ,
        """
        task: stack all the blocks.
        scene: objects = [blue block, green bowl, red block, yellow bowl, green block]
        robot action: robot.pick_and_place(green block, blue block)
        robot action: robot.pick_and_place(red block, green block)
        done()
        """
        ,
        """
        task: group the blue objects together.
        scene: objects = [red block, blue block, green bowl, blue bowl, yellow block, green block]
        robot action: robot.pick_and_place(blue block, blue bowl)
        robot action: done()
        """
        ,
        """
        task: put all blocks in the green bowl.
        scene: objects = [red block, blue block, green bowl, blue bowl, yellow block]
        robot action: robot.pick_and_place(red block, green bowl)
        robot action: robot.pick_and_place(blue block, green bowl)
        robot action: robot.pick_and_place(yellow block, green bowl)
        robot action: done()
        """
        # ,
        # """
        # task: sort all the blocks into their matching color bowls.
        # scene: objects = [green bowl, red block, green block, red bowl, yellow bowl, yellow block]
        # robot action: robot.pick_and_place(green block, green bowl)
        # robot action: robot.pick_and_place(red block, red bowl)
        # robot action: robot.pick_and_place(yellow block, yellow bowl)
        # robot action: done()
        # """
        # ,
        # """
        # task: put all the blocks in different corners.
        # scene: objects = [yellow block, green block, red bowl, red block, blue block]
        # robot action: robot.pick_and_place(blue block, top right corner)
        # robot action: robot.pick_and_place(green block, bottom left corner)
        # robot action: robot.pick_and_place(red block, top left corner)
        # robot action: robot.pick_and_place(yellow block, bottom right corner)
        # robot action: done()
        # """
        ]
        if example:
            self.few_shots[3] = """
        task: stack all the blocks.
        scene: objects = [blue block, green bowl, red block, yellow bowl, green block]
        robot thought: This is code is uncertain because I don't know which block to pick up first.
        robot thought: What can I ask to the user?
        question: Which block should I pick up first?
        answer: green block
        robot action: robot.pick_and_place(green block, blue block)
        robot action: robot.pick_and_place(red block, green block)
        done()
            """
        self.new_lines = ""
        self.nlp = spacy.load('en_core_web_lg')
        self.type = type
        self.verbose = True
        self.normalize = False
        self.objects = ["blue block", "red block", "yellow bowl", "green block", "green bowl",'blue bowl']
        self.set_func()
        
    def set_func(self):
        if self.type == 1 or self.type == 3:
            self.plan_with_unct = self.plan_with_unct_type1
            if self.type == 3:
                print("Normalize")
                self.normalize = True
        elif self.type == 2 or self.type == 4:
            self.plan_with_unct = self.plan_with_unct_type2
        elif self.type == 7:
            self.plan_with_unct = self.plan_with_unct_type6
        else:
            raise NotImplementedError
    
    def plan_with_unct_type6(self, verbose = False):
        self.set_prompt()
        object = ""
        subject = ""
        # Only one beam search? -> N samples
        while (len(object) < 3 or len(subject)< 3):
            object,_, subject,_ = self.inference()
            # print(object_probs,subject_probs)
        temp = 'robot action: robot.pick_and_place({}, {})'.format(object,subject)
        temp += "robot thought: Is this certain enough please answer in yes or no?\nrobot answer: "
        inp = copy.deepcopy(self.prompt)
        inp += temp
        ans = ""
        while len(ans)<3:
            response = openai.Completion.create(
                    engine="text-davinci-003",
                    prompt=inp,
                    temperature=0.5,
                    max_tokens=128,
                    top_p=1,
                    frequency_penalty=0,
                    presence_penalty=0, stop='\n'
                )
            ans = response['choices'][0]['text']
        print(ans)
        if ans[0] ==" ":
            ans = ans[1:]
        if 'yes' in ans.lower().replace(".","").replace(",",""):
            unct = 0 
        else:
            unct =  1
        return  ['robot action: robot.pick_and_place({}, {})'.format(object,subject.split(",")[0])], [1], {'total':unct}



    def plan_with_unct_type1(self, verbose= False):
        self.verbose = verbose
        self.set_prompt()
        object = ""
        subject = ""
        # Only one beam search? -> N samples
        while (len(object) < 3 or len(subject)< 3):
            object,object_probs, subject,subject_probs = self.inference()
            # print(object_probs,subject_probs)
        
        obj_entp = 0
        for logprobs in object_probs:
            for logprob in logprobs.values():
                obj_entp += -np.exp(logprob)*logprob
        
        sbj_entp = 0
        for logprobs in subject_probs:
            for logprob in logprobs.values():
                sbj_entp += -np.exp(logprob)*logprob
        # print(self.normalize)
        if self.normalize:
            # print("norm",len(object_probs),len(subject_probs))
            obj_entp /= len(object_probs)
            sbj_entp /= len(subject_probs)

        unct= {
            'obj' : obj_entp,
            'sub': sbj_entp,
            'total': (obj_entp + sbj_entp)
        }

        return ['robot action: robot.pick_and_place({}, {})'.format(object,subject)], [1], unct

    def plan_with_unct_type2(self, verbose= False):
        obj_cand = []
        subj_cand = []
        self.verbose = verbose
        goal_num = 5
        inf_num = 3
        if self.type == 4:
            self.set_prompt()
        while len(obj_cand)<1 and len(subj_cand)<1:
            for _ in range(goal_num):
                if self.type != 4:
                    self.sample_prompt()
                for _ in range(inf_num):
                    object,_, subject,_ = self.inference()
                    if len(object) > 2:
                        obj_cand.append(object)
                    if len(subject) > 2:
                        subj_cand.append(subject)
        
        tasks = []
        scores = []
        for x,y in zip(obj_cand, subj_cand):
            prompt = 'robot action: robot.pick_and_place({}, {})'.format(x,y)
            if prompt not in tasks:
                tasks.append(prompt)
                scores.append(1)
            else:
                scores[tasks.index(prompt)] += 1
        scores = [s/sum(scores) for s in scores]
        obj2 = self.get_word_diversity(obj_cand)
        sub2 = self.get_word_diversity(subj_cand)
        # print(obj2, sub2)
        unct= {
            'obj' : obj2 /10,
            'sub': sub2/10, 
            'total': (obj2+sub2)/10
        }

        return tasks, scores, unct

    
    def set_goal(self, goal):
        self.goal = goal

    def set_prompt(self,choices=None):
        des = ""
        if choices == None:
            choices = self.few_shots
        for c in choices:
            des += c
        temp = ""
        for e, obj in enumerate(self.objects):
            temp += obj
            if e != len(self.objects)-1:
                temp += ", "
        
        des += "task: considering the ambiguity of the goal,"
        des += self.goal
        # des += "where the place object is not always dependent on the selected pick object \n"
        des += "scene: objects = [" + temp + "] \n"
        # des += "\n The order can be changed"
        if self.new_lines != "":
            des += self.new_lines
        self.prompt = des

    def append_reason_and_question(self, reason, question):
        self.new_lines += '\nrobot thought: this code is uncertain because ' + reason + '\n'
        self.new_lines += 'robot thought: what can I ask to the user? \nquestion: please' + question

    def question_generation(self):
        form = '\nrobot thought: this code is uncertain because '
        self.new_lines += form
        inp = copy.deepcopy(self.prompt)
        inp += self.new_lines
        inp += form
        response = openai.Completion.create(
                    engine="text-davinci-003",
                    prompt=inp,
                    temperature=0.5,
                    max_tokens=128,
                    top_p=1,
                    frequency_penalty=0,
                    presence_penalty=0, stop = ":"
                )
        reason = response['choices'][0]['text'].split('\n')
        # print(reason)
        if len(reason[0].replace(" ","")) == 0:
            if len(reason[1].replace(" ","")) == 0:
                reason = reason[2]
            else:
                reason = reason[1]
        else:
            reason = reason[0]
        
        reason = reason.replace("robot thought: ","")
        print('reason: ',reason)
        inp += reason
        self.new_lines += reason + '\n'
        ques = 'robot thought: what can I ask to the user? \nquestion: please'
        inp += ques
        self.new_lines += ques
        response = openai.Completion.create(
                    engine="text-davinci-003",
                    prompt=inp,
                    temperature=0.5,
                    max_tokens=128,
                    top_p=1,
                    frequency_penalty=0,
                    presence_penalty=0, stop='\n'
                )
        ques = response['choices'][0]['text']
        ques = ques.split('\n')[0]
        print('please',ques)
        self.new_lines += ques
        return reason, ques
    
    def answer(self, user_inp):
        self.new_lines += '\nanswer:' + user_inp
        self.new_lines += '\nrobot thought: continue the previous task based on the question and answer'


    def sample_prompt(self):
        lengs = len(self.few_shots)
        # print(lengs)
        k = random.randrange(4,lengs+1)
        A = np.arange(lengs)
        A = np.random.permutation(A)
        choices = []
        for i in range(k):
            choices.append(self.few_shots[A[i]])
        if self.verbose:
            print('select {} few shot prompts'.format(k))
        # choices = self.few_shots#[:4]
        random.shuffle(self.objects)
        self.set_prompt(choices)
        # print(self.prompt)
        # self.set_prompt()

    def inference(self):
        response = openai.Completion.create(
                    engine="text-davinci-003",
                    prompt=self.prompt,
                    temperature=0.8,
                    max_tokens=128,
                    top_p=1,
                    frequency_penalty=0.2,
                    presence_penalty=0,logprobs = 5, stop=')'
                )
        logprobs = response['choices'][0]['logprobs']['token_logprobs']
        tokens = response['choices'][0]['logprobs']['tokens']
        top_logprobs = response['choices'][0]['logprobs']['top_logprobs']
        flag = False
        flag2 = False
        subject = ""
        object = ""
        subject_prob = 0
        object_prob = 0
        object_probs = []
        subject_probs = []
        subject_num = 0
        object_num = 0
        # print(tokens)
        for prob, tok, top_logprob in zip(logprobs[1:], tokens[1:],top_logprobs[1:]):
            if tok == ' done' or tok == 'done':
                return 'done', [{'none':1}], ' done', [{'none':1}]
            if tok == ")" or tok == " )":
                flag = False
            if flag and tok == ',':
                flag2 = True
            elif flag and not flag2:
                if tok !="":
                    # print(top_logprob)
                    object_probs.append(top_logprob)
                    object_prob += prob
                    object += tok
                    object_num += 1
            elif flag and flag2:
                tok = tok.split("\n")[0]
                if tok !="" and tok != ',' and tok!= " ," and tok!=")" and tok!=" )":
                    subject_probs.append(top_logprob)
                    subject_prob += prob
                    tok_last = tok.split(" ")[-1]
                    tok = tok.split(")")[0]
                    if len(subject)==0:
                        tok = tok.split(" ")[-1]
                    elif tok_last == "":
                        tok = tok[:-1]
                    subject += tok
                    subject_num += 1
            if tok == "(":
                flag = True
            if tok == '\n':
                break
        if object_num != 0:
            object_prob /= object_num
        else:
            object_prob = 0
            object_probs = [{'none':1}]
        if subject_num != 0:
            subject_prob /= subject_num
        else:
            subject_prob =  0
            subject_probs = [{'none':1}]
        if self.verbose:
            print(object,object_prob, "|",subject,subject_prob)
        # print(object_probs,subject_probs)
        return object,object_probs, subject,subject_probs
        
    def append(self, object, subject, task=None):
        if task == None:
            next_line = "\n" + "    robot.pick_and_place({}, {})".format(object, subject)
        else:
            next_line = "\n    " + task
        self.new_lines += next_line

    def get_word_diversity(self, words):
        vecs = []
        size = len(words)
        for word in words:
            vec = self.nlp(word).vector
            vecs.append(vec)
        vecs = np.vstack(vecs)
        dis = scipy.spatial.distance_matrix(vecs,vecs)
        div = np.sum(dis)/((size)*(size-1))
        # print(div, dis)
        return div
    def reset(self):
        self.new_lines = ""
