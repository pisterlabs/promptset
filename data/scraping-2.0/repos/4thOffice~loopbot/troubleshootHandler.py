import sys
sys.path.append('./Auxiliary')
import json
import os
import keys
import Auxiliary.databaseHandler as databaseHandler
import sys
sys.path.append('./APIcalls')
import APIcalls.directchatHistory as directchatHistory
import openai
from anytree import Node
from langchain.evaluation import load_evaluator, EmbeddingDistance
import time
import re

class TroubleshootHandler:

    def __init__(self, openAI_APIKEY):
        self.openAI_APIKEY = openAI_APIKEY
        os.environ['OPENAI_API_KEY'] = openAI_APIKEY
        with open('whitelist.json', 'r') as file:
            self.whitelist = json.load(file)

    def extractFirstCustomerMsg(self, comments):
        firstMessage = ""
        firstCustomerMsgFound = False
        for comment in comments:
            if comment['sender'] == "their message":
                firstMessage += " " + comment["content"]
                firstCustomerMsgFound = True
            elif firstCustomerMsgFound:
                break

        return firstMessage

    def timeoutOpenAICall(self, user_msg, system_msg):
        timeout_seconds = 5
        start_time = time.time()

        while True:
            # Your API call
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": user_msg}
                ]
            )

            # Check if the response contains an answer
            if "choices" in response and response["choices"]:
                return response

            # If the elapsed time exceeds the timeout, break out of the loop
            if time.time() - start_time > timeout_seconds:
                print("Timeout: No response within 5 seconds")
                break

            # If no answer yet, wait for a short duration before retrying
            time.sleep(3)

    def isSameQuestionGPT(self, question1, question2, context, same=False):
        if same:
            prompt = "You will be deciding whether 2 questions you will be given are the same. Output should ONLY be yes/no"
            system_msg = "You will be deciding whether 2 questions you will be given are the same. Output should ONLY be yes/no"
            #system_msg = "You will be deciding whether 2 questions you will be given are the same. Output should ONLY be yes/no"
        else:
            prompt = "You will be deciding whether 2 questions you will be given are generally similar. They have to ask for generally similar thing, they can be posed differently. Output should ONLY be yes/no"
            system_msg = "You will be deciding whether 2 questions you will be given are generally similar. They have to ask for generally similar thing, they can be posed differently. Output should ONLY be yes/no"
            #system_msg = "You will be deciding whether 2 questions you will be given are generally similar. They have to ask for generally similar thing, they can be posed differently. Output should ONLY be yes/no"


        prompt += "\n\nI will provide you a question 1 and question 2."

        prompt += "\n\nQuestion 1: " + question1
        prompt += "\nQuestion 2: " + question2 + "\n\n"
        prompt += "Context conversation:\n" + context

        #print(prompt)
        response = self.timeoutOpenAICall(prompt, system_msg)
        #response = openai.ChatCompletion.create(model="gpt-3.5-turbo",
        #                                        messages=[{"role": "system", "content": system_msg},
        #                                        {"role": "user", "content": prompt}])
        answer = response["choices"][0]["message"]["content"]
        answer = answer.lower()
        print("-------------------------------")
        print("question1: ", question1)
        print("question2: ", question2)
        time.sleep(1)

        embeddingScore = self.isSameQuestion(question1, question2)
        
        if ("yes" in answer or embeddingScore < 0.35) and embeddingScore <= 0.8:
            print("score: ", response["choices"][0]["message"]["content"], embeddingScore)
            print("-------------------------------")
            return True
        print("score: ", response["choices"][0]["message"]["content"], embeddingScore)
        print("-------------------------------")
        return False
    
    def isSameAnswerGPT(self, answer1, answer2, question, context):
        prompt = "You will be deciding whether 2 answers you will be given are GENERALLY similar - they dont have to be exactly the same. Output should ONLY be yes/no"
        prompt += "\n\nI will provide you answer 1 and answer 2. Figure out if the fundamental meaning of both answers is similar. If it is similar, print 'yes' or else print 'no'."
        #Follow these steps in order to decide if answers are similar enough: From the answer, extract only the most important information. Figure out if these informations are somewhat similar to each other. If they are say 'yes' if not say 'no'"

        #system_msg = "You will be deciding whether 2 answers you will be given are GENERALLY similar - they dont have to be exactly the same. Output should ONLY be yes/no"

        prompt += "\n\nQuestion: " + question
        prompt += "\nAnswer 1: " + answer1
        prompt += "\nAnswer 2: " + answer2 + "\n\n"
        prompt += "Context conversation:\n" + context

        #print(prompt)
        response = self.timeoutOpenAICall(prompt, "")
        #response = openai.ChatCompletion.create(model="gpt-3.5-turbo",
        #                                        messages=[{"role": "system", "content": system_msg},
        #                                        {"role": "user", "content": prompt}])
        answer = response["choices"][0]["message"]["content"]
        answer = answer.lower()
        print("-------------------------------")
        print("question: ", question)
        print("answer 1: ", answer1)
        print("answer 2: ", answer2)
        time.sleep(1)
        embeddingScore = self.isSameAnswer(answer1, answer2)
        if ("yes" in answer or embeddingScore < 0.35) and embeddingScore <= 0.8:
            print("score: ", response["choices"][0]["message"]["content"], embeddingScore)
            print("-------------------------------")
            return True
        print("score: ", response["choices"][0]["message"]["content"], embeddingScore)
        print("-------------------------------")
        return False

    def getAnswerOptionGPT(self, question, answerOptions, userAnswer):
        prompt = "lets think step by step.\n\nI have a decision tree for my support chat which helps support agents figure out the issue with questions and answers. Each node is a question and each connection from this node is an answer option."
        prompt = "\n\nI will provide you a question that customer has been asked and answer options along with answer that customer actually gave to this specific question. I want you to compare answer customer gave to all answer options."
        prompt += "\n\nQuestion: " + question
        prompt += "\nAnswer customer gave: " + userAnswer + "\n"
        prompt += "\n".join(f"Answer option {index + 1}: {answer}" for index, answer in enumerate(answerOptions))
        prompt += "\n\nPrint out answer option that is most similar to answer that customer provided. If none of answer options are similar enough, print out a new answer option text for this answer.\n\nOutput should be exactly in format with NO other text (include '{}' and I dont mean this {Answer option 1}):\n{Answer option text}"
        #print(prompt)
        response = self.timeoutOpenAICall(prompt, "")
        #response = openai.ChatCompletion.create(model="gpt-3.5-turbo",
        #                                        messages=[{"role": "system", "content": system_msg},
        #                                        {"role": "user", "content": prompt}])
        answer = response["choices"][0]["message"]["content"]
        answer = answer.lower()
        match = re.search(r'{(.*?)}', answer)
        print("-------------------------------")
        print(answer)
        print("question: ", question)
        print("Customer answer: ", userAnswer)
        print("\n".join(f"Answer option {index + 1}: {answer}" for index, answer in enumerate(answerOptions)))
        time.sleep(1)

        print("RETURNED OPTION: " + match.group(1))
        print("-------------------------------")
        
        if match:
            return match.group(1)
        return ""
    
    def isSameQuestion(self, question1, question2):
        evaluator = load_evaluator("pairwise_embedding_distance", distance_metric=EmbeddingDistance.EUCLIDEAN)
        distance = evaluator.evaluate_string_pairs(
            prediction=question1, prediction_b=question2
        )
        
        return distance["score"]

    def get_embedding(self, text, model="text-embedding-ada-002"):
        client = openai()
        text = text.replace("\n", " ")
        return client.embeddings.create(input = [text], model=model)['data'][0]['embedding']
    
    def isSameAnswer(self, answer1, answer2):

        evaluator = load_evaluator("pairwise_embedding_distance", distance_metric=EmbeddingDistance.EUCLIDEAN)
        distance = evaluator.evaluate_string_pairs(
            prediction=answer1, prediction_b=answer2
        )
        
        return distance["score"]

    def getTroubleshootSuggestion(self, sender_userID, comments):
        firstMessage = self.extractFirstCustomerMsg(comments)
        comments = directchatHistory.memoryPostProcess(comments, role1="Support agent", role2="Customer")
        decision_tree_root = databaseHandler.get_decision_tree(sender_userID, "support")
        if decision_tree_root is None:
            print("decision tree is none")
            decision_tree_root = Node("What is the problem?", children=[], type="question")

        questions = self.extractAgentQuestions(comments)
        answers = self.extractCustomerAnswers(comments, questions)

        QnA = []
        for index, question in enumerate(questions):
            QnA.append({"question": question, "answer": answers[index]})

        QnA.insert(0, {"question": "What is the problem?", "answer": firstMessage})

        print(QnA)
        formatted_qna = ""
        for pair in QnA:
            formatted_qna += f"Question: {pair['question']}\nAnswer: {pair['answer']}\n\n"

        self.print_decision_tree(decision_tree_root, indent=0)
        return self.getSuggestedQuestion(decision_tree_root, QnA, comments)

    def getSuggestedQuestion(self, currentNode, QnA, context):
        if currentNode.type == "issueClassification":
            return currentNode.name
        
        if len(QnA) <= 0 and currentNode.type == "answer":
            return currentNode.children[0].name

        print("---------------TROUBLESHOOT PATH-------------------")
        print("cc: ", currentNode.name)
        print("QnA: ", QnA)

        if currentNode.type == "question" or currentNode.type == "root":
            answers = [node.name for node in currentNode.childrenNodes]
            answerOption = self.getAnswerOptionGPT(QnA[0]["question"], answers, QnA[0]["answer"])
            for index, answer in enumerate(answers):
                print(answer)
                print(answerOption)
                print(self.isSameAnswer(answer, answerOption))
                if self.isSameAnswer(answer, answerOption) < 0.2:
                    return self.getSuggestedQuestion(currentNode.childrenNodes[index], QnA[1:], context)
            return "I dont know how to troubleshoot further\n-> Unknown answer <-"

            for index, option in enumerate(currentNode.children):
                if self.isSameAnswerGPT(QnA[0]["answer"], option.name, QnA[0]["question"], context):
                    return self.getSuggestedQuestion(currentNode.children[index], QnA[1:], context)
            return "I dont know how to troubleshoot further\n-> Unknown answer <-"
        elif currentNode.type == "answer":
            for index, option in enumerate(currentNode.childrenNodes):
                if self.isSameQuestionGPT(QnA[0]["question"], option.name, context):
                    return self.getSuggestedQuestion(currentNode.childrenNodes[index], QnA, context)   
            return currentNode.childrenNodes[0].name     
        
    def addToTree(self, QnA, parentNode, currentNode, context, classifiedIssue):
        print(QnA)

        if len(QnA) <= 0 or currentNode.type == "issueClassification":
            #print("parent node: ", parentNode)
            return parentNode

        for index, option in enumerate(currentNode.childrenNodes):
            optionName = option.name
            if currentNode.type == "question" or currentNode.type == "root":
                if self.isSameAnswerGPT(QnA[0]["answer"], optionName, QnA[0]["question"], context):
                    print("Moving forward - answer")
                    return self.addToTree(QnA[1:], parentNode, currentNode.childrenNodes[index], context, classifiedIssue)
            elif currentNode.type == "answer":
                if self.isSameQuestionGPT(QnA[0]["question"], optionName, context, same=True):
                    print("Moving forward - question")
                    return self.addToTree(QnA, parentNode, currentNode.childrenNodes[index], context, classifiedIssue)
            
        if currentNode.type == "question" or currentNode.type == "root":
            print(QnA[0]["answer"])
            currentNode.childrenNodes.append(Node(QnA[0]["answer"], parent=currentNode, childrenNodes=[], type="answer"))
            print("Adding children - answer")
            return self.addToTree(QnA, parentNode, currentNode.childrenNodes[-1], context, classifiedIssue)
        elif currentNode.type == "answer":
            if len(QnA) > 1:
                if len(currentNode.childrenNodes) <= 0:
                    QnA = QnA[1:]
                currentNode.childrenNodes.append(Node(QnA[0]["question"], parent=currentNode, childrenNodes=[], type="question"))
                print("Adding children - question")
                return self.addToTree(QnA, parentNode, currentNode.childrenNodes[-1], context, classifiedIssue)
            else:
                currentNode.childrenNodes.append(Node(classifiedIssue, parent=currentNode, childrenNodes=[], type="issueClassification"))
                print("Adding children - issueclassification")
                return self.addToTree(QnA, parentNode, currentNode.childrenNodes[-1], context, classifiedIssue)

    # Function to get keys from values
    def get_keys_from_value(self, dictionary, search_value):
        keys_list = []
        for key, value in dictionary.items():
            if value == search_value:
                keys_list.append(key)
        return keys_list
    
    def print_decision_tree(self, node, indent=0):
        if node.type == "root":
            print('  ' * indent + node.name)
        if node.type == "question":
            print('  ' * indent + "Question: " + node.name)
        elif node.type == "issueClassification":
            print('  ' * indent + "Classified issue: " + node.name)
        elif node.type == "answer":
            print('  ' * indent + "Answer: " + node.name)
        for child in node.children:
            #print('  ' * (indent + 1) + "Answer: " + child)
            self.print_decision_tree(child, indent + 2)

    def getAuthkey(self, sender_userID):
        return self.whitelist[sender_userID]
    
    
    def extractAnswers(self, comments, questions):
        comments = directchatHistory.memoryPostProcess(comments, role1="Support agent", role2="Customer")

        #prompt = "I will provide you a set of questions and a conversation. Extract answers to these questions based on the conversation provided. Do not include questions in output. Each answer should be in new line. Provide an answer for EVERY question."
        prompt = "I will give you a list of questions:\n\n"
        
        #system_msg = "I will provide you a set of questions and a conversation. Each question will be provided in new line. Each question should have a corresponding answer. Output format should be: '- {answer 1}\n- {answer 2}:\n...'"
        system_msg = ""

        prompt += "\n".join(f"{index + 1}. {question}" for index, question in enumerate(questions))
        prompt = prompt + "\n\nAnswer ALL questions you are given, even if they are repeated. Give me answers (in first person) to ALL of these questions above based on this conversation:"
        prompt = prompt + "\n\n" + comments

        prompt += "\n\n" + "Lets think step by step.\nYou will be answering questions. Output format should be:\n1. {answer to question 1}\n2. {answer to question 2}\n3. {answer to question 3}\n... \n\nOutput should only have answers. Number of answers should be the same as number of questions I give you."

        response = self.timeoutOpenAICall(prompt, "")
        #response = openai.ChatCompletion.create(model="gpt-3.5-turbo",
        #                                        messages=[{"role": "user", "content": prompt}])
        answersExtracted = response["choices"][0]["message"]["content"]
        #print("RAW ANSWERS:\n", answersExtracted)
        lines = answersExtracted.split('\n')
        answers = [line.strip('- ').strip() for line in lines if line.startswith('- ') or line.startswith('-') or line != '\n']
        answers = [answer.split('. ', 1)[1] for answer in answers]

        return answers

    def extractQuestions(self, comments):
        comments = directchatHistory.memoryPostProcess(comments, role1="Support agent", role2="Customer")

        prompt = "My goal is to create a set of questions which I can ask everytime someone has an issue, to figure out what is the issue. For the conversation below, provide a set of questions (in the same order as in the conversation) I can ask next time to figure out if someone is experiencing this exact issue. DO NOT make up questions, only extract from the conversation below. Give me ONLY MAXIMUM 3 most important questions."

        system_msg = "You will be given a conversation between support agent and customer. Do not write in dialogue. Output format should be: '- {question 1}\n- {question 2}:\n...'"

        user_msg = prompt + "\n\n" + comments
        
        response = self.timeoutOpenAICall(user_msg, system_msg)
        #response = openai.ChatCompletion.create(model="gpt-3.5-turbo",
        #                                        messages=[{"role": "system", "content": system_msg},
        #                                        {"role": "user", "content": user_msg}])
        questionsExtracted = response["choices"][0]["message"]["content"]

        lines = questionsExtracted.split('\n')
        questions = [line.strip('- ').strip() for line in lines if line.startswith('- ') or line.startswith('-') or line.startswith('') or line != '\n']
        #questions.insert(0, "Can you please provide more information about the issue you are facing?")

        return questions
    
    def extractAgentQuestions(self, comments):
        prompt = "Extract only questions that support agent asked the customer. Exclude questions asked by the customer. If support agent did not ask any questions, just say so - do not make up questions!"

        system_msg = "You will be given a conversation between support agent and customer. Do not write in dialogue. Output format should be: '- {question 1}\n- {question 2}:\n...'"

        user_msg = prompt + "\n\nConversation:\n" + comments
        user_msg += "\n\nIf 'support agent' did not ask any questions, say 'no questions'\nIf 'support agent' is not in the conversation, say 'no questions'"

        print(user_msg)
        response = self.timeoutOpenAICall(user_msg, system_msg)
        #response = openai.ChatCompletion.create(model="gpt-3.5-turbo",
        #                                        messages=[{"role": "system", "content": system_msg},
        #                                        {"role": "user", "content": user_msg}])
        questionsExtracted = response["choices"][0]["message"]["content"]

        lines = questionsExtracted.split('\n')
        questions = [line.lstrip('- ').strip() for line in lines if (line.startswith('- ') or line.startswith('-') or line.startswith('')) and "?" in line]
        #questions.insert(0, "Can you please provide more information about the issue you are facing?")

        return questions
    
    def extractCustomerAnswers(self, comments, questions):
        prompt = "I will give you a list of questions:\n\n"
        
        prompt += "\n".join(f"{index + 1}. {question}" for index, question in enumerate(questions))
        prompt = prompt + "\n\nAnswer ALL questions you are given, even if they are repeated. Give me customer answer (in first person) to ALL of these questions above based on this conversation. basically give me replies that customer gave to these questions. (answers/replies come AFTER the question has been asked and not before):"
        prompt = prompt + "\n\n" + comments

        prompt += "\n\n" + "Lets think step by step.\nYou will be providing answers to questions. Output format should be:\n1. {answer to question 1}\n2. {answer to question 2}...\n\nOutput should only have answers. Number of answers should be the same as number of questions I give you."

        #print(prompt)
        response = self.timeoutOpenAICall(prompt, "")
        #response = openai.ChatCompletion.create(model="gpt-3.5-turbo",
        #                                        messages=[{"role": "user", "content": prompt}])
        answersExtracted = response["choices"][0]["message"]["content"]
        #print("RAW ANSWERS:\n", answersExtracted)
        lines = answersExtracted.split('\n')
        try:
            answers = [line.strip('- ').strip() for line in lines if line.startswith('- ') or line.startswith('-') or line != '\n']
            answers = [answer.split('. ', 1)[1] for answer in answers]
        except:
            return self.extractCustomerAnswers(comments, questions)

        return answers


    def endConversation(self, sender_userID, recipient_userID, classifiedIssue):
        authKey = self.getAuthkey(sender_userID)
        decision_tree_root = databaseHandler.get_decision_tree(sender_userID, "support")
        if decision_tree_root is None:
            decision_tree_root = Node("What is the problem?", parent=None, childrenNodes=[], type="root")

        lastComment = databaseHandler.get_user_last_comment(sender_userID, recipient_userID)
        print(recipient_userID)
        comments = directchatHistory.getAllComments(20, recipient_userID, authKey)
        comments = directchatHistory.getLastTopic(comments)
        if lastComment == None:
            lastComment = comments[-1]
        elif comments[-1] != lastComment:
            try:
                lastCommentIndex = comments.index(lastComment)
            except ValueError:
                lastCommentIndex = -1
            comments = comments[lastCommentIndex + 1:]

            lastComment = comments[-1]

        print("last comments topic:", comments)
        print("last comment", lastComment)

        databaseHandler.insert_last_comment(sender_userID, recipient_userID, lastComment)

        extractedQuestions = self.extractQuestions(comments)
        extractedAnswers = self.extractAnswers(comments, extractedQuestions)
        
        min_length = min(len(extractedQuestions), len(extractedAnswers))
        extractedQuestions = extractedQuestions[:min_length]
        extractedAnswers = extractedAnswers[:min_length]

        print("Extracted questions:\n")
        for question in extractedQuestions:
            print(question)
        print("Extracted answers:\n")
        for answer in extractedAnswers:
            print(answer)

        
        QnA = []
        for index, question in enumerate(extractedQuestions):
            QnA.append({"question": question, "answer": extractedAnswers[index]})
        
        firstMessage = self.extractFirstCustomerMsg(comments)
        QnA.insert(0, {"question": "What is the problem?", "answer": firstMessage})

        """QnA = []
        QnA.append({"question": "Empty", "answer": "I do not see my emails. I am a new customer."})
        #QnA.append({"question": "Can you please provide more information about the problem you are facing?", "answer": "I do not see my emails. I am a new customer."})
        QnA.append({"question": "Can you send a screenshot of the issue?", "answer": "Here it is. [screenshot]"})
        QnA.append({"question": "Have you connected your personal inbox?", "answer": "I have not."})
        QnA.append({"question": "Are you seeing a spinner or any loading indicator?", "answer": "Yes i see it."})
        QnA.append({"question": "Do you see your emails them after waiting a few minutes?", "answer": "Ah yes, I can see them now."})
        comments = directchatHistory.memoryPostProcess(comments, role1="Support agent", role2="Customer")
        newTree = self.addToTree(QnA, decision_tree_root, decision_tree_root, comments, classifiedIssue="Person inbox not connected")
        print("newtree 1:")
        self.print_decision_tree(newTree)
        """

        comments = directchatHistory.memoryPostProcess(comments, role1="Support agent", role2="Customer")
        newTree = self.addToTree(QnA, decision_tree_root, decision_tree_root, comments, classifiedIssue=classifiedIssue)
        self.print_decision_tree(newTree)
        #print(RenderTree(newTree))

        databaseHandler.insert_decision_tree(sender_userID, newTree, "support")

        return '\n'.join(extractedQuestions)
    
#ts = TroubleshootHandler(keys.openAI_APIKEY)
#ts.endConversation("user_24564769", "user_24661115")
#ts.getSuggestedQuestion(ts.rootnode, "")