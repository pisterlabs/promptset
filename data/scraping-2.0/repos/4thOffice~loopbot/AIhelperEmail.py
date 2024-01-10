import sys
sys.path.append('./Auxiliary')
sys.path.append('./APIcalls')
sys.path.append('./FeedbackHandlers')
import os
import time
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
import Auxiliary.loader as loader
import APIcalls.directchatHistory as directchatHistory
import APIcalls.emailHistory as emailHistory
from langchain.evaluation import load_evaluator, EmbeddingDistance
import Auxiliary.promptCreator as promptCreator
import FeedbackHandlers.userFeedbackHandlerEmail as userFeedbackHandlerEmail
import json

class AIhelperEmail:
    feedbackHandler = None

    def __init__(self, openAI_APIKEY, userDataHandler):
        global loader
        self.openAI_APIKEY = openAI_APIKEY

        os.environ['OPENAI_API_KEY'] = openAI_APIKEY

        with open('whitelist.json', 'r') as file:
            self.whitelist = json.load(file)

        self.user_data = {}

        """#LOOPBOT CONVERSATIONS EMBEDDING
        self.fs = LocalFileStore("./cache/")
        json_path='./jsons/split.json'
        
        cached_embedder = CacheBackedEmbeddings.from_bytes_store(
            underlying_embeddings, self.fs, namespace=underlying_embeddings.model
        )

        loader_ = loader.JSONLoader(file_path=json_path)
        documents = loader_.load()
        self.db_loopbot_data = FAISS.from_documents(documents, cached_embedder)

        print("Finished embbeding loopbot data")"""

        self.userDataHandler_ = userDataHandler

        self.feedbackHandler = userFeedbackHandlerEmail.UserFeedbackHandlerEmail(feedbackBuffer=2)

    def handleGoodResponse(self, sender_userID, sender_name, contactID, cardID, AIresponse):
        authKey = self.whitelist[sender_userID]
        self.userDataHandler_.checkUserData(sender_userID)
        self.feedbackHandler.handleGoodResponse(sender_userID, sender_name, contactID, cardID, AIresponse, self.userDataHandler_.user_data[sender_userID]["good_responses_email"], self.userDataHandler_.user_data[sender_userID]["bad_responses_email"], authKey)
        return (AIresponse + " -> handeled as positive")
    
    def handleBadResponse(self, sender_userID, sender_name, contactID, cardID, AIresponse):
        authKey = self.whitelist[sender_userID]
        self.userDataHandler_.checkUserData(sender_userID)
        self.feedbackHandler.handleBadResponse(sender_userID, sender_name, contactID, cardID, AIresponse, self.userDataHandler_.user_data[sender_userID]["good_responses_email"], self.userDataHandler_.user_data[sender_userID]["bad_responses_email"], authKey)
        return (AIresponse + "  -> handeled as negative")
    
    #print relavant information about a query
    def printRelavantChats(relavant_chats):
        for i, comment in enumerate(relavant_chats):
            print("Conversation context:", i, "score:", comment[1])

            context = comment[0].metadata["context"].split("    ")
            for txt in context:
                print(txt)

    #find relavant information abotu a query
    def findRelavantChats(self, input):
        start_time1 = time.time()
        relavant_chats = self.db_loopbot_data.similarity_search_with_score(input, k=3)
        start_time2 = time.time()
        print("relavant chat finding duration:", start_time2-start_time1)
        return relavant_chats

    def findResponses(self, sender_userID, recipient_userID, authkey):
        context = directchatHistory.getAllComments(5, recipient_userID, authkey)
        contextLastTopic = directchatHistory.getLastTopic(context)

        if len(contextLastTopic) > 0:
            context = contextLastTopic

        lastMsg1 = context[-1]["content"]
        
        context = directchatHistory.memoryPostProcess(context)

        self.checkUserData(sender_userID)

        goodResponses = self.userDataHandler_.user_data[sender_userID]["good_responses"]["docs"].similarity_search_with_score(context, k=3)
        badResponses = self.userDataHandler_.user_data[sender_userID]["bad_responses"]["docs"].similarity_search_with_score(context, k=3)

        responsesGood = []
        print("good:")
        for response in goodResponses:
            print("score", response[1])
            print("data", response[0])
            if response[1] <= 0.13:
                print("lastMsg1: ", lastMsg1)
                lastMsg2 = response[0].page_content.split("\n")[-1]
                lastMsg2 = lastMsg1.replace("AI:", "")
                lastMsg2 = lastMsg1.replace("user:", "")

                print("lastMsg2: ", lastMsg2)
 
                evaluator = load_evaluator("pairwise_embedding_distance", distance_metric=EmbeddingDistance.EUCLIDEAN)
                distance = evaluator.evaluate_string_pairs(
                    prediction=lastMsg1, prediction_b=lastMsg2
                )

                print("distance: ", distance["score"])

                if distance["score"] < 0.2:
                    responsesGood.append(response[0].metadata["AIresponse"])

        responsesBad = []
        print("bad:")
        for response in badResponses:
            print("score", response[1])
            print("data", response[0])
            if response[1] <= 0.07:
                responsesBad.append(response[0].metadata["AIresponse"])
            
        return responsesGood, responsesBad

    def returnAnswer(self, sender_userID, sender_name, cardID, contactID, badResponsesPrevious, explicit_question):
        start_time1 = time.time()
        authKey = self.whitelist[sender_userID]
        
        regular_user = True
        #if sender_userID == "user_1552217" or sender_userID == "user_24564769":
        if sender_userID == "user_1552217":
            regular_user = False

        impersonated_userID, impersonated_username = emailHistory.getContactUserID(contactID, sender_userID, sender_name, authKey)
        comments = emailHistory.getEmailHistory(cardID, impersonated_userID, impersonated_username, authKey)
        
        memory_anonymous = emailHistory.memoryPostProcess(comments, impersonated_username)
        memory = emailHistory.memoryPostProcess(comments)
        
        print(memory)
        if comments[-1]["sender"] == impersonated_username and len(explicit_question) == 0:
            return "Wait for user to reply.", ""
        
        #PRVA 2 RELAVANT CHATA STA OD USER INOUT IN ZADNJI JE OD USERINPUT + HISTORY
        """if not regular_user:
            relavantChatsQuery = self.findRelavantChats(user_input)
            relavantChatsHistory = self.findRelavantChats(memory)

            #Take 2 top results for query similarity search (if similarity not over threshold) and 1 for whole history similarity search
            relavantChats = []
            for comment in relavantChatsQuery:
                print(comment)
                if len(relavantChats) >= 1:
                    break
                if comment[1] < 0.4:
                    relavantChats.append(comment)
            for comment in relavantChatsHistory:
                print(comment)
                if len(relavantChats) >= 3:
                    break
                relavantChats.append(comment)

            #relavantChats_noscore = [relavantChat[0].metadata["context"] for relavantChat in relavantChats]
            relavantChats_noscore = ""
            for index, relavantChat in enumerate(relavantChats):
                relavantChats_noscore += f"\nConversation {index}:\n"
                relavantChats_noscore += relavantChat[0].metadata["context"]
                #relavantChats_noscore += relavantChat[0].page_content
        
        else:
            relavantChats_noscore = """""

        relavantChats_noscore = ""
        
        #lastEmail = comments[-1]["content"]
        chat_prompt = promptCreator.createPromptEmail([], [], [], not regular_user, impersonated_username, memory)

        end_time1 = time.time()
        start_time2 = time.time()
        chain = LLMChain(
        #llm=ChatOpenAI(temperature="1.0", model_name='gpt-3.5-turbo-16k'),
        llm=ChatOpenAI(temperature="1.0", model_name='gpt-4'),
        prompt=chat_prompt,
        verbose=True
        )
        reply = chain.run({"relavant_messages": str(relavantChats_noscore)})

        end_time2 = time.time()

        elapsed_time = end_time1 - start_time1
        print(f"Time taken to execute preprocess steps: {elapsed_time:.6f} seconds")

        elapsed_time = end_time2 - start_time2
        print(f"Time taken to execute CHATGPT API call: {elapsed_time:.6f} seconds")
        reply = reply.replace("\n", "\\n")
        return reply, memory_anonymous
    
#lb = AIhelperEmail(keys.openAI_APIKEY)
#print(lb.returnAnswer("user_24534935", "niko sneberger", "ACsz9LnMu4tbKY0eclbTFX0UfIg0T", "CCr7jNfk92eIEmEGrRRfw_cbfVw0T", [], "")[0])