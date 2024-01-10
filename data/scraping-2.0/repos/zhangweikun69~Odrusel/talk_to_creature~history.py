







# history_list -> [ [(Q , A) , timestamp , type] , [(Q , A) , timestamp , type] , ...]

def history_selector(llm, user_input, history_list, vec_history_list, tokens_limit=1000):
    llm = llm
    history_QAlist = []
    history_Qlist = []
    history_Alist = []
    history_timestamplist = []
    tokens = 0
    tokens = tokens + llm.calculate_tokens(str(user_input))
    tokens_list = []
    count = 0
    for history_item in history_list:
        if(tokens>tokens_limit):
            break
        history_QA = history_item[0]
        history_Q = history_item[0][0]
        history_A = history_item[0][1]
        history_QAlist.append(history_item[0])
        history_Qlist.append(history_item[0][0])
        history_Alist.append(history_item[0][1])
        history_timestamplist.append(history_item[1])

        t = llm.calculate_tokens(str(history_A)) + llm.calculate_tokens(str(history_Q))
        tokens_list.append(t)
        tokens = tokens + t


        count = count+1
    
    return_list = [count, tokens, tokens_list, history_QAlist, history_Qlist, history_Alist]
    
    return return_list


from openai_func import ChatGPT
import configparser
config = configparser.ConfigParser()
config.read('config.ini', encoding='utf-8')
openai_config = config['OPENAI']
llm = ChatGPT(openai_config)

history_list = [[('My name is Weikun Zhang','Hi Weikun, How can I help you?'), 1, 2],
                [('Fish is good!','Yes fish is delicious!'), 2, 2],
                [('Hi','Hi How can I help you?'), 3, 2]  
                ] 

user_input = "What is my name?"

r = history_selector(llm, user_input, history_list, [], 30)