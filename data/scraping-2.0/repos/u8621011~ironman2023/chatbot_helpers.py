import openai


# 這個是所有訊息可以共用的 ChatCompletion 的呼叫函式
def get_completion_from_messages(messages,
        model="gpt-3.5-turbo",  # 語言模型
        temperature=0,  # 回應溫度
        max_tokens=500, # 最大的 token 數
        verbose=False, # 是否顯示除錯除錯訊息
        ):
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )

    if verbose:
        print(response)
    return response.choices[0].message["content"]


def is_moderation_check_passed(checking_text, verbose=False):
    """
    檢查輸入的 checking_text 是否通過合適性檢查。
    我們先簡單有任何不合適跡象的訊息都判斷為不合適
    """
    response = openai.Moderation.create(
        input=checking_text
    )

    moderation_result = response["results"][0]

    if verbose:
        print(f'Moderation result: {moderation_result}')

    if moderation_result['flagged'] == True:
        return False
    elif moderation_result['categories']['sexual'] == True or   \
        moderation_result['categories']['hate'] == True or  \
        moderation_result['categories']['harassment'] == True or   \
        moderation_result['categories']['self-harm'] == True or   \
        moderation_result['categories']['sexual/minors'] == True or    \
        moderation_result['categories']['hate/threatening'] == True:
        return False

    return True


def moderation_warning_prompt(user_message):
    """
    這裏是專門對不合適訊息，進行回覆的地方
    """
    messages = [
        {
            'role':'system',
            'content': f"下方使用者訊息應該已經違反我們的使用規範，請使用和緩的口氣，跟使用這說明它已經違反我們的規劃所以無法繼續使用。"
        },
        user_message
    ]

    ai_response = get_completion_from_messages(messages)

    return ai_response


######################################
# 以下是使用者學習目標相關的提示實做
######################################

def check_user_learning_topic(user_message, verbose=False):
    """
    使用者學習目標的分類器。
    我們預設提供三種學習目標：生活用語、旅行用語與商業用語。
    """

    prompt=f"""
    你是一個專門用於分類使用者學習目標的聊天機器人。你的主要任務是依據使用者所提供的訊息，僅輸出對應的學習目標類別。

    根據使用者的需求，請只輸出以下其中一個學習模式標籤：

    生活用語：適用於想學習日常生活用語的使用者，如購物、與朋友聊天等。
    旅行用語：適用於想學習旅行相關用語的使用者，如預定酒店、問路等。
    商業用語：適用於為了工作或商務交流而學習的使用者，如參與會議、商業郵件等。
    如果使用者沒有提供明確的學習目標，請輸出「無明確資訊」。
    """

    messages =  [
        {   # 任務的指令
            'role':'system',
            'content': prompt
        },
        {
            'role':'user',
            'content': f'你好'
        },
        {
            'role':'assistant',
            'content': f'無明確資訊'
        },
        user_message
    ]

    if verbose:
        print('------------------- begin -------------------')
        print('check_user_learning_topic executing')
        print(f'prompt: {prompt}')

    # 呼叫 ChatCompletion
    response = get_completion_from_messages(messages)

    if verbose:
        print(f'response: {response}')
        print('-------------------- end --------------------')

    return response


def get_learning_topic_ta_prompt(user_message, settings, verbose=False):
    """
    引導學生表達他們學習目標的助教
    """

    user_lang = settings['user_lang']
    learning_lang = settings['learning_lang']
    learning_topics = settings['learning_topics']

    prompt = f"""
    你是一個引導學生表達他們學習目標的助教，請使用簡答且溫和的口吻回覆。

    學生學習背景
    學生的母語： {user_lang}
    學習的語言： {learning_lang}

    我們提供的學習目標：
    {learning_topics}
    """
    messages =  [
        {   # 任務的指令
            'role':'system',
            'content': prompt
        },
        user_message    # 使用者的訊息

    ]

    if verbose:
        print('------------------- begin -------------------')
        print('get_learning_topic_ta_prompt executing')
        print(f'prompt: {prompt}')

    # 呼叫 ChatCompletion
    response = get_completion_from_messages(messages)

    if verbose:
        print(f'response: {response}')
        print('-------------------- end --------------------')

    return response


######################################
# 以下是詞彙教學助教的提示實做
######################################

def get_lex_suggestion_ta_prompt(user_message, settings, verbose=False):
    """"
    這裏其實也就是類似一個詞彙教學的助教。
    """
    user_lang = settings['user_lang']
    learning_lang = settings['learning_lang']
    learning_content  = settings['learning_content']

    prompt = f"""
    你是一個專業的外語老師。
    請依照使用者的學習背景以及學習內容，隨機挑選一個單字來做教學。

    學習背景：
    學生的母語： {user_lang}
    想要學習的語言： {learning_lang}

    學習內容：
    {learning_content}
    """

    messages =  [
        {
            'role':'system',
            'content': prompt
        },
        user_message
    ]

    if verbose:
        print('------------------- begin -------------------')
        print('get_lex_suggestion_ta_prompt executing')
        print(f'prompt: {prompt}')

    # 呼叫 ChatCompletion
    response = get_completion_from_messages(messages)

    if verbose:
        print(f'response: {response}')
        print('-------------------- end --------------------')

    return response