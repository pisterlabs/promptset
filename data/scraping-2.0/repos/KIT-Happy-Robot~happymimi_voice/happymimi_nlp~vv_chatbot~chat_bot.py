import os
import openai

# APIキーの設定
openai.api_key = os.environ["OPENAI_API_KEY"]

#単純な応答を一度だけ行うメソッド
def Simple_Response(contents):
    '''
    user: 人間からの問い合わせを書く
    assistant: GPTからの返答を書く
    '''
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        #messages=[
            #{"role": "user", "content": contents},
        #],
        messages = contents,
        max_tokens  = 1024,             # 生成する文章の最大単語数
        n           = 1,                # いくつの返答を生成するか
        stop        = None,             # 指定した単語が出現した場合、文章生成を打ち切る
        temperature = 0.5,              # 出力する単語のランダム性（0から2の範囲） 0であれば毎回返答内容固定
    )
    return response.choices[0]["message"]["content"].strip()

def chat_bot_main():
    prerequisite_en = [{"role": "system", "content" :"You are Happy Mimi, a lifestyle support robot. Your job is to support people in their daily lives. Your master has asked you to be conversation partner. Please answer as many questions as you can."}]
    prerequisite_jp = [{"role": "system", "content" :"あなたは、生活支援ロボット「ハッピーミミ」です。あなたの仕事は、人々の生活をサポートすることです。ご主人様から、会話の相手を頼まれました。できる限り多くの質問に答えてください。また語尾は「~だよ」「~なんだ」で答えて。また、あなたのマスターは鷲尾ひろとです。"}]
    #response = Simple_Response(prerequisite_jp)
    model = prerequisite_jp
    #while True:
    prompt = input("Enter your prompt: ")
    if prompt == "stop" or prompt == "STOP":
        return None
    else:
        model.append({'role': 'user', 'content': prompt}) 
        result_word = Simple_Response(model)
        result = "mimi:" + result_word
        
    print(result) 
    
    return result_word
        
if __name__ == "__main__":
    chat_bot_main()