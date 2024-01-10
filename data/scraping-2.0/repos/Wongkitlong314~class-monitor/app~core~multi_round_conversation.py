import openai
import json
import time
openai.api_key = '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'

def get_completion(messages, model="gpt-3.5-turbo", temperature=0):
    response = ''
    except_waiting_time = 0.1
    while response == '':
        try:
            response = openai.ChatCompletion.create(
                model=model,
                messages=messages,
                temperature=temperature,
                request_timeout=50
            )
            # k_tokens = response["usage"]["total_tokens"]/1000
            # p_tokens = response["usage"]["prompt_tokens"]/1000
            # r_tokens = response["usage"]["completion_tokens"]/1000
            # print("Tokens used: {:.2f}k".format(k_tokens))
            # print("Prompt tokens: {:.2f}k".format(p_tokens))
            # print("Response tokens: {:.2f}k".format(r_tokens))

        except Exception as e:
            #print(e)
            #print("Sleep for {:.2f}s".format(except_waiting_time))
            time.sleep(except_waiting_time)
            if except_waiting_time < 2:
                except_waiting_time *= 2
    return response.choices[0].message["content"]



def main():
    messages = [{"role": "user","content":""}]
    while 1:
        try:
            text = input('Your message ("quit" to end):')
            if text == 'quit':
                break
            # 问
            d = {"role":"user","content":text}
            messages.append(d)
            text = get_completion(messages)
            d = {"role":"assistant","content":text}
            # 答
            print('Answer:'+text+'\n')
            messages.append(d)
        except:
            messages.pop()
            print('ChatGPT:error\n')

main()