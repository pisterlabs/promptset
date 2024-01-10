import openai
import time

openai.api_key = 'api key'


def get_completion(prompt, sys_prompt, model="gpt-3.5-turbo", temperature=0.1):
    messages = [{"role": "user", "content": prompt}, {"role": "system", "content": sys_prompt}]
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
            # print(e)1
            # print("Sleep for {:.2f}s".format(except_waiting_time))
            time.sleep(except_waiting_time)
            if except_waiting_time < 2:
                except_waiting_time *= 2
    return response.choices[0].message["content"]


# system_prompt = """
# Act as yourself.
# """
# user_prompt = """
# Hi, it's a nice day!
# """
# response = get_completion(prompt=user_prompt, sys_prompt=system_prompt, model="gpt-3.5-turbo", temperature=0.2)
# print(response)
