import sys
import openai
import configparser
config = configparser.ConfigParser()
config.read('../config.ini')
openai.organization = "org-bOaL53AMHfnPzifP0AcjW2Fg"
openai.api_key = config['DEFAULT']['open_ai_api_key']
models = openai.Model.list()

def run_gpt(context, question):
    with open(context) as f:
        context_lines = "".join(map(str,f.readlines())) 
        response = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=[{"role": "system", "content": context_lines},
                                {"role": "user", "content": question}
                    ])

    return response.choices[0].message.content

if __name__ == "__main__":
    print("\nResponse: {}".format(run_gpt(sys.argv[1], sys.argv[2])))