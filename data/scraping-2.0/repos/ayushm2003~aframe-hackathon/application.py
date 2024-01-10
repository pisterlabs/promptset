from flask import Flask, render_template, json, request, jsonify
import openai

app = Flask(__name__)
chat_log = ""

FILENAME = 'requests.html'

@app.route('/', methods=["GET", "POST"])
def hello():
    '''The display page for the user'''

    return render_template(FILENAME)

@app.route('/index')
def index():
    '''The display page for the user'''
    return render_template("index.html")


@app.route("/reply", methods=['POST'])
def reply():
    '''Processses the input from user and returns it back to the html page'''

    openai.api_key = "nice try"
    completion = openai.Completion()

    start_chat_log = """The following is a conversation between a human and their friend AI. The AI is quite friendly, thoughtful, optimistic, empathetic, and is good at listening.

Human: Hello
AI: Hey, how's it going?
Human: Alright. I have been feeling kinda down lately.
AI: I'm sorry to hear that. How long have you been feeling that way?
Human: About a week or so. I have no motivation to do anything and things just haven't been working out in general.
AI: Did you talk to your friends about this?
Human: Not yet. I feel like I am not close enough with any of my friends to talk about stuff like this. It would be weird.
AI: I see. Well, you are not alone. You can talk to me!
Human: Thanks I appreciate you being here for me.
AI: It's my pleasure. So let's talk about it. Why do you think you feel the way you do?
Human: I don't know. Maybe because I work all day. I never have time for myself and I haven't hung out with friends in a while.
AI: I can see why you feel this way. Your'e not appreciating life. Sure you're setting goals for yourself and working towards them, but it seems like you're not taking the time to work on your wellbeing.
Human: Can you give me some advice? What helps in these situations?
AI: Do what you enjoy most of the time. It doesn't have to be something serious. Just do something you like to do. And exercise. Spend more time with your friends too.
Human: I haven't been exercising much lately so you might be right. I've also been so caught up with work. I don't remember the last time I watched a good movie or went out with friends. I guess I just feel like I have had more important things to do.
AI: You do. But you have to find a balance. Having time for your hobbies and friends will help you deal with the stress.
Human: I have tried to balance my life but I always fail to make it permanent. How can I get better at this?
AI: Well, you can try to do these things on a daily basis. Don't wait for the weekends or special occasions to take care of your well-being. Start doing them now and see if it works.
Human: That makes sense. Now that I think of it I don't treat everyday like a special occasion, but maybe I should. After all I don't know which day might be my last. I'm gonna try that strategy!
"""

    question = ([i for i in request.form.keys()][0])
    print("human input: " + question)
    global chat_log

    '''Check if it's the first question?'''
    if not chat_log:
        chat_log = start_chat_log

    prompt = f"{chat_log}Human: {question}\nAI:"  # format the prompt

    # generate reponse from the API

    response = completion.create(
            prompt=prompt, engine="davinci", stop=['\nHuman'], temperature=0.9,
            top_p=1, frequency_penalty=0, presence_penalty=0.6, best_of=1,
            max_tokens=150)

    answer = response.choices[0].text.strip()

    #chat_log += "Human:" + question + "AI:" + response

    print(answer)

    print('received request')

    return jsonify({'answer': answer})


# if __name__ == '__main__':
#     app.run(debug=True)
