import os

import openai
import requests

# Generate stories using the OpenAI API then convert to speech using the RapidAPI API

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
X_RapidAPI_Key = os.environ.get("X_RapidAPI_Key")
X_RapidAPI_Host = "large-text-to-speech.p.rapidapi.com"


openai.api_key = OPENAI_API_KEY

print(f"OPENAI_API_KEY: {OPENAI_API_KEY} X_RapidAPI_Key: {X_RapidAPI_Key}")

def get_story(
  user,
  agenda,
  temperature=0,
):
  input_prompt = (
    "Below is an instruction that describes a task, paired with an input that provides further context."
    "Write a response that appropriately completes the request.\n\n"
    "### Instruction:\n"
    f"Teach a {user['age']} year old a new topic with a story about {user['interest']}\n\n"
    "### Input: \n"
    f"{agenda['course']}-{agenda['topic']}\n\n"
    "### Response: \n"
    "[your_response]")

  messages = [{"role": "user", "content": input_prompt}]
  response = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=messages,
    temperature=temperature,
  )
  return response.choices[0].message["content"]


def request_speach(text):
  url = "https://large-text-to-speech.p.rapidapi.com/tts"

  payload = {"text": text}
  headers = {
    "content-type": "application/json",
    "X-RapidAPI-Key": X_RapidAPI_Key,
    "X-RapidAPI-Host": X_RapidAPI_Host
  }

  response = requests.post(url, json=payload, headers=headers).json()
  job_id = response['id']
  job_status = response['status']

  return job_id, job_status


def get_speach(job_id):
  url = "https://large-text-to-speech.p.rapidapi.com/tts"

  querystring = {"id": job_id}

  headers = {
    "X-RapidAPI-Key": X_RapidAPI_Key,
    "X-RapidAPI-Host": X_RapidAPI_Host
  }

  response = requests.get(url, headers=headers, params=querystring).json()
  job_status = response['status']
  if job_status != 'success':
    url = ""
  else:
    url = response['url']
  return url, job_status


stories = {
  "1": {
    "story":
    "Once upon a time, there was a baseball player named Joe. Joe was trying to improve his batting average, so he decided to use quadratic functions to analyze his hits. Quadratic functions are a type of math equation that can help predict the path of a ball. Joe used this knowledge to adjust his swing and hit the ball at the perfect angle to get a home run. So, you see, math can even help you become a better baseball player!",
    "url":
    "https://s3.eu-central-1.amazonaws.com/tts-download/3c01a3e1ec3c40b72177e569331c4029.wav?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAZ3CYNLHHVKA7D7Z4%2F20230617%2Feu-central-1%2Fs3%2Faws4_request&X-Amz-Date=20230617T220149Z&X-Amz-Expires=86400&X-Amz-SignedHeaders=host&X-Amz-Signature=e9203ff3652df5bdb180f1ae977554a156a1bb4a54e949ea6b1cf6751d7fd2c0"
  },
  "2": {
    "story":
    "Once upon a time, there was a baseball player named Joe. Joe was trying to improve his batting average, so he decided to use quadratic functions to analyze his hits. Quadratic functions are a type of math equation that can help predict the path of a ball. Joe used this knowledge to adjust his swing and hit the ball at the perfect angle to get a home run. So, quadratic functions can be used in baseball to help players improve their performance. Do you want to learn more about quadratic functions and how they can be used in other areas?",
    "url":
    "https://s3.eu-central-1.amazonaws.com/tts-download/4f7162784a3e9e3d7de1948ed6245c5a.wav?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAZ3CYNLHHVKA7D7Z4%2F20230617%2Feu-central-1%2Fs3%2Faws4_request&X-Amz-Date=20230617T220438Z&X-Amz-Expires=86400&X-Amz-SignedHeaders=host&X-Amz-Signature=c34a2218cf7fe183654be9c28b12f3902d4a8fdf3730b68aa49775b8c5738def"
  },
  "3": {
    "story":
    "Once upon a time, there was a baseball player named Joe. Joe was trying to improve his batting average, so he decided to use quadratic functions to analyze his hits. Quadratic functions are a type of math equation that can help predict the path of a ball. Joe used this knowledge to adjust his swing and hit the ball at the perfect angle to get a home run. So, quadratic functions can be used in baseball to help players improve their performance.",
    "url":
    "https://s3.eu-central-1.amazonaws.com/tts-download/2e2623f1636b746c62f870b69161e131.wav?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAZ3CYNLHHVKA7D7Z4%2F20230617%2Feu-central-1%2Fs3%2Faws4_request&X-Amz-Date=20230617T220650Z&X-Amz-Expires=86400&X-Amz-SignedHeaders=host&X-Amz-Signature=fd4dac28b719f84b1f2512db751b368c0f5cbad72b049cd6f17654e07708b493"
  },
  "4": {
    "story":
    "Once upon a time, there was a baseball player named Sammy. Sammy was known for hitting home runs and stealing bases, but he struggled with understanding quadratic functions in math class. One day, his coach explained to him that quadratic functions are like the arc of a baseball when it's hit. The ball starts at a certain height, then reaches its highest point before coming back down to the ground. The path of the ball can be represented by a quadratic function. Sammy was able to understand this concept better and even started to apply it to his swing, adjusting the angle of his bat to hit the ball at the perfect trajectory. So, you see, even baseball players need to know math!",
    "url":
    "https://s3.eu-central-1.amazonaws.com/tts-download/495663d16ddaea68d3e9b42ecafa3e24.wav?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAZ3CYNLHHVKA7D7Z4%2F20230617%2Feu-central-1%2Fs3%2Faws4_request&X-Amz-Date=20230617T220955Z&X-Amz-Expires=86400&X-Amz-SignedHeaders=host&X-Amz-Signature=9ddc020304507f6a37eb41d908ff49b9ebaee48a1b6de17c00b7b1a8986e2c6c"
  },
  "5": {
    "story":
    "Once upon a time, there was a baseball player named Sammy. Sammy was known for hitting home runs and stealing bases, but he struggled with understanding quadratic functions in math class. One day, his coach explained to him that quadratic functions are like the arc of a baseball when it's hit. The ball starts at a certain height, then reaches its highest point before coming back down to the ground. The path of the ball can be represented by a quadratic function. Sammy was able to understand this concept better and even started to apply it to his swing, adjusting the angle of his bat to hit the ball at the perfect trajectory. So, you see, even baseball players need to know math!",
    "url":
    "https://s3.eu-central-1.amazonaws.com/tts-download/1cc7a72355c56828ea53c835040af604.wav?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAZ3CYNLHHVKA7D7Z4%2F20230617%2Feu-central-1%2Fs3%2Faws4_request&X-Amz-Date=20230617T221050Z&X-Amz-Expires=86400&X-Amz-SignedHeaders=host&X-Amz-Signature=e83cdab15daf429e058456fceaf45c583350f7289716dcac074dbb87fe2f3674"
  },
  "6": {
    "story":
    "Once upon a time, there was a baseball team that was struggling to score runs. They had great batters, but they were having trouble hitting the ball just right to get it over the fence. One of the players, let's call him Joe, decided to use quadratic functions to help him hit the ball farther. He realized that by finding the optimal angle to hit the ball based on the distance between him and the fence, he could increase his chances of hitting a home run. So he used the quadratic formula to figure out the exact angle he needed to hit the ball in order for it to soar over the fence. Not only did Joe start hitting more home runs, but his teammates also started to catch on and use quadratic functions to improve their hitting. So, quadratic functions can not only help with math problems, but also help you hit more home runs in baseball!",
    "url":
    "https://s3.eu-central-1.amazonaws.com/tts-download/6bbe0027905390e5aa274f5e33c17e8e.wav?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAZ3CYNLHHVKA7D7Z4%2F20230617%2Feu-central-1%2Fs3%2Faws4_request&X-Amz-Date=20230617T221507Z&X-Amz-Expires=86400&X-Amz-SignedHeaders=host&X-Amz-Signature=3cebfa70b9df140c6afab34f6d7e51ec79b1814e42bdb21c2f113e5dc6cd7a10"
  },
  "7": {
    "story":
    "Once upon a time there was a baseball team that wanted to know how far their pitcher could throw a ball. They decided to use a quadratic function to figure it out.\nThe function they used was d = -16t^2 + vt + h, where d is the distance, t is the time, v is the initial velocity, and h is the initial height.\nThey measured how high the pitcher released the ball and the velocity at which he threw it. Then they plugged these numbers into the function and calculated the maximum distance the ball would travel.\nThis helped the team figure out how far they needed to hit the ball to score a home run and how far the outfielders needed to run to catch it.Now, let's use this same function to calculate how far an object we throw in the air will travel!",
    "url":
    "https://s3.eu-central-1.amazonaws.com/tts-download/f14ce236de620bf2b10f007a8f848313.wav?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAZ3CYNLHHVKA7D7Z4%2F20230617%2Feu-central-1%2Fs3%2Faws4_request&X-Amz-Date=20230617T221640Z&X-Amz-Expires=86400&X-Amz-SignedHeaders=host&X-Amz-Signature=e02b145c2c35584daa4864f37eaa1e980d73cb061d16e800b5c6fec04802c8bf"
  },
  "8": {
    "story":
    "Let me teach you about quadratic functions through a story about baseball! Imagine you're at a baseball game and the hitter hits a ball straight up into the air. The height of the ball can be represented by a quadratic function. As the ball goes up and comes back down, its height can be modeled by the equation h(t) = -16t^2 + 60t + 6, where t represents the time in seconds and h represents the height in feet.\nIn this equation, the -16t^2 is the quadratic term. It tells us how fast the ball is accelerating towards the ground (gravity). The 60t term is the linear term and represents the initial velocity of the ball. Finally, the constant term, 6 in this case, is the initial height of the ball when it was hit.\nSo, when we graph this equation, the curve shows us how high the ball got and when it hit the ground. We can use the quadratic formula to solve for when the ball will reach its highest point and when it will hit the ground.\nCool, right? You can use quadratic functions to model all sorts of things, not just baseballs!",
    "url":
    "https://s3.eu-central-1.amazonaws.com/tts-download/f8205ee12426ae52ca062487d03a9fe3.wav?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAZ3CYNLHHVKA7D7Z4%2F20230617%2Feu-central-1%2Fs3%2Faws4_request&X-Amz-Date=20230617T221928Z&X-Amz-Expires=86400&X-Amz-SignedHeaders=host&X-Amz-Signature=870521cf9783e8e045ba87b9780fc526cabaaa8551329c099fded1426cf9c70d"
  },
  "9": {
    "story":
    "Once upon a time, there was a baseball player named Jackie Robinson. He faced many challenges in his career, but he never gave up. One day, he was facing a pitcher who threw a ball in a parabolic arc, following a quadratic function. Robinson knew that he needed to figure out the equation of the function in order to hit the ball. He used his knowledge of quadratic functions to make predictions about where the ball would go and was able to hit a home run. Quadratic functions can help you make predictions about real-life scenarios, just like Jackie Robinson used them to win the game. Let's learn more about how they work!",
    "url":
    "https://s3.eu-central-1.amazonaws.com/tts-download/be4d003e2de9008edfd5eb7fc0b93036.wav?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAZ3CYNLHHVKA7D7Z4%2F20230617%2Feu-central-1%2Fs3%2Faws4_request&X-Amz-Date=20230617T222040Z&X-Amz-Expires=86400&X-Amz-SignedHeaders=host&X-Amz-Signature=5b1bf8b47268ef1ab0f3506b3d4e2f02d32afd64e2d62494124af46070a17f16"
  },
  "10": {
    "story":
    "Once upon a time, there was a baseball pitcher named Joe. He noticed that the trajectory of his pitches followed a certain pattern, similar to the shape of a parabola. To better understand this, Joe studied quadratic functions in his math class. He learned that these functions are used to analyze and predict the path of objects that follow a curved path, like a ball in flight. Joe was able to apply this knowledge to his pitching strategies, adjusting the angle and speed of his throws to improve his game. So you see, math can even help in baseball! Let me teach you about quadratic functions using some more fun examples.",
    "url":
    "https://s3.eu-central-1.amazonaws.com/tts-download/2576cde697e7c0796fc31f790ecd5d8e.wav?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAZ3CYNLHHVKA7D7Z4%2F20230617%2Feu-central-1%2Fs3%2Faws4_request&X-Amz-Date=20230617T222141Z&X-Amz-Expires=86400&X-Amz-SignedHeaders=host&X-Amz-Signature=e4ab82c6b35cfcac00ce6141f7c31742aba391d25e57233706ce83c862c99501"
  },
}

# import time
# story = get_story(user, agenda)
# job_id, status = request_speach(story)
# while status != 'success':
#   time.sleep(5)
#   url, status = get_speach(job_id)
