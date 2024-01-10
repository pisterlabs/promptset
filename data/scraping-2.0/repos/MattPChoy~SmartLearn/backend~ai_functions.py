# from keys import open_ai_key
import openai

PROMPT = """Generate 3 questions from the transcript. Questions should be in JSON format with fields for question, difficulty, answer_type, and answer. Answer_type can be multiple choice, number, or text. Include options field only if the question is multiple choice. Ensure only one option is the answer. Include a question of each difficulty: easy, medium, and hard. JSON response format should be:

[
  {
    "question": "Main purpose of React?",
    "difficulty": "easy",
    "answer_type": "text",
    "answer": "Build user interfaces"
  },
  {
    "question": "What is JSX?",
    "difficulty": "medium",
    "answer_type": "text",
    "answer": "JavaScript and HTML markup syntax"
  },
  {
    "question": "React state management library?",
    "difficulty": "hard",
    "answer_type": "multiple choice",
    "options": ["Redux", "MobX", "Flux", "Recoil", "XState"],
    "answer": "Redux"
  }
]"""

def transcribe(path):
    """
    REMEBER TO ADD YOUR SECRET KEY TO CONDA ENVIRONMENT VARIABLES

    Transcribe a video file into text.
    :param path: Path to the video file.
    :return: String representing the text of the video.
    """

    # Load the video file
    audio_file= open(path, "rb")
    transcript = openai.Audio.transcribe("whisper-1", audio_file)
    return transcript.text

def generate_questions(transcript):
    """
    Generate questions from a transcript using the OpenAI API.
    :param transcript: String representing the transcript of a video.
    :return: List of questions.
    """

    response = openai.Completion.create(
    model="text-davinci-003",
    prompt=f'{PROMPT}\n\n{transcript}\n',
    temperature=1,
    max_tokens=256,
    top_p=1,
    frequency_penalty=0,
    presence_penalty=0
    )
    print(response)
    return response.choices[0].text



if __name__ == "__main__":
    print(generate_questions("React, a JavaScript library for building user interfaces. Developed at Facebook and released in 2013, it's safe to say React has been the most influential UI library of recent memory. We use it to build components that represent logical, reusable parts of the UI. The beauty of React is that the simplicity of building a component has been brought down to its theoretical minimum. It's just a JavaScript function. It's so easy, a caveman could do it. The return value from this function is your HTML, or UI, which is written in a special syntax called JSX, allowing you to easily combine JavaScript with HTML markup. If you want to pass data into a component, you simply pass it a props argument, which you can then reference inside the function body or in the UI using braces. If the value changes, React will react to update the UI. If we want to give our component its own internal state, we can use the state hook. The hook is just a function that returns a value, as well as a function to change the value. In this case, count is our reactive state, and setCount will change the state. When used in the template, the count will always show the most recent value. Then we can bind setCount to a button click event so the user can change the state. React provides a variety of other built-in hooks to handle common use cases, but the main reason you might want to use React is not the library itself, but the massive ecosystem that surrounds it. React itself doesn't care about routing, state management, animation, or anything like that. Instead, it lets those concerns evolve naturally within the open source community. No matter what you're trying to do, there's very likely a good supporting library to help you get it done. If you need a static site, you have Gatsby. Need server-side rendering, you have Next. For animation, you have Spring. For forms, you have Formic. State management, you've got Redux, MobX, Flux, Recoil, XState, and more. You have an endless supply of choices to get things done the way you like it. As an added bonus, once you have React down, you can easily jump into React Native and start building mobile apps. And it's no surprise that knowing this little UI library is one of the most in-demand skills for front-end developers today. This has been React in 100 seconds. If you want to see more short videos like this, make sure to like and subscribe, and check out more advanced React content on FireshipIO. And if you're curious how I make these videos, make sure to check out my new personal channel and video on that topic. Thanks for watching, and I'll see you in the next one."))




