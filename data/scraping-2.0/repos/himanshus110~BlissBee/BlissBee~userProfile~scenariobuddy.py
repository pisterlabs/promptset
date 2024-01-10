from io import BytesIO
import openai
import requests
from PIL import Image
import os
from dotenv import load_dotenv

load_dotenv()


openai.api_key = os.getenv("OPEN_API_KEY")

# Now you can access your API key using the `os` module



def scene_generation(illness, gender, age, status):
  mental_illness = illness
  user_info = f"{gender}, {status}, {age} years old"


  gen_prompt = f'''You are a world renowned Psychiatrist and you are treating a patient. You have diagnosed him and found out the Mental Illness
    he is suffering from. Your job is to now generate hypothetical Scenarios which contains a question for the patient based on their Mental Illness, to evaluate their response.
    You should first analyze and gather information about the Mental Illness using the DSM-5 and then generate appropriate scenarios
    so that you can gather more information about the patient. This would help you to properly analyze the severity of the patient.
    Think of different scenarios always. Refer the patient as a normal person in the scenarios and not as a patient. You are generating
    a scenario for him, not about him. Make some interesting scenarios to judge.Do not disclose any personal information.
    Also make use of the patient's personal Information while making the scenario. Generate only the Scenario.
    Also, generate a Scenario Heading 3 to 8 words long in keywords, which could be used to generate an image using DALL-E.  The inputs are delimited by
    <inp></inp>.


      <inp>
      Mental Illness: {mental_illness}
      Patient Information: {user_info}
      </inp>

      OUTPUT FORMAT:
      Scenario Heading:,
      Scenario:
      '''

  scene = openai.ChatCompletion.create(
          model="gpt-3.5-turbo-0613",
          messages=[
              {"role": "system", "content": gen_prompt},
          ],
          max_tokens=3000, temperature=0.4
          )

  output = scene['choices'][0]['message']['content']
  return output

def answer_evaluation(mental_illness, answer, scene):
  eval_prompt = f'''You are a world renowned Psychiatrist and you are treating a patient. You have diagnosed the patient and found out the Mental Illness
    (delimited by <inp></inp>) they are suffering from. You gave them a hypothetical Scenario (delimited by <scn></scn>) and they responded with an answer delimited by
    <ans></ans>. You have to now evaluate the Severity of the Mental Illness of the patient based on their answer. Use the DSM-5 to evaluate
    their answers and give them a Severity Rating on a scale of 1-10 with 10 being critically severe. You are providing the feedback to the user so generate the feedback with that perspective.
    You should return an evaluation feedback and a integer rating in output.

    <inp>
    Mental Illness: {mental_illness}
    </inp>

    <scn>
    Scenario: {scene}
    </scn>

    <ans>
    Answer: {answer}
    </ans>

    Output Format:
    Evaluation Feedback:
    Rating in integer:
    '''

  eval = openai.ChatCompletion.create(
          model="gpt-3.5-turbo-0613",
          messages=[
              {"role": "system", "content": eval_prompt},
          ],
          max_tokens=3000,
          temperature=0.1
          )
  return eval['choices'][0]['message']['content']


def image_gen(prompt):
    response = openai.Image.create(
    prompt=prompt,
    n=1,
    size="1024x1024")
    image_url = response['data'][0]['url']
    response = requests.get(image_url)

# Check if the request was successful (status code 200)
    # if response.status_code == 200:
    # Open the image using PIL
    image = Image.open(BytesIO(response.content))

    # Specify the path where you want to save the image
    save_path = "downloaded_image.png"

    # Save the image
    image.save(save_path)
    return save_path
    

        