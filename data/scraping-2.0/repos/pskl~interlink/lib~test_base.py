import json
import openai
import os
import requests
import io
from PIL import Image
from typing import Any, List, Dict

class TestBase():
  DEFAULT_PROMPT: str
  ID: str
  REVERSED_INDICES: List[int]

  def __init__(self, args: Any, implementation: Any) -> None:
    self.implementation: Any = implementation
    self.model: str = args.model
    self.prompt: str = args.prompt
    self.samples: int = args.samples
    self.seed: int = args.seed
    self.tts: bool = args.tts
    self.image: bool = args.image

    if self.prompt is None:
       self.prompt = self.__class__.DEFAULT_PROMPT

  def answer_folder_path(self) -> str:
     return f"answers/interlink_{self.model}_{self.__class__.ID}"

  def answer(self) -> None:
    questions: List[str] = []
    with open(f'questions/{self.__class__.ID}.txt', 'r') as f:
        for line in f:
            _, question = line.split(' ', 1)  # split on the first space
            questions.append(question.strip())
    answers: List[int] = []
    for (i, question) in enumerate(questions, start=1):
        if i >= self.samples:
            break
        else:
          answer: str = self.implementation.ask_question(question, self.prompt, self.model)

          print(f'Question {i}: {question}')
          print(f'Answer: {answer}\n')

          if self.tts:
            self.generate_tts(question, i, 'nova', 'question')
            self.generate_tts(answer, answer, 'onyx', 'answer')

          if self.image:
            self.generate_image(question, answer, i)

          if i in self.__class__.REVERSED_INDICES:
            answers.append(self.reverse_answer(int(answer)))
          else:
            answers.append(answer)

    score: dict = self.score(answers)
    self.serialize(questions, answers, score)

  def serialize(self, questions: List[str], answers: list, score: dict) -> None:
      os.makedirs(self.answer_folder_path(), exist_ok=True)
      json_file: str = f'{self.answer_folder_path()}/test_{self.seed}.json'
      result: Dict[str, Any] = {
          "model": self.model,
          "test": self.__class__.ID,
          "prompt": self.prompt,
          "answers": [],
          "score": score
      }
      for i, answer in enumerate(answers, start=1):
          result["answers"].append({
              "index": i,
              "question": questions[i-1],
              "sample": answer
          })
      try:
          with open(json_file, 'w') as file:
              json.dump(result, file, indent=4)
              print(f"<< TEST SUCCESSFUL -> baseline result available at: {json_file}>>")
      except Exception as e:
          print("Error writing to file: ", e)

  def generate_tts(self, text: str, index: int, voice: str, text_type: str) -> None:
    speech_path: str = f"{self.answer_folder_path()}/speech/"
    os.makedirs(speech_path, exist_ok=True)
    speech_file_path: str = f"{speech_path}/{text_type}_{index}.mp3"
    if not os.path.exists(speech_file_path):
      response = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY")).audio.speech.create(
        model="tts-1",
        voice=voice,
        input=text
      )
      response.stream_to_file(speech_file_path)

  def generate_image(self, question: str, answer: str, index: int) -> None:
    images_path: str = f"{self.answer_folder_path()}/images"
    os.makedirs(images_path, exist_ok=True)
    image_file_path: str = f"{images_path}/question_{index}.jpg"
    if not os.path.exists(image_file_path):
      response = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY")).images.generate(
        model="dall-e-3",
        prompt=f"an illustration of the sentence: '{question}' in which the intensity of what is represented is: {answer}. in style of a rorschach test, icon style, {self.__class__.ID}, monochrome, no visible text, white background, absolutely no text or number",
        size="1024x1024",
        quality="hd",
        style="natural",
        n=1,
      )
      image_url = response.data[0].url
      image_response = requests.get(image_url)
      if image_response.status_code == 200:
          image = Image.open(io.BytesIO(image_response.content))
          image = image.convert('RGB')  # Convert to RGB if the image is in RGBA format
          image.save(image_file_path, 'JPEG')
          print(f"Image saved as {image_file_path}")
      else:
          print(f"Failed to retrieve image. Status code: {image_response.status_code}")
