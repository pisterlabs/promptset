
import os
import openai
 
import sys
openai.api_key =  "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"


def genstory(query) :
  response = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=[
      {
        "role": "user",
        "content": "Generate a story on: {} ".format(query)
      },
      {
        "role": "assistant",
        "content": "Once upon a time in a small village, there lived a young boy named Rahul. He was a curious and imaginative child, always in search of new adventures. One sunny day, while exploring his grandfather's attic, Rahul stumbled upon an old dusty box. As he carefully opened the box, his eyes gleamed with excitement when he saw a beautifully crafted kite.\n\nRahul's mind immediately raced with ideas, and he couldn't wait to fly the kite. He rushed downstairs, ready to go outside and test it out. As he reached the field near his house, he felt a gentle breeze that seemed to be calling him. Without wasting any time, Rahul unfolded the kite and started attaching the string.\n\nHis heart danced with excitement as the kite soared high into the sky. It swirled and twirled with grace, dancing among the fluffy white clouds. Rahul was in awe of its beauty; it felt as if he held the power of the sky within his hands.\n\nAs Rahul gazed up at the kite, he noticed something peculiar. It seemed to be leading him somewhere. Guided by an invisible force, Rahul followed the kite as it elegantly drifted through the village. The villagers watched in astonishment as the boy and the kite weaved through the streets.\n\n"
      }
    ],
    temperature=1,
    max_tokens=256,
    top_p=1,
    frequency_penalty=0,
    presence_penalty=0
  )
  

  if __name__ == "__main__":
    prompt = sys.argv[1]
    result = genstory(prompt)
    print(result)
