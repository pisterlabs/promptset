from gtts import gTTS
import json
# import openai

# def generate_audiobook(self, audioprompt):
#     self.audiotext_messages.append(
#         {"role": "user", "content": audioprompt},
#     )

#     chat = openai.ChatCompletion.create(
#         model="gpt-3.5-turbo",max_tokens=2000, messages=self.audiotext_messages
#     )

#     reply = chat.choices[0].message.content

#     self.audiotext_messages.append({"role": "assistant", "content": reply})

#     with open('./data/audiotext_input_data.json', 'a') as slide_data:
#         slide_data.write(str(reply)+",")
    
#     # Convert the text format into a Python dictionary
#     reply_dict = json.loads(reply)
#     return reply_dict
def text_to_speech(text):
    output_file="./ebook_and_audiobook/input.mp3"
    tts = gTTS(text=text, lang="en")
    tts.save(output_file)
    print(f"Text converted to speech and saved as {output_file}")
text="In conclusion, analyzing YouTube data can provide valuable insights into a channel's performance, content strategy, and audience engagement. By utilizing the YouTube API and various data analysis techniques, we can extract and analyze data such as video views, likes, comments, and titles. This allows us to understand viewer preferences, trends, and patterns, which can help creators optimize their content and engage with their audience more effectively. Whether you're a content creator, data analyst, or simply interested in learning more about YouTube analytics, exploring and analyzing YouTube data can be an exciting and informative project. So go ahead, dive into the world of YouTube data analysis, and uncover the stories hidden within the numbers and metrics!"
text_to_speech(text)