import os
import openai

#openai.organization = "Seoul National University"
openai.api_key =("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")
#print(openai.Model.list())

"""response = openai.Image.create(
          prompt="studio photographic portrait, ",
            n=4,
              size="1024x1024"
              )
image_url = response['data'][0]['url']
print(image_url)
"""
#text= '''I look up from my laptop after a 3-hour study-binge . 3:46am . Eyes still raw from hours of straining , I see something out of the corner of my eye . This is n't out of the ordinary , so I ignore it . Slowly I rise from my chair in to a full-on , bone-crackling , almost orgasmic stretch . A couple squints and my vision focuses on a small envelope sitting in the middle of my floor in front of my closet . `` That 's odd , '' I thought . I did n't recall getting any mail today and I certainly did n't recall dropping it on the floor .    The front of the envelope reads 'Rachel ' scrawled in red sharpie . My heart beat starts to pick up . I absolutely would have remembered getting a hand-addressed letter . Let 's be real , I do n't have a lot of friends and it is a rare occasion indeed that I receive any mail at all that is n't a bill or junk .    I think back to the last time I smoked , about four hours ago . I came down completely at least two hours ago . Smoking usually relaxes me but it 's moments like this I know exactly why I 'm so paranoid .    I consider this , snort to myself and grab the envelope off the ground . I dig a nail in to the side of the seam and tear it open . A small , white piece of stationary flies out as I rip the thing open in excitement . My 21st was a month ago , but knowing my family , this could be some late , much-needed birthday money . I unfold the sheet and begin to read the note . As my eyes scan the sentiment , my mouth runs dry and my heart drops out of my ass .    `` Rachel ,   As I stand here looking at you now , you have never looked more beautiful . Vulnerable . Delicious .   You 'll see what I mean .   Much love ,   Your Roomie ''     My eyes snap towards the closet door . Closed . I do n't close this door because the handle is busted and drifts ajar on its own . The only way it will stay closed is if someone were to hold it from the other side . I audibly gasp when I make this revelation .    As if taking this as an invitation , I see the closet door slowly begin drifting open . I back towards my dresser and retrieve the large pocket knife I was gifted at my high sc I 've ever learned in my years of hunting . I think of my first kill , a middle-aged man . He had this wild look in his eyes as I was carving out his jugular . The look of a person begging to live . I revel in thi'''


Questions=[
        "Question : What is the person doing?",
    ]

"""informations = '''
Question : What is the person's characteristc?

Answer: The person is described as cautious, paranoid, and resourceful.

Question : What is the text's atmosphere?

Answer: The atmosphere of the text is suspenseful and mysterious.

Question : What is the person doing?

Answer: The person is reading a note from their roomie and preparing to defend themselves with a pocket knife.

Question : What had the person done?

Answer: The person had been studying for three hours and had smoked four hours ago.

Question : What will the person do?

Answer: The person will use the pocket knife to defend themselves against whatever is coming out of the closet.

Question : Where the person is now?

Answer: The person is in their room, standing by their dresser with a pocket knife in hand.

Question : How does the person feel like?

Answer: The person feels anxious and vulnerable, but also determined and prepared.

Question : What time is it in the situation?

Answer: It is 3:46am.

Question : Why is the person doing that?

Answer: The person is preparing to defend themselves because they saw something out of the corner of their eye and the closet door was slowly opening.

Question : Is there anyone else?

Answer: No, the person is alone.

Question : Does the person talk?

Answer: No, the person does not talk.

Question : Is the person he or she?

Answer: The person is not explicitly identified as either he or she.

Question : Is this text written by first person view?

Answer: Yes, this text is written in the first person view.
'''
"""

response=openai.Completion.create(
  model="text-davinci-003",
#  prompt= text + Questions[0],
  prompt=informations + "Above these information, make a fiction story in 500-800 words.",
    max_tokens= 1000,
  temperature= 0.7,
  top_p= 1,
  n= 1,
  #stop= "\n"
)
print(response["choices"][0]["text"])
#print(response)
