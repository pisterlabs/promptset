import openai
import os

openai.api_key = os.getenv("OPENAI_API_KEY")

response = openai.Completion.create(
  model="text-davinci-003",
  prompt="How to become famous youtuber",
  max_tokens=300,
)
print(response)

# {
#   "choices": [
#     {
#       "finish_reason": "length",
#       "index": 0,
#       "logprobs": null,
#       "text": "\n\n1. Choose a Topic: The first step to becoming a famous YouTuber is to choose a topic that you\u2019re passionate about. People love to tune in to watch people who are passionate about and knowledgeable about their topic. \n\n2. Get the Gear: Invest in the necessary equipment to create high-quality and visually appealing videos. You don\u2019t need the latest and greatest in equipment, but it doesn\u2019t hurt to splurge a little on higher-end recording hardware and lighting.\n\n3. Work on Your Brand: Spend time and effort honing your brand and turning it into something people can recognize. This can include developing logos, video series, and a particular style.\n\n4. Get Creative: In order to stand out, you\u2019ll have to create content that stands out. Don\u2019t be afraid to think outside the box and come up with creative, original content.\n\n5. Become a Social Media Rock Star: The key to growing your audience is to get people to recognize you and your content. You can do this by building a presence on social media and driving your viewers to your channel.\n\n6. Be Consistent: Consistency is key for growing your audience. You want to upload regularly and form relationships with viewers.\n\n7. Invest in Paid Promotion: You can grow your audience faster by investing in paid promotion. You can promote your channel and videos on social media"
#     }
#   ],
#   "created": 1676960840,
#   "id": "cmpl-6mGPQv5uAbTBlj8tGlwUBHNGrsrgK",
#   "model": "text-davinci-003",
#   "object": "text_completion",
#   "usage": {
#     "completion_tokens": 300,
#     "prompt_tokens": 7,
#     "total_tokens": 307
#   }
# }

# print(response['choices'][0]['text'])


