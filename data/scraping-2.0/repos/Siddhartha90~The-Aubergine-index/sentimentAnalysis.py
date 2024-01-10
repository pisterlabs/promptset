import openai
import os, json
openai.api_key = os.environ.get("OPENAI_API_KEY")


def sentimentAnalysis(reviews, keyword):
# 	reviews = """
# 	0: Loved it!! Awesome service. The food was so good that I didn't have time to take too many pictures. The service was impeccable and very attentive. It was overall a very din experience. Also you need to try the Greek coffee. I loved it so much that ended up having two cups.
# 1: Stunning restaurant, delicious food, great wine selection and incredible staff. I had dinner with some friends a few months ago. Even though we had a reservation it took some time to get us seated and the waiters more than made it up to us by offering outstanding service and a complimentary desert. Highly recommend. Perfect date night spot.
# """
	print(reviews)
	system_msg = "You are an assistant that only returns valid JSON, with no pretext or posttext. "

	# keyword = "gluten free food"
	# Define the user message
	user_json_msg = f"You are answering questions on the following reviews```{reviews}```"
	assistant_json_msg = f"[10]"
	user_msg = f"Given this keyword ```{keyword}, Reply with how the related sentiment is for the given result. Use lateral thinking, for example, if it's implied all they sell is steak, that's probably gluten free"
	response = openai.ChatCompletion.create(model="gpt-4",
											temperature=0,
	                                        messages=[{"role": "system", "content": system_msg},
	                                         {"role": "user", "content": user_json_msg},
	                                         {"role": "assistant", "content": assistant_json_msg},
	                                         {"role": "user", "content": user_msg},
	                                         ])


	print(response)
	sentiment = response.choices[0].message.content
	
	# content = '"content": "{\n  \"keyword\": \"gluten free food\",\n  \"sentiment\": \"neutral\",\n  \"explanation\": \"The reviews do not mention gluten free food specifically, but the overall sentiment is positive.\"\n}"'
	print(type(sentiment))
	# json_acceptable_string = content.replace("'", "\"")

	sentimentJson = json.loads(sentiment)
	print(type(sentimentJson))
	# print("sentiment - " + sentimentJson["sentiment"])
	# print("reason - " + sentimentJson["reason"])
	return sentimentJson

# sentimentAnalysis()
