import openai
import json
import random

def generate_prompt_withou_cot(
    reviews
    ):
    """
        test_case:

    """
    detect_Prompt = """Please retrieve the potential outlier reviews(with rating labels) from Yelp. Rate their outlier score on a scale from 0 to 1, and explain why"""
    prompt = detect_Prompt
    for i in reviews["0"][:3]:
        rating_score =i[3]
        review_label = i[2]
        user_label = i[1]
        user_id = i[0]
        review = i[4]
        prompt+=f"rating_score: {rating_score},review:{review}"
    return prompt

def parse_answer(
    prompt,
    ):
    """"
    把大括号里的内容正则匹配
    """
    pass

def llama_generate(prompt, models):
    messages = [
        {
            "role": "system",
            "content": "You are an AI assistant that helps people find information.",
        }
    ]

    # Add the prompt to the messages
    message_prompt = {"role": "user", "content": prompt}
    messages.append(message_prompt)
    
    # Calculates the total number of tokens
    total_length = sum(len(message["content"].split()) for message in messages)
    if total_length+256 > 3500:
        raise ValueError("Total token count exceeds the model limit.")

    flag = True
    i = 0
    MODEL = models["data"][0]["id"]
    
    response = openai.ChatCompletion.create(
        model=MODEL,  # llama
        messages=messages,
        temperature=0, # when temperature is 0, the output is fixed. When increasing the temperature, the output will be different each time
        max_tokens=256, # the max token length of input
        # top_p=0.9, # Choose the top p tokens as Greedy Policy when inference
        frequency_penalty=0,
    )
    # self consistency
    result = response["choices"][0]["message"]["content"]
    return result

def generate_mapper_prompt(
    reviews
    ):
    CoT_prompt = """Q: Review list:
    review1: Let me begin by saying that there are two kinds of people, those who will give the Tokyo Hotel 5 stars and rave about it to everyone they know, or... people who can't get past the broken phone, blood stains, beeping fire alarms, peg-legged receptionist, lack of water pressure, cracked walls, strange smells, questionable elevator, televisions left to die after the digital conversion, and the possibility that the air conditioner may fall out the window at any moment. That being said, I whole-heartedly give the Tokyo Hotel 5 stars. This is not a place to quietly slip in and out of with nothing to show but a faint memory of the imitation Thomas Kinkade painting bolted to the wall above your bed. And, there is no continental breakfast or coffee in the lobby. There are a few vending machines, but I'm pretty sure they wont take change minted after 1970. Here your senses will be assaulted, and after you leave you will have enough memories to compete with a 1,000 mile road-trip. I beg anyone who is even mildly considering staying here to give it a chance. The location is prime. We were able to walk down Michigan Ave and the river-walk in the middle of the night, all without straying too far from the hotel. There is a grocery store a block away and parking (which may cost more that your hotel room) across the street. Besides, this place is cheap. Super-cheap for downtown Chicago. The closest price we found in the area was four times as expensive. But, be sure to grab some cash. They don't accept credit cards. Some rules though: - Say hello to Clifton Jackson, the homeless guy by Jewel-Osco. - Buy him a drink, some chicken and look him up on Facebook. - Stay on the 17 floor. All the way at the top. - Go out the fire escape (be sure to prop the door open or you'll have a looong walk down) - Be very very careful. - Explore. (Yes, that ladder will hold your weight) - Be very very careful. - Don't be alarmed by any weird noises you hear. - Spend the night on the roof. 17 stories up, in the heart of Chicago. - Write your own Yelp review. I want to see that others are getting the Tokyo Hotel Experience. - Check out is at noon. Be sure to drink lots of water. - Spend the next day hung over. And... Please be careful on the roof.
    rating score on Yelp:5.0
    review2: The only place inside the Loop that you can stay for $55/night. Also, the only place you can have a picnic dinner and get a little frisky on the 17th floor roof and then wake up in your room the next morning to an army of ants going in on your picnic leftovers.
    rating score on Yelp:3.0
    review3: I have walked by the Tokyo Hotel countless times. It reminds me of the type place that Peter Parker lives in in the Spiderman movies, Or the sort of place Marla Singer might live in, or maybe it is a carbon copy of the Hotel Zamenhof. Basically it is scuzzy and sleazy, through and through it is a fleabag, and I love it. Yesterday I finally walked in. I don't know why. Maybe because I just told my roommates and my landlord that I wouldn't be signing another lease, essentially leaving me "pre-homeless". Maybe because earlier that day I told boss that I'm quitting my job at the end of July, leaving me "pre-unemployed". Maybe because after all this time teetering between living on the edge of salvation and the brink of self destruction, I'm finally choosing my path, I'm hitting bottom. I'm giving up. Shit maybe I'm Nick Cage in Leaving Las Vegas, but where is my Elizabeth Shue? Who will look after me, who will walk me gently into that good night? If I choose to stay in this city, the city I love, then I may just stay in the "Hotel Tokyo". Because if I choose to stay, and that is a big if, then I'm going to need to stay somewhere cheap, with like-minded vagabonds, vagrants and naredowells. I'll need help, I'll need guidance.
    rating score on Yelp:5.0
    review4: This place is disgusting, absolutely horrible, it is my second stay here, there is a strange stain on the other side of the bed I stay on, and my friends and I are rendered air conditionless, but it is 55 for downtown Chicago, what am I suppose to do?
    rating score on Yelp:3.0
    review5: Disgusting!!! There is literally duct tape holding together the elevator. Crack heads live here.. not safe! Walked in, bought a room.. saw it and left. No locks on doors, no parking, no air conditioning, no way! For the $$ you pay for the room ($ 55 a night) and parking ($ 38 a night) you could stay down the street at The Ohio House and NOT sleep with one eye open. Would not recommend this hotel to my worst enemy. Seriously, DO NOT STAY HERE!
    rating score on Yelp:1.0
    review6: Yes I am giving this place a 4 star review. No, not because it one of the nicer places I've stayed but because of the whole experience itself. Yes, the elevator is a tad sketchy being the fact that it takes an unusual amount of time for it to even start moving once youre in it. But i did have locks on the door and while being peg-legged the receptionist was very friendly, let us look at the room before we checked in and let us know what we were in for. I did put my own sheets on the bed and wore shoes almost all the time. I liked the charm of the place plugs cut off of all the electrical devices in the room and all. My only complaint I have about the place is the creepy guy who lives on the 14th floor right around the corner from the room we had and right across the hall from the elevator...he likes to sit with his door ajar on a chair in the corner of his room only wearing his tidy whiteys and just stares out the door. While being creepy he didnt bother with us at all. Some might say it was not the safest decision for 2 eighteen/ninteeen year old girls but was the only place in/around the city where you didnt have to be 21 to check in and was at a decent price and its nice they dont jack up the rates for Lollapalooza like every single other hotel/motel in the area. I would recommend Tokyo Hotel to a person whos willing to experience something a tad out of the ordinary or if youre just as broke as I was and desperately needed somewhere to stay for a weekend.
    rating score on Yelp:4.0
    review7: After reading previous reviews I had to see this place. After coming into Chicago for a night I went to take a tour of the rooms. It was absolutely horrible. My friend and I left to have a few drinks only then could I bring myself to say we should stay there. $55 was reasonable which left me a few hundred more to spend out for the night. When I came in a 3:30 AM I decided to take a look at the other floors and the roof. It was worth the experience the views are amazing and if you are a fan of history this place has it. In 1936 Serial Killer Robert Nixon killed one of his victims on the fifth floor of this hotel. The over night clerk says he isn't sure which room it happened in but room 515 is creepy. Since I know the murder took place on the fifth floor that could be the spot. I did drop my toothbrush on the floor so it was lost forever and yes it smells and yes the walls are cracked and crumbling but to be in the heart of the city for that cheap of a price was well worth it. We were on 16th floor and the other people there were Belgians who stay there every year when they come to town. I really enjoyed the experience but was very happy to take a hot Shower somewhere else the next day...
    rating score on Yelp:4.0
    review8: I want to overdose in this hotel and go out full blown rock n roll style
    rating score on Yelp:5.0
    review9: Here's the deal. If you want a cheap hotel in Chicago you CAN find a nice one and avoid this joint. Go to Hotwire and get a room in a very nice downtown Chicago hotel for $60...only a few dollars more.
    rating score on Yelp:1.0

    A: Outlier reviews ID: review6,review7,review8, review9
    confidence for outlier: review6(HIGH),review7(HIGH),review8(MEDIUM), review9(MEDIUM)
    reasons:
    - review6: Despite describing several negative aspects like a sketchy elevator and unusual encounters, the review still awards a high rating. This is unusual because a 4-star rating typically implies a much higher standard of quality and comfort, making this review inconsistent with the norm.
    - review7: The reviewer focuses on the historical aspect and personal adventure rather than the hotel's amenities or service quality. This unconventional perspective, along with the positive spin on otherwise negative aspects (like poor conditions), deviates from the typical criteria used in hotel reviews.
    - review8: The content of this review is highly unusual, focusing on a desire for a dramatic and negative personal event rather than commenting on the hotel itself. This extreme and non-standard content makes it an outlier.
    - review9: This review doesn't provide a personal experience of staying at the hotel. Instead, it offers advice on finding alternative accommodations. The lack of direct commentary about the hotel's features or services, and focusing solely on price comparison, is atypical for a hotel review.

    Q: Review list: """
    
    detect_Prompt = """
    Give you a series of reviews on Yelp and its rating score, the reviews is all from one target(hotel, food), please detect all the reviews outlier or not for each review, Give your answer in the form below:
        reviews ID:<give the reviews ID for rating the confidence as HIGH and MEDIUM>
        confidence for outlier: <one of: LOW, HIGH, MEDIUM>;
        reasons:<text for why it is outlier>;"""

    prompt = detect_Prompt+CoT_prompt
    num=0
    ground_truth = []
    for i in reviews["8"][10:15]:
        rating_score =i[3]
        review_label = i[2]
        user_label = i[1]
        user_id = i[0]
        review = i[4]
        num+=1
        ground_truth.append({str(num):review_label})
        prompt+=f"""review{num}: {review}rating score on Yelp:{rating_score}\n"""
    return prompt, ground_truth

def chatGPT_generate(prompt):
    messages = [
        {
            "role": "system",
            "content": "You are an AI assistant that helps people find information.",
        }
    ]
    message_prompt = {"role": "user", "content": prompt}
    messages.append(message_prompt)
    flag = True
    i = 0
    # 请教我一个数学题目
    MODEL = "gpt-3.5-turbo"
    response = openai.ChatCompletion.create(
        model=MODEL,  # 35-turbo
        messages=messages,
        temperature=0, #temperature是0的时候，给定固定的input，他的每次输出都是固定的，当你调高温度的时候，每次输出都会不一样
        max_tokens=300, #input的token限制是多少
        # top_p=0.9, # 当你让你模型做inference的时候，他选择前多少个作为贪心策略
        frequency_penalty=0,
    )
    # self consistency 投票法
    result = response["choices"][0]["message"]["content"]
    return result

if __name__ == '__main__':
    # Modify OpenAI's API key and API base to use vLLM's API server.
    openai.api_key = "EMPTY"
    openai.api_base = "http://localhost:8000/v1"

    # List models API
    models = openai.Model.list()

    with open(
        "./Data/PRUNED_DATA_prod-ID_usr-ID_rating_label_review.json",
        "r",
        ) as f:
        data = json.load(f)
    # only prompt and answer via chatGPT generate func
    # data = random.sample(data, 30)
    prompt, ground_truth = generate_mapper_prompt(data)

    ### Generate using chatGPT
    # result = chatGPT_generate(prompt)
    ### Generate using llama
    result = llama_generate(prompt, models)
    
    # save results to a file
    with open("results.txt", "w") as f:
        f.write("Models:\n")
        f.write(str(models) + "\n")

        f.write("\nChat completion results:\n")
        f.write(str(result))
    print("Results saved to results.txt")
    
    print("prompt:",prompt)
    print("result",result)
    print("ground truth:",ground_truth)