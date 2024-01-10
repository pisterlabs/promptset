import cohere

from server.env import COHERE_API_KEY

prompt = f"""Passage: It charges fast as advertised. What really sells to me is the size. It is smaller than an iPad charger and charges faster than the MacBook charger. I can charge my headphones, my phone, my iPad and my laptop with a single charger, AND IT FITS. I am a student and I stay on campus in between lectures so not needing a brick for each device is great when I study all day. I definitely recommend this to anyone. I go out of my way to put my friends on it. I even bought one with an EU plug from amazon.nl so I can use it when I go back home. I was so fascinated by how well built it is and how good it works, I went back to repeat my electronics courses to see how it could be done. When I need a cable or an adapter I go to the Anker page before googling it. Even their power strips are my go to.

TLDR: fast, small, built, fits
--
Passage: Looking for socks that are thin, loose fitting, thinner after the first wash and no thickness/bulk cushioning? If so, add to cart. If not, move on to another brand. I thought ordering the Hanes brand for the first time. they would be good quality. I was wrong!!!

TLDR: thin, loose, cushionless
--
Passage: We use several of these at work, my mom and I also each have one at home. They are not consistent at all. One day #1 and #2 are completely different readings from each other, the next they're the same, but #3 is different and #4 says something completely different. #1 will give 4 different readings in a row varying over 1 degree one day, then the same reading 4x in a row the next day. I cannot trust them at all without using multiple ones and hoping at least 2 work that day. My mom's does a weird flashing thing even with new batteries. Mine usually reads differently than another model I have. They're also not super fast.

TLDR: inconsistent, weird, slow
--
Passage: You get what you pay for. The mirror feels cheap and is very flimsy. The handle and bezels will most definitely snap and break apart if you are slightly too rough with it. The mirror itself however, I have no complaints about. It's well polished and clear. Perfect for everyday use, just be extra careful with it.

TLDR: cheap, flimsy, polished
--
Passage: Love this keyboard, it is the best one I have ever used at this price point. I dislike mechanical keyboards and since they are kind of the standard today I have bought a lot of non-mechanical keyboards in the $20-$30 price range looking for one that I like. This one by far is the winner. It is sturdy, the smooth texture of the metal frame feels nice and is easy to keep clean. My only nit-pick is that the brightness adjustment of the LED does not go as low as I would like, the lowest setting is still a bit bright.

TLDR: sturdy, smooth, easy
--
Passage: This is a great shirt, fits well and is very comfortable. However with my experience, the Jerzees brand does seem to run a bit small, unless you're one that prefers a snug fit. Personally, I like my shirts a little on the loose side. If you read some of the other reviews, I think you'll find that other people found that the sizing was not what they expected. So I would definitely order a size up from what you normally wear. I normally wear a large. I got this shirt in an extra large and it fits and feels great.

TLDR: fits, comfortable
--
Passage: Disaster. Have tried to self-install, have our HVAC tech install, and no one can consistently get this thermostat to control our brand new HVAC system. It blows cold when the heat is on, blows nothing when the a/c is on. Amazon won’t let me return because I’m out of my window… and Google won’t let me return because I bought on Amazon. SCAM.

TLDR: disaster, useless, scam
--
Passage: I have been using Chapstick for a while without any issues. Yesterday night I tried the Burt's bees lip balm and then next morning, my lips got so irritated and peeled off compared to the previous days. Initially I wasn't sure what caused this, then after a while I realized that I applied a different lip balm and then came here to Amazon to see the reviews having irritated skin and lips getting more chapped. I thought that Burt's bees was a good brand. I am so disappointed with the results.

TLDR: irritated, disappointed
--
Passage: Yeah too small

TLDR: small
--"""


def run_cohere(stars_and_reviews):

    print("cohere start")

    new_reviews = {1: [], 2: [], 3: [], 4: [], 5: []}
    co = cohere.Client(COHERE_API_KEY)

    for key in stars_and_reviews:
        for review in stars_and_reviews[key]:
            custom_prompt = f"{prompt}\nPassage: {review}\n\nTLDR:"
            response = co.generate(
                model='medium',
                prompt=custom_prompt,
                max_tokens=10,
                temperature=0.3,
                stop_sequences=["--"])
            keywords = response.generations[0].text.strip("\n--").lstrip(" ").split(", ")
            new_reviews[key] = new_reviews[key] + keywords

    keyword_freq = {}
    useless_keywords = ["disappointed", "angry", "frustrated", "unhappy", "upset", "pleased", "happy", "impressed",
                        "satisfied", "overjoyed", "shocked", "thrilled", "amazed", "delighted", "encouraged", "grateful",
                        "relieved", "great", "good", "nice", "bad", "okay", "fine", "amazing", "horrible", "terrible",
                        "decent", "average", "fantastic", "awful", "perfect", "best", "worst", "excellent", "mediocre",
                        "incredible", "pretty", "poor", "so-so", "wonderful", "easy"]

    for star in new_reviews:
        for keyword in new_reviews[star]:
            if keyword in keyword_freq:
                keyword_freq[keyword]['freq'] += 1
                keyword_freq[keyword]['ratings'] += star
            else:
                keyword_freq[keyword] = {'freq': 1,
                                         'ratings': star}

    keyword_freq = sorted(keyword_freq.items(), key=lambda x: x[1]["freq"], reverse=True)

    for item in keyword_freq:
        item[1]['ratings'] = round((item[1]['ratings'] / item[1]['freq']), 1)

    keyword_freq[:] = [item for item in keyword_freq if not item[0] in useless_keywords]
    keyword_freq = keyword_freq[:5]

    print("cohere end")
    return keyword_freq

