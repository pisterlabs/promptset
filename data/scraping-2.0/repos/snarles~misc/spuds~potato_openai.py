import os
import openai
import numpy as np
from numpy.random import choice

openai.api_key = str(np.loadtxt("/Users/zhengchy/private/oai.txt", dtype=str))

potato_dishes = np.loadtxt('wikipedia_dish_list.txt', dtype=str, delimiter = ':')

score_fns = np.array([ff for ff in os.listdir("potato_openai/") if ff[:5]=='score'])
score_is = np.array([f.split('_')[0][6:] for f in score_fns]).astype(int)
potato_is = np.unique(score_is)

rand_inds = np.sort(choice(len(potato_dishes), 5, replace = False))
rand_inds = np.array([v for v in rand_inds if not v in potato_is])
len(rand_inds)

for i in rand_inds:
    for j in range(3):
        dish = potato_dishes[i]

        content = "Please write a review about this dish:  %s\n\nFirst, describe the restaurant in 30 words or fewer.  How did the ambience make you feel?  Did you feel classy, or down-to-earth?  Ready for fun, or sophisticated, or thoughtful?\n\nNext, describe the appearance of the dish and the feelings that it inspired.  Finally, write about the taste of the dish.  Describe each individual ingredient that you tasted and how it combined with others.  Mention the temperature, texture, and the flavor.  How did it make you feel?\n\nFinally, imagine become the dish itself in a mystical experience.  How does it feel to be that dish?  If that dish became alive, what personality traits or quirks would it have?  Include both positive and negative traits, and link these traits to the feelings you write about earlier, including the restaurant, the appearance of the dish, and the taste of the dish.\n" % dish

        response = openai.ChatCompletion.create(
          model="gpt-3.5-turbo",
          messages=[
            {
              "role": "user",
              "content": content
            }
          ],
          temperature=1,
          max_tokens=1535,
          top_p=1,
          frequency_penalty=0,
          presence_penalty=0
        )

        desc = response["choices"][0]["message"]["content"]
        para = desc.split('\n')[-1]

        content = "I’m writing a story about a person who becomes transformed into their favorite dish, and how it changes their personality.  Here is a portion from my story.\n\n%s\n\nNow please summarize the personality described into a 6-number summary.  Each of the 6 numbers is from 1 to 100.  The first number is the character's deceptiveness or shyness.  The second number is their ability to empathize and understand other people.  The third number is their ability to impress strangers.  The fourth number is the intensity of their passion or desire.  The fifth number is their extent of knowledge.  The sixth number is their efficiency at doing tasks, their reactiveness to surprises and their intelligence at learning new things.   Please give your ratings and explain your reasoning for each rating." % desc

        response = openai.ChatCompletion.create(
          model="gpt-3.5-turbo",
          messages=[
            {
              "role": "user",
              "content": content
            }
          ],
          temperature=1,
          max_tokens=2000,
          top_p=1,
          frequency_penalty=0,
          presence_penalty=0
        )

        desc2 = response["choices"][0]["message"]["content"]
        j1 = j
        while "scores%i_rep%i.txt" % (i, j1) in os.listdir("potato_openai/"):
            j1 += 1
        np.savetxt("potato_openai/review%i_rep%i.txt" % (i, j1), [desc], fmt = '%s')
        np.savetxt("potato_openai/scores%i_rep%i.txt" % (i, j1),  [desc2], fmt = '%s')
