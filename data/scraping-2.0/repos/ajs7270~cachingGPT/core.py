from langchain import PromptTemplate, LLMChain
from util import convert_and_caching_prob, convert_digit2alph, float_to_words
import json
import random
import re
from itertools import permutations

PoT_template = """
Read the following passages to answer questions with Python code, store the result as a 'ans' variable:

# Passage: James bought number0 red and number1 blue stickers, he used number2 red sticker on his fridge and number3 blue stickers on his laptop.
# Question: How many red stickers does James have?
number0 = 93
number1 = 10
number2 = 31
number3 = 7
ans = number0 - number2

# Passage: Allen went to supermarket to buy eggs, each egg costs number0 dollars, if the discount is number1 dollars.
# Question: How much do you have to pay to buy for each egg?
number0 = 80
number1 = 29
ans = number0 - number1

# Passage: Dianna collects both cases and books. He bought number0 cases and number1 books from the store. Now he has number2 cases and number3 books.
# Question: How many books did danny have at first?
number0 = 22
number1 = 5
number2 = 57
number3 = 25
ans = number3 - number1

# Passage: There were number0 chickens and number1 sheeps at the farm, some of chickens and sheeps were sold. There are number2 chickens and number3 sheeps left now.
# Question: How many chickens were sold?
number0 = 108
number1 = 20
number2 = 87
number3 = 18
ans = number0 - number2

# Passage: Katty scored number0 goals on monday, number1 goals on tuesday and number2 goals on wednesday.
# Question: How many did Katty score on monday and wednesday?
number0 = 2
number1 = 8
number2 = 9
ans = number0 + number2

# Passage: There are number0 girls and number1 boys in the Masquerade, number2 more girls and number3 more boys joined. 
# Question: How many more girls than boys are in the Masquerade?
number0 = 5
number1 = 4
number2 = 12
number3 = 7
total_girls = number0 + number1
total_boys = number2 + number3
ans = total_girls - total_boys

# Passage: Joseph and Getty went to buy ice creams, they together bought number0 ice creams. On the way back, Joseph ate number1 of the ice creasm, and he has number2 ice creams left now. 
# Question: How much ice cream did Getty purchase?
number0 = 36
number1 = 12
number2 = 2
num_ice_creams_bought_by_joseph = number2 + number1
ans = number0 - num_ice_creams_bought_by_joseph

# Passage: {passage}
# Question : {question}
{cache}
"""


PoT_template_inplace = """
Read the following passages to answer questions with Python code, store the result as a 'ans' variable:

# Passage: James bought 93(number0) red and 10(number1) blue stickers, he used 31(number2) red sticker on his fridge and 7(number3) blue stickers on his laptop.
# Question: How many red stickers does James have?
number0 = 93
number1 = 10
number2 = 31
number3 = 7
ans = number0 - number2

# Passage: Allen went to supermarket to buy eggs, each egg costs 80(number0) dollars, if the discount is 29(number1) dollars.
# Question: How much do you have to pay to buy for each egg?
number0 = 80
number1 = 29
ans = number0 - number1

# Passage: Dianna collects both cases and books. He bought 22(number0) cases and 5(number1) books from the store. Now he has 57(number2) cases and 25(number3) books.
# Question: How many books did danny have at first?
number0 = 22
number1 = 5
number2 = 57
number3 = 25
ans = number3 - number1

# Passage: There were 108(number0) chickens and 20(number1) sheeps at the farm, some of chickens and sheeps were sold. There are 87(number2) chickens and 18(number3) sheeps left now.
# Question: How many chickens were sold?
number0 = 108
number1 = 20
number2 = 87
number3 = 18
ans = number0 - number2

# Passage: Katty scored 2(number0) goals on monday, 8(number1) goals on tuesday and 9(number2) goals on wednesday.
# Question: How many did Katty score on monday and wednesday?
number0 = 2
number1 = 8
number2 = 9
ans = number0 + number2

# Passage: There are 5(number0) girls and 4(number1) boys in the Masquerade, 12(number2) more girls and 7(number3) more boys joined. 
# Question: How many more girls than boys are in the Masquerade?
number0 = 5
number1 = 4
number2 = 12
number3 = 7
total_girls = number0 + number1
total_boys = number2 + number3
ans = total_girls - total_boys

# Passage: Joseph and Getty went to buy ice creams, they together bought 36(number0) ice creams. On the way back, Joseph ate 12(number1) of the ice creasm, and he has 2(number2) ice creams left now. 
# Question: How much ice cream did Getty purchase?
number0 = 36
number1 = 12
number2 = 2
num_ice_creams_bought_by_joseph = number2 + number1
ans = number0 - num_ice_creams_bought_by_joseph

# Passage: {passage}
# Question : {question}
{cache}
"""


PoT_d2a_template = """
Read the following passages to answer questions with Python code, store the result as a 'ans' variable:

# Passage: James bought Ninety Three red and Ten blue stickers, he used Thirty One red sticker on his fridge and Seven blue stickers on his laptop.
# Question: How many red stickers does James have?
original_red_stickers = 93
used_red_stickers = 31
ans = original_red_stickers - used_red_stickers

# Passage: Allen went to supermarket to buy eggs, each egg costs Eighty dollars, if the discount is Twenty Nine dollars.
# Question: How much do you have to pay to buy for each egg?
original_egg_price_in_dollars = 80
discount_dollars = 29
ans = original_egg_price_in_dollars - discount_dollars

# Passage: Dianna collects both cases and books. He bought Twenty Two cases and Five books from the store. Now he has Fifty Seven cases and Twenty Five books.
# Question: How many books did danny have at first?
num_books_bought_at_store = 5
num_books_now = 25
ans = num_books_now - num_books_bought_at_store

# Passage: There were One Hundred Eight chickens and Twenty sheeps at the farm, some of chickens and sheeps were sold. There are Eighty Seven chickens and Eighteen sheeps left now.
# Question: How many chickens were sold?
num_chicken_before = 108
num_chicken_now = 87
ans = num_chicken_before - num_chicken_now

# Passage: Katty scored Two goals on monday, Eight goals on tuesday and Nine goals on wednesday.
# Question: How many did Katty score on monday and wednesday?
num_goals_on_monday = 2
num_goals_on_wednesday = 9
ans = num_goals_on_monday + num_goals_on_wednesday

# Passage: There are Five girls and Four boys in the Masquerade, Twelve more girls and Seven more boys joined. 
# Question: How many more girls than boys are in the Masquerade?
num_girls_before = 5
num_girls_joined = 12
num_boys_before = 4
num_boys_joined = 7
total_girls = num_girls_before + num_girls_joined
total_boys = num_boys_before + num_boys_joined
ans = total_girls - total_boys

# Passage: Joseph and Getty went to buy ice creams, they together bought Thirty Six ice creams. On the way back, Joseph ate Twelve of the ice creams, and he has Two ice creams left now. 
# Question: How much ice cream did Getty purchase?
num_ice_creams_bought_by_joseph = 2 + 12
total_ice_creams = 36
ans = total_ice_creams - num_ice_creams_bought_by_joseph

# Passage: {passage}
# Question : {question}
"""

d2e_num_eng_template = """
Read the following passages to answer questions with Python code, store the result as a 'ans' variable:

# Passage: James bought Ninety Three red and Ten blue stickers, he used Thirty One red sticker on his fridge and Seven blue stickers on his laptop.
# Question: How many red stickers does James have?
# Hint:
Ninety Three = 93
Ten = 10
Thirty One = 31
Seven = 7
# Python:
original_red_stickers = 93
used_red_stickers = 31
ans = original_red_stickers - used_red_stickers

# Passage: Allen went to supermarket to buy eggs, each egg costs Eighty dollars, if the discount is Twenty Nine dollars.
# Question: How much do you have to pay to buy for each egg?
# Hint:
Eighty = 80
Twenty Nine = 29
# Python:
original_egg_price_in_dollars = 80
discount_dollars = 29
ans = original_egg_price_in_dollars - discount_dollars

# Passage: Dianna collects both cases and books. He bought Twenty Two cases and Five books from the store. Now he has Fifty Seven cases and Twenty Five books.
# Question: How many books did danny have at first?
# Hint:
Twenty Two = 22
Five = 5
Fifty Seven = 57
Twenty Five = 25
# Python:
num_books_bought_at_store = 5
num_books_now = 25
ans = num_books_now - num_books_bought_at_store

# Passage: There were One Hundred Eight chickens and Twenty sheeps at the farm, some of chickens and sheeps were sold. There are Eighty Seven chickens and Eighteen sheeps left now.
# Question: How many chickens were sold?
# Hint:
One Hundred Eight = 108
Twenty = 20
Eighty Seven = 87
Eighteen = 18
# Python:
num_chicken_before = 108
num_chicken_now = 87
ans = num_chicken_before - num_chicken_now

# Passage: Katty scored Two goals on monday, Eight goals on tuesday and Nine goals on wednesday.
# Question: How many did Katty score on monday and wednesday?
# Hint:
Two = 2
Eight = 8
Nine = 9
# Python:
num_goals_on_monday = 2
num_goals_on_wednesday = 9
ans = num_goals_on_monday + num_goals_on_wednesday

# Passage: There are Five girls and Four boys in the Masquerade, Twelve more girls and Seven more boys joined. 
# Question: How many more girls than boys are in the Masquerade?
# Hint:
Five = 5
Four = 4
Twelve = 12
Seven = 7
# Python:
num_girls_before = 5
num_girls_joined = 12
num_boys_before = 4
num_boys_joined = 7
total_girls = num_girls_before + num_girls_joined
total_boys = num_boys_before + num_boys_joined
ans = total_girls - total_boys

# Passage: Joseph and Getty went to buy ice creams, they together bought Thirty Six ice creams. On the way back, Joseph ate Twelve of the ice creams, and he has Two ice creams left now. 
# Question: How much ice cream did Getty purchase?
# Hint:
Thirty Six = 36
Twelve = 12
Two = 2
# Python:
num_ice_creams_bought_by_joseph = 2 + 12
total_ice_creams = 36
ans = total_ice_creams - num_ice_creams_bought_by_joseph

# Passage: {passage}
# Question : {question}
# Hint:
{hint}# Python:
"""

PoT_num_eng_template = """
Read the following passages to answer questions with Python code, store the result as a 'ans' variable:

# Passage: James bought 93 red and 10 blue stickers, he used 31 red sticker on his fridge and 7 blue stickers on his laptop.
# Question: How many red stickers does James have?
# Hint:
93 = Ninety Three
10 = Ten
31 = Thirty One
7 = Seven
# Python:
original_red_stickers = 93
used_red_stickers = 31
ans = original_red_stickers - used_red_stickers

# Passage: Allen went to supermarket to buy eggs, each egg costs 80 dollars, if the discount is 29 dollars.
# Question: How much do you have to pay to buy for each egg?
# Hint:
80 = Eighty
29 = Twenty Nine
# Python:
original_egg_price_in_dollars = 80
discount_dollars = 29
ans = original_egg_price_in_dollars - discount_dollars

# Passage: Dianna collects both cases and books. He bought 22 cases and 5 books from the store. Now he has 57 cases and 25 books.
# Question: How many books did danny have at first?
# Hint:
22 = Twenty Two
5 = Five
57 = Fifty Seven
25 = Twenty Five
# Python:
num_books_bought_at_store = 5
num_books_now = 25
ans = num_books_now - num_books_bought_at_store

# Passage: There were 108 chickens and 20 sheeps at the farm, some of chickens and sheeps were sold. There are 87 chickens and 18 sheeps left now.
# Question: How many chickens were sold?
# Hint:
108 = One Hundred Eight
20 = Twenty
87 = Eighty Seven
18 = Eighteen
# Python:
num_chicken_before = 108
num_chicken_now = 87
ans = num_chicken_before - num_chicken_now

# Passage: Katty scored 2 goals on monday, 8 goals on tuesday and 9 goals on wednesday.
# Question: How many did Katty score on monday and wednesday?
# Hint:
2 = Two
8 = Eight
9 = Nine
# Python:
num_goals_on_monday = 2
num_goals_on_wednesday = 9
ans = num_goals_on_monday + num_goals_on_wednesday

# Passage: There are 5 girls and 4 boys in the Masquerade, 12 more girls and 7 more boys joined. 
# Question: How many more girls than boys are in the Masquerade?
# Hint:
5 = Five
4 = Four
12 = Twelve
7 = Seven
# Python:
num_girls_before = 5
num_girls_joined = 12
num_boys_before = 4
num_boys_joined = 7
total_girls = num_girls_before + num_girls_joined
total_boys = num_boys_before + num_boys_joined
ans = total_girls - total_boys

# Passage: Joseph and Getty went to buy ice creams, they together bought 36 ice creams. On the way back, Joseph ate 12 of the ice creasm, and he has 2 ice creams left now. 
# Question: How much ice cream did Getty purchase?
# Hint:
36 = Thirty Six
12 = Twelve
2 = Two
# Python:
num_ice_creams_bought_by_joseph = 2 + 12
total_ice_creams = 36
ans = total_ice_creams - num_ice_creams_bought_by_joseph

# Passage: {passage}
# Question : {question}
# Hint:
{hint}# Python:
"""


PoT_org_template = """
Read the following passages to answer questions with Python code, store the result as a 'ans' variable:

# Passage: James bought 93 red and 10 blue stickers, he used 31 red sticker on his fridge and 7 blue stickers on his laptop.
# Question: How many red stickers does James have?
original_red_stickers = 93
used_red_stickers = 31
ans = original_red_stickers - used_red_stickers

# Passage: Allen went to supermarket to buy eggs, each egg costs 80 dollars, if the discount is 29 dollars.
# Question: How much do you have to pay to buy for each egg?
original_egg_price_in_dollars = 80
discount_dollars = 29
ans = original_egg_price_in_dollars - discount_dollars

# Passage: Dianna collects both cases and books. He bought 22 cases and 5 books from the store. Now he has 57 cases and 25 books.
# Question: How many books did danny have at first?
num_books_bought_at_store = 5
num_books_now = 25
ans = num_books_now - num_books_bought_at_store

# Passage: There were 108 chickens and 20 sheeps at the farm, some of chickens and sheeps were sold. There are 87 chickens and 18 sheeps left now.
# Question: How many chickens were sold?
num_chicken_before = 108
num_chicken_now = 87
ans = num_chicken_before - num_chicken_now

# Passage: Katty scored 2 goals on monday, 8 goals on tuesday and 9 goals on wednesday.
# Question: How many did Katty score on monday and wednesday?
num_goals_on_monday = 2
num_goals_on_wednesday = 9
ans = num_goals_on_monday + num_goals_on_wednesday

# Passage: There are 5 girls and 4 boys in the Masquerade, 12 more girls and 7 more boys joined. 
# Question: How many more girls than boys are in the Masquerade?
num_girls_before = 5
num_girls_joined = 12
num_boys_before = 4
num_boys_joined = 7
total_girls = num_girls_before + num_girls_joined
total_boys = num_boys_before + num_boys_joined
ans = total_girls - total_boys

# Passage: Joseph and Getty went to buy ice creams, they together bought 36 ice creams. On the way back, Joseph ate 12 of the ice creasm, and he has 2 ice creams left now. 
# Question: How much ice cream did Getty purchase?
num_ice_creams_bought_by_joseph = 2 + 12
total_ice_creams = 36
ans = total_ice_creams - num_ice_creams_bought_by_joseph

# Passage: {passage}
# Question : {question}
"""

CoT_org_template = """Q: There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?
A: There are 15 trees originally. Then there were 21 trees after some more were planted. So there must have been 21 - 15 = 6. The answer is 6.
Q: If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?
A: There are originally 3 cars. 2 more cars arrive. 3 + 2 = 5. The answer is 5.
Q: Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?
A: Originally, Leah had 32 chocolates. Her sister had 42. So in total they had 32 + 42 = 74. After eating 35, they had 74 - 35 = 39. The answer is 39.
Q: Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops. How many lollipops did Jason give to Denny?
A: Jason started with 20 lollipops. Then he had 12 after giving some to Denny. So he gave Denny 20 - 12 = 8. The answer is 8.
Q: Shawn has five toys. For Christmas, he got two toys each from his mom and dad. How many toys does he have now?
A: Shawn started with 5 toys. If he got 2 toys each from his mom and dad, then that is 4 more toys. 5 + 4 = 9. The answer is 9.
Q: There were nine computers in the server room. Five more computers were installed each day, from monday to thursday. How many computers are now in the server room?
A: There were originally 9 computers. For each of 4 days, 5 more computers were added. So 5 * 4 = 20 computers were added. 9 + 20 is 29. The answer is 29.
Q: Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On wednesday, he lost 2 more. How many golf balls did he have at the end of wednesday?
A: Michael started with 58 golf balls. After losing 23 on tuesday, he had 58 - 23 = 35. After losing 2 more, he had 35 - 2 = 33 golf balls. The answer is 33.
Q: Olivia has $23. She bought five bagels for $3 each. How much money does she have left?
A: Olivia had 23 dollars. 5 bagels for 3 dollars each will be 5 x 3 = 15 dollars. So she has 23 - 15 dollars left. 23 - 15 is 8. The answer is 8.
Q: {question}
"""

CoT_d2e_template = '''Q: There are Fifteen trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be Twenty One trees. How many trees did the grove workers plant today?
A: There are 15 trees originally. Then there were 21 trees after some more were planted. So there must have been 21 - 15 = 6. The answer is 6.
Q: If there are Three cars in the parking lot and Two more cars arrive, how many cars are in the parking lot?
A: There are originally 3 cars. 2 more cars arrive. 3 + 2 = 5. The answer is 5.
Q: Leah had Thirty Two chocolates and her sister had Forty Two. If they ate Thirty Five, how many pieces do they have left in total?
A: Originally, Leah had 32 chocolates. Her sister had 42. So in total they had 32 + 42 = 74. After eating 35, they had 74 - 35 = 39. The answer is 39.
Q: Jason had Twenty lollipops. He gave Denny some lollipops. Now Jason has Twelve lollipops. How many lollipops did Jason give to Denny?
A: Jason started with 20 lollipops. Then he had 12 after giving some to Denny. So he gave Denny 20 - 12 = 8. The answer is 8.
Q: Shawn has five toys. For Christmas, he got two toys each from his mom and dad. How many toys does he have now?
A: Shawn started with 5 toys. If he got 2 toys each from his mom and dad, then that is 4 more toys. 5 + 4 = 9. The answer is 9.
Q: There were nine computers in the server room. Five more computers were installed each day, from monday to thursday. How many computers are now in the server room?
A: There were originally 9 computers. For each of 4 days, 5 more computers were added. So 5 * 4 = 20 computers were added. 9 + 20 is 29. The answer is 29.
Q: Michael had Fifty Eight golf balls. On tuesday, he lost Twenty Three golf balls. On wednesday, he lost Two more. How many golf balls did he have at the end of wednesday?
A: Michael started with 58 golf balls. After losing 23 on tuesday, he had 58 - 23 = 35. After losing Two more, he had 35 - 2 = 33 golf balls. The answer is 33.
Q: Olivia has $Twenty Three. She bought five bagels for $Three each. How much money does she have left?
A: Olivia had 23 dollars. 5 bagels for 3 dollars each will be 5 x 3 = 15 dollars. So she has 23 - 15 dollars left. 23 - 15 is 8. The answer is 8.
Q: {question}
'''

professor_comment_template = """
Read the following passages to answer questions with Python code, store the result as a 'ans' variable:

# Passage: James bought 93 red and 10 blue stickers, he used 31 red sticker on his fridge and 7 blue stickers on his laptop.
# Question: How many red stickers does James have?
# Hint: 
93 = 9 * 10^1 + 3 * 10^0
# Python:
original_red_stickers = 93
used_red_stickers = 31
ans = original_red_stickers - used_red_stickers

# Passage: Allen went to supermarket to buy eggs, each egg costs 80 dollars, if the discount is 29 dollars.
# Question: How much do you have to pay to buy for each egg?
# Hint: 
80 = 8 * 10^1 + 0 * 10^0
29 = 2 * 10^1 + 9 * 10^0
# Python:
original_egg_price_in_dollars = 80
discount_dollars = 29
ans = original_egg_price_in_dollars - discount_dollars

# Passage: Dianna collects both cases and books. He bought 22 cases and 5 books from the store. Now he has 57 cases and 25 books.
# Question: How many books did danny have at first?
# Hint:
22 = 2 * 10^1 + 2 * 10^0
5 = 5 * 10^0
57 = 5 * 10^1 + 7 * 10^0
25 = 2 * 10^1 + 5 * 10^0
# Python:
num_books_bought_at_store = 5
num_books_now = 25
ans = num_books_now - num_books_bought_at_store

# Passage: There were 108 chickens and 20 sheeps at the farm, some of chickens and sheeps were sold. There are 87 chickens and 18 sheeps left now.
# Question: How many chickens were sold?
# Hint:
108 = 1 * 10^2 + 0 * 10^1 + 8 * 10^0
20 = 2 * 10^1 + 0 * 10^0
87 = 8 * 10^1 + 7 * 10^0
18 = 1 * 10^1 + 8 * 10^0
# Python:
num_chicken_before = 108
num_chicken_now = 87
ans = num_chicken_before - num_chicken_now

# Passage: Katty scored 2 goals on monday, 8 goals on tuesday and 9 goals on wednesday.
# Question: How many did Katty score on monday and wednesday?
# Hint:
2 = 2 * 10^0
8 = 8 * 10^0
9 = 9 * 10^0
# Python:
num_goals_on_monday = 2
num_goals_on_wednesday = 9
ans = num_goals_on_monday + num_goals_on_wednesday

# Passage: There are 5 girls and 4 boys in the Masquerade, 12 more girls and 7 more boys joined. 
# Question: How many more girls than boys are in the Masquerade?
# Hint:
5 = 5 * 10^0
4 = 4 * 10^0
12 = 1 * 10^1 + 2* 10^0
7 = 7 * 10^0
# Python:
num_girls_before = 5
num_girls_joined = 12
num_boys_before = 4
num_boys_joined = 7
total_girls = num_girls_before + num_girls_joined
total_boys = num_boys_before + num_boys_joined
ans = total_girls - total_boys

# Passage: Joseph and Getty went to buy ice creams, they together bought 36 ice creams. On the way back, Joseph ate 12 of the ice creasm, and he has 2 ice creams left now. 
# Question: How much ice cream did Getty purchase?
# Hint:
36 = 3 * 10^1 + 6 * 10^0
12 = 1 * 10^1 + 2 * 10^0
2 = 2 * 10^0
# Python:
num_ice_creams_bought_by_joseph = 2 + 12
total_ice_creams = 36
ans = total_ice_creams - num_ice_creams_bought_by_joseph

# Passage: {passage}
# Question : {question}
# Hint: {hint}
# Python:
"""

def PoT(llm, problem, inplace=False):

    if not inplace:
        prompt = PromptTemplate(template=PoT_template, input_variables=["passage", "question", "cache"])
    else:
        prompt = PromptTemplate(template=PoT_template_inplace, input_variables=["passage", "question", "cache"])

    llm_chain = LLMChain(prompt=prompt, llm=llm)

    passage, question, cache = convert_and_caching_prob(problem, inplace=inplace)

    print("problem:")
    print(prompt.format(passage=passage, question=question, cache=cache))

    output = llm_chain.run(passage=passage, question=question, cache=cache)

    print("output:")
    print(cache + output)

    return cache, output


def PoT_original(llm, problem):

    prompt = PromptTemplate(template=PoT_org_template, input_variables=["passage", "question"])

    llm_chain = LLMChain(prompt=prompt, llm=llm)

    print("problem:")
    print(prompt.format(passage=problem.passage, question=problem.question))

    output = llm_chain.run(passage=problem.passage, question=problem.question)

    print("output:")
    print(output)

    return "", output


def CoT_original(llm, problem):

    prompt = PromptTemplate(template=CoT_org_template, input_variables=["question"])

    llm_chain = LLMChain(prompt=prompt, llm=llm)

    print("problem:")
    print(prompt.format(question=problem.passage + ' ' + problem.question))

    output = llm_chain.run(problem.passage + ' ' + problem.question)

    print("output:")
    print(output)

    return "", output


def CP_rendezvous(llm, problem, cot_filepath, i):

    with open(cot_filepath, 'r') as f:
        results = json.load(f)

        cot_output = results["Results"][i]["openai"]

    prompt = PromptTemplate(template=PoT_org_template, input_variables=["passage", "question"])

    llm_chain = LLMChain(prompt=prompt, llm=llm)

    question = problem.question + '(hint: ' + cot_output + ')'

    print("Question:")
    print(prompt.format(passage=problem.passage, question=question))

    output = llm_chain.run(passage=problem.passage, question=question)

    print("output:")
    print(output)

    return "", output


def digit2alph(llm, problem):

    prompt = PromptTemplate(template=PoT_d2a_template, input_variables=["passage", "question"])

    llm_chain = LLMChain(prompt=prompt, llm=llm)

    passage, question = convert_digit2alph(problem)

    print("Question:")
    print(prompt.format(passage=passage, question=question))

    output = llm_chain.run(passage=passage, question=question)

    print("output:")
    print(output)

    return "", output


def digit2alph_CoT(llm, problem):

    prompt = PromptTemplate(template=CoT_d2e_template, input_variables=["question"])

    llm_chain = LLMChain(prompt=prompt, llm=llm)

    passage, question = convert_digit2alph(problem)

    print("question:")
    print(prompt.format(question=passage + ' ' + question))

    output = llm_chain.run(question=passage + ' ' + question)

    print("output:")
    print(output)

    return "", output


def CP_rendezvous_d2a(llm, problem, cot_filepath, i):

    with open(cot_filepath, 'r') as f:
        results = json.load(f)

        cot_output = results["Results"][i]["openai"]

    prompt = PromptTemplate(template=PoT_d2a_template, input_variables=["passage", "question"])

    llm_chain = LLMChain(prompt=prompt, llm=llm)

    question = problem.question + '(hint: ' + cot_output + ')'

    print("Question:")
    print(prompt.format(passage=problem.passage, question=question))

    output = llm_chain.run(passage=problem.passage, question=question)

    print("output:")
    print(output)

    return "", output


permute_prompt = '''
List the following numbers in increasing order
{numbers}
'''


def _compare_single_token(llm, n=3):

    prompt = PromptTemplate(template=permute_prompt, input_variables=["numbers"])

    llm_chain = LLMChain(prompt=prompt, llm=llm)

    numbers = ""
    for i in range(n):
        num = random.randrange(100, 521)
        numbers += str(num) + '\n'
    print(numbers)

    print("Question:")
    print(prompt.format(numbers=numbers))

    output = llm_chain.run(numbers=numbers)

    print("output:")
    print(output)

    nums = re.findall(r"\d+\.\d+|\d+", output)
    result = ""
    if len(nums) >= n:
        result = nums[-n]
        for i in range(1-n, 0):
            result += " " + nums[i]

    return result


def _compare_permutation(llm, sources):

    prompt = PromptTemplate(template=permute_prompt, input_variables=["numbers"])

    llm_chain = LLMChain(prompt=prompt, llm=llm)

    nums = []
    for elems in permutations(sources):
        num = ''
        for elem in elems:
            num += elem
        nums.append(num)

    random.shuffle(nums)
    nums = nums[:6]
    n = 6

    numbers = ""
    for num in nums:
        numbers += num + '\n'

    print(numbers)

    print("Question:")
    print(prompt.format(numbers=numbers))

    output = llm_chain.run(numbers=numbers)

    print("output:")
    print(output)

    nums = re.findall(r"\d+\.\d+|\d+", output)
    result = ""
    if len(nums) >= n:
        result = nums[-n]
        for i in range(1-n, 0):
            result += " " + nums[i]

    return result

def make_hint(num : str):
    result = []

    for i in range(len(num)):
        result.append(num[i] + " * 10^" + str(len(num) - i - 1))

    return " + ".join(result)
def get_hint(problem: str):
    result = ""
    nums = re.findall(r"\d+\.\d+|\d+", problem)

    for num in nums:
        result += "\n" + num + " = " + make_hint(num)

    return result


def professor_comment(llm, problem):

    prompt = PromptTemplate(template=professor_comment_template, input_variables=["passage", "question", "hint"])

    llm_chain = LLMChain(prompt=prompt, llm=llm)

    print("question:")
    print(prompt.format(passage=problem.passage, question=problem.question, hint=get_hint(problem.passage + ' ' + problem.question)))

    output = llm_chain.run(passage=problem.passage, question=problem.question, hint=get_hint(problem.passage + ' ' + problem.question))

    print("output:")
    print(output)

    return "", output


def d2e_num_eng_matching(llm, problem):

    prompt = PromptTemplate(template=d2e_num_eng_template, input_variables=["passage", "question", "hint"])

    llm_chain = LLMChain(prompt=prompt, llm=llm)

    nums = re.findall(r"\d+\.\d+|\d+", problem.passage + ' ' + problem.question)

    hint = ""
    for n in nums:
        hint += float_to_words(n) + ' = ' + n + '\n'

    passage, question = convert_digit2alph(problem)

    print("question:")
    print(prompt.format(passage=passage, question=question, hint=hint))

    output = llm_chain.run(passage=passage, question=question, hint=hint)

    print("output:")
    print(output)

    return "", output


def PoT_num_eng_matching(llm, problem):
    prompt = PromptTemplate(template=PoT_num_eng_template, input_variables=["passage", "question", "hint"])

    llm_chain = LLMChain(prompt=prompt, llm=llm)

    nums = re.findall(r"\d+\.\d+|\d+", problem.passage + ' ' + problem.question)

    hint = ""
    for n in nums:
        hint += n + ' = ' + float_to_words(n) + '\n'

    print("question:")
    print(prompt.format(passage=problem.passage, question=problem.question, hint=hint))

    output = llm_chain.run(passage=problem.passage, question=problem.question, hint=hint)

    print("output:")
    print(output)

    return "", output
