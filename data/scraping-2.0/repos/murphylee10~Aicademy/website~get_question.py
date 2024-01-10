#openai.Model.list()
#print("\n\nLinked Lists in Computer Science\n\nLinked lists are a fundamental data structure used in computer science. They are a type of linear data structure, meaning they store data in a linear fashion. Unlike arrays, linked lists are dynamic in size and can grow or shrink as needed.\n\nA linked list consists of nodes, which are the individual elements of the list. Each node contains a data element and a pointer to the next node in the list. This makes it possible to traverse the list in a linear fashion.\n\nPictures\n\n[Insert Pictures]\n\nLinked lists are useful in a variety of applications, such as sorting and searching data. They can also be used to implement stacks and queues, which are also important data structures in computer science.\n\nLinked lists are also used in graph algorithms, such as depth-first search and breadth-first search. These algorithms traverse the graph by following the edges of the graph, which are represented by the linked list.\n\nPictures\n\n[Insert Pictures]\n\nLinked lists are an important data structure in computer science, used in a variety of applications. This course has discussed how linked lists are structured, how they are used in sorting and searching data, and how they are used in graph algorithms.\n\nThis course has provided an overview of linked lists in computer science, including their structure, uses, and applications. It has also discussed how linked lists are used in graph algorithms, such as depth-first search and breadth-first search. The course has also included pictures to illustrate the concepts discussed.")
#list1 = [1, 2, 3, 4]
#list2 = [5, 6, 7, 8]

#list3 = list1 + list2

#print(list3)
# user_input = input("hi")
# response = openai.Completion.create(
#   model="text-davinci-003",
#   prompt=f"""We need to make a course which is with the topic {user_input}. For this we need an introduction, 4 main body paragraphs and a concluclusion paragraph. This course should have accurate details and be engaging with clear understandable sentenses. There should be the delimeter "@@" before every paragraph, which should each be seperated from each other with an empty line.""",
#   max_tokens=3700,
#   frequency_penalty=0.0,
#   presence_penalty=0.0
# )
# intro = response["choices"][0]["text"]
# stuff = intro.split("@@")
# # response = openai.Completion.create(
# #   model="text-davinci-003",
# #   prompt=f"Generate 15 multiple choice problems and solutions on the subject \"{subject}\" which could accomidate pictures in between. for example, a python course could have which of these would print hello world 100 times: a) print('hello world') b) sadfgsfg c)sdrfsdfg d)for i in range(100): print('hello world') and the answer would be d. you should also explain after every one of the 15 questions how you got the answer like for this question an explination will be it is d because this look prints hello world 100 times. first we start a for loop. in order for the for loop to iterate 100 times, we first choose an iterating variable i, and then we specified the range of i, 100. Then, we printed hello world for every one of the for loops iterations, resulting in the program printing hello world 100 times.",
# #   temperature=0.5,
# #   max_tokens=3700,
# #   frequency_penalty=0.0,
# #   presence_penalty=0.0
# # )
# # problems = response["choices"][0]["text"]
# # response = openai.Completion.create(
# #   model="text-davinci-003",
# #   prompt=f"Generate a summary on the subject \"{subject}\" which could accomidate pictures in between",
# #   temperature=0.5,
# #   max_tokens=3700,
# #   frequency_penalty=0.0,
# #   presence_penalty=0.0
# # )
# #summary = response["choices"][0]["text"]
# #print(stuff)
# try:
#   stuff.remove('\n\n')
# except:
#   pass
# qs = []
# ans = []
# #i = 0
# for thing in stuff:
#   qs.append(openai.Completion.create(
#     model="text-davinci-003",
#     prompt=f"give a multiple choice or true/false question on this paragraph without giving the answer: {thing}",
#     temperature=0.5,
#     max_tokens=3700,
#     frequency_penalty=0.0,
#     presence_penalty=0.0
#   )["choices"][0]["text"])
#   ans.append(openai.Completion.create(
#     model="text-davinci-003",
#     prompt=f"give an answer to the question, followed by a comprehensive detailed explanation: {qs[-1]}",
#     temperature=0.5,
#     max_tokens=3700,
#     frequency_penalty=0.0,
#     presence_penalty=0.0
#   )["choices"][0]["text"])
#   #i += 1
# print(stuff)
# print("\n")
# print(qs)
# print("\n")
# print(ans)
  
def get_question(arr):
  import openai
  openai.api_key = "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"
  import random
  type_of_question = ["true or false", "multiple choice with four difference choices"]
  question_index = random.randint(0, 1)
  package = {}
  package["reference"] = random.randint(1,len(arr) - 2)
  paragraph = arr[package["reference"]]
  package["question"] = openai.Completion.create(
    model="text-davinci-003",
    prompt=f"Give a {type_of_question[question_index]} question on this paragraph without giving the answer: {paragraph}",
    temperature=0.5,
    max_tokens=500,
    frequency_penalty=0.0,
    presence_penalty=0.0
  )["choices"][0]["text"]
  question = package["question"]
  package["answer"] = openai.Completion.create(
    model="text-davinci-003",
    prompt=f"Give an answer to the question, followed by a comprehensive detailed explanation: {question}",
    temperature=0.2,
    max_tokens=500,
    frequency_penalty=0.0,
    presence_penalty=0.0
  )["choices"][0]["text"]
  return package

#print(get_question(['Introduction \nLinear Algebra is one of the most important subject in mathematics and is used in a multitude of applications in a large variety of fields. Linear Algebra is a branch of mathematics dealing with groups of linear equations and their representations in vector spaces. It has tangible applications in physics, economics, engineering and many other fields. This course will provide an introduction to the foundational concepts and techniques of the subject and demonstrate how they can be applied to solve real world problems. \n\n', 'Main body 1 \nThe first concept to understand in Linear Algebra is that of a vector. A vector is an object that has both a magnitude (or length) and a direction associated with it. Vectors can be represented by arrows, with the length of the arrow representing the magnitude and the direction indicated by the direction of the arrow. In Linear Algebra we will generally use the standard mathematical notation for vectors and regard them as elements of a vector space. \n \n', 'Main body 2 \nThe second important concept in Linear Algebra is that of a Matrix. A matrix is a table of numbers that can be used to represent a group of linear equations. A matrix can be thought of as a collection of vectors and can be used to transform one vector into another. Matrices can be used to solve systems of linear equations and can also be used to represent linear transformations. \n\n', 'Main body 3 \nThe third topic covered in this course is that of linear independence. In Linear Algebra, two vectors are said to be linearly independent if they are not multiples of each other. Linear independence is a key concept in solving systems of linear equations and is a basic property of the vector spaces we will encounter. \n\n', 'Main body 4 \nThe fourth and final concept we will cover is inner product spaces. An inner product space is a vector space where the vectors can be "multiplied" together. This multiplication operation is known as an inner product and is used to calculate certain properties of the vector. Inner product spaces are also used to compute certain distances between vectors. \n\n', 'Conclusion \nIn conclusion, a good understanding of the fundamental concepts and techniques of linear algebra are necessary for many applications. In this course, we have provided a comprehensive introduction to the foundational concepts and techniques of the subject and demonstrated how they can be applied to solve real world problems. We hope you now have the knowledge and skills necessary to begin your journey into the exciting world of linear algebra.']))
