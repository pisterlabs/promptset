#print('hello world!')
# a=400

# def test2():
#     a=300
#     print('----test2----修改前---a=%d,id=%d'%(a,id(a)))
#     if 1>2:
#         b=100
#         print(a)
#     else:
#         b='a'
#     print(b)


# def test1():
#     global a
#     # a=100
#     print('----test1----修改前---a=%d,id=%d'%(a,id(a)))
#     a=200
#     # print('----test1----修改前---a=%d,id=%d'%(a,id(a)))
#     # test2()
# #print(test1)
# a=400
# print('----test1----修改前---a=%d,id=%d'%(a,id(a)))




# test1()
#递归阶乘
# def calnum(n):
#     # if not n.isdigit():
#     #     return '请输入数字'
#     if n <= 1:
#         return 1
#     return n * calnum(n - 1)

# print(calnum('n'))

#斐波那契数列
# def fib(n):
#     if n==1:
#         return 1
#     if n==2:
#         return 1
#     return fib(n-1)+fib(n-2)
# print (fib(5))


#汉诺塔
# def hanoi(n, a, b, c):
#     global step
    
#     if n == 1:
#         step +=1
#         print(str(step),a, '-->', c)
        
#     else:
#         hanoi(n - 1, a, c, b)
#         step +=1
#         print(str(step),a, '-->', c)
#         hanoi(n - 1, b, a, c)
# # 调用
# step=0
# hanoi(3, 'A', 'B', 'C')


# f=open('test.txt','w')
# f.write('hello world!')
# f.close


# f=open('test.txt','r')
# print(f.read())
# f.close


# import os
# print(os.listdir)

# class Car:
#     def __init__(self,x,y):
#         self.x=x
#         self.y=y
#     def __new__(cls: type[Self]) -> Self:
#         pass
#     def move():
#         print('--------------')
# car=Car(1,2)
# print(id(car))

#斐波那契数列
# def fib(n):   #递归实现           
#     if n==1:
#         return 5
#     if n==2:
#         return 1
#     return fib(n-1)+fib(n-2)
# print (fib(5))

# def fib(n):   #循环实现
#     a,b=0,1
#     for i in range(n):
#         yield b
#         a,b=b,a+b
#     return a
# print (list(fib(5)))

#汉诺塔
# def hanoi(n, a, b, c):
#     if n == 1:
#         print(a, '-->', c)
#     else:
#         hanoi(n - 1, a, c, b)
#         print(a, '-->', c)
#         hanoi(n - 1, b, a, c)
# hanoi(3, 'A', 'B', 'C')

#八皇后问题
# def eight_queen(n):
#     def check(queens,row,col):
#         for i in range(row):
#             if queens[i]==col or abs(row-i)==abs(queens[i]-col):
#                 return False
#         return True
#     def dfs(queens,row):
#         if row==n:
#             print(queens)
#         else:
#             for col in range(n):
#                 if check(queens,row,col):
#                     queens[row]=col
#                     dfs(queens,row+1)
#                     queens[row]=-1
#     queens=[-1 for i in range(n)]
#     dfs(queens,0)
# eight_queen(8)
  

#迭代器
# class MyRange:
#     def __init__(self,start,end):
#         self.start=start
#         self.end=end
#     def __iter__(self):
#         return self
#     def __next__(self):
#         if self.start<self.end:
#             self.start+=1
#             return self.start
#         else:
#             raise StopIteration
# for i in MyRange(1,10):
#     print(i)

#异常处理
# try:
#     print(1/0)
# except Exception as e:
#     print(e)
# finally:
#     print('finally')
# print('end')

# from numpy import *
# from pandas import DataFrame
# df =DataFrame(np.random.randint(0,10,size=(3,4)),columns=['a','b','c','d'])
# df


# a=[1,2,3] 
# b=[4,5,6]
# print(a + b)

# import numpy as np
# a=np.arrange(3)
# print(a)

# a=[1,2,3]
# print(a[1:])


import os
import openai

openai.api_key = os.getenv("OPENAI_API_KEY")

response = openai.Completion.create(
  model="text-davinci-003",
  prompt="I am a highly intelligent question answering bot. If you ask me a question that is rooted in truth, I will give you the answer. If you ask me a question that is nonsense, trickery, or has no clear answer, I will respond with \"Unknown\".\n\nQ: What is human life expectancy in the United States?\nA: Human life expectancy in the United States is 78 years.\n\nQ: Who was president of the United States in 1955?\nA: Dwight D. Eisenhower was president of the United States in 1955.\n\nQ: Which party did he belong to?\nA: He belonged to the Republican Party.\n\nQ: What is the square root of banana?\nA: Unknown\n\nQ: How does a telescope work?\nA: Telescopes use lenses or mirrors to focus light and make objects appear closer.\n\nQ: Where were the 1992 Olympics held?\nA: The 1992 Olympics were held in Barcelona, Spain.\n\nQ: How many squigs are in a bonk?\nA: Unknown\n\nQ: Where is the Valley of Kings?\nA:",
  temperature=0,
  max_tokens=100,
  top_p=1,
  frequency_penalty=0.0,
  presence_penalty=0.0,
  stop=["\n"]
) 