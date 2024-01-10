from langchain.agents import load_tools
from langchain.agents import initialize_agent
from langchain.llms import OpenAI
from typing import Generic, TypeVar

class P():
    x = 5

    @classmethod
    def print_c(cls):
        print(cls)

    def print_x(self):
        print('x', self.x)

class C(P):
    def __init__(self, x):
        self.x = x

P.print_c()
C.print_c()

"""
llm = OpenAI(temperature=0)

text = "What would be a good company name a company that makes colorful socks?"

tools = load_tools(["serpapi", "llm-math"], llm=llm)
agent = initialize_agent(tools, llm, agent="zero-shot-react-description", verbose=True)
agent.run("What are the common adverse events in a hep A trial?")

class P():
    def say_hello():
        print('hello from P')

class C1(P):
    def c1():
        pass

class C2(P):
    pass

class C3():
    pass

#PX = Generic.__class_getitem__(TP)

TP = TypeVar('TP', bound=C1)

class X(Generic[TP]):
    def __init__(self, p: TP):
        self.p = p

    def x1(self, p: TP):
        print(p)
        return p

#print(type(Generic[TP]))
#print("-----")
print(type(X))
print(type(X[int]))

class Y():
    def __class_getitem__(cls, p):
        return Y

#class X(Generic[TP]):
#class X(Generic.__class_getitem__(TP)):
class X(PX):
    def __init__(self, p1: TP):
        print("X type", type(p1))

#class Y(X[C1]):
class Y(X.__class_getitem__(C1)):
    def __init__(self, p1: C1):
        print("Y type", type(p1))

I = X.__class_getitem__(C1)
print(I)

#print(dir(Generic[TP]))
#print(dir(PX))
#print(dir(X))
#print(dir(Generic[TP]))

#if Generic[TP] == Generic.__class_getitem__(TP):
#    print('yes')

T = TypeVar('T')

class A(Generic[T]):
    def __init__(self):
        self.items = []

    def a(self, i: T):
        print(i)

class B():
    pass

def printt(x):
    print(type(x))

class XC1(X[C1]):
    pass

class XC2(X[C2]):
    pass

class XC3(X[C3]):
    pass

class A:
    def __class_getitem__(cls, i):
        pass

def x(a,b,c)
def x(a,b,c,d=10)

x(1,2,3,d=4)


class Hello():
    def __call__(self, msg):
        print('hello, ' + msg)

hello = Hello()
hello("there")

"""
