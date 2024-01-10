# Databricks notebook source
"""
Inventory:

Kernel
Semantic (and Native) functions -- you can do a lot with these
BusinessThinking plugin --> SWOTs in ways you could never imagine
DesignThinking plugin ... Here you are
"""

# COMMAND ----------

"""
Get a kernel ready
"""

# COMMAND ----------

import semantic_kernel as sk
from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion, OpenAIChatCompletion
from IPython.display import display, Markdown

kernel = sk.Kernel()

useAzureOpenAI = False

if useAzureOpenAI:
    deployment, api_key, endpoint = sk.azure_openai_settings_from_dot_env()
    kernel.add_text_completion_service("azureopenai", AzureChatCompletion(deployment, endpoint, api_key))
else:
    api_key, org_id = sk.openai_settings_from_dot_env()
    kernel.add_text_completion_service("openai", OpenAIChatCompletion("gpt-3.5-turbo-0301", api_key, org_id))

print("A kernel is now ready.")    

# COMMAND ----------

import json

pluginsDirectory = "./plugins-sk"

strength_questions = ["What unique recipes or ingredients does the pizza shop use?","What are the skills and experience of the staff?","Does the pizza shop have a strong reputation in the local area?","Are there any unique features of the shop or its location that attract customers?", "Does the pizza shop have a strong reputation in the local area?", "Are there any unique features of the shop or its location that attract customers?"]
weakness_questions = ["What are the operational challenges of the pizza shop? (e.g., slow service, high staff turnover)","Are there financial constraints that limit growth or improvements?","Are there any gaps in the product offering?","Are there customer complaints or negative reviews that need to be addressed?"]
opportunities_questions = ["Is there potential for new products or services (e.g., catering, delivery)?","Are there under-served customer segments or market areas?","Can new technologies or systems enhance the business operations?","Are there partnerships or local events that can be leveraged for marketing?"]
threats_questions = ["Who are the major competitors and what are they offering?","Are there potential negative impacts due to changes in the local area (e.g., construction, closure of nearby businesses)?","Are there economic or industry trends that could impact the business negatively (e.g., increased ingredient costs)?","Is there any risk due to changes in regulations or legislation (e.g., health and safety, employment)?"]

strengths = [ "Unique garlic pizza recipe that wins top awards","Owner trained in Sicily","Strong local reputation","Prime location on university campus" ]
weaknesses = [ "High staff turnover","Floods in the area damaged the seating areas that are in need of repair","Absence of popular calzones from menu","Negative reviews from younger demographic for lack of hip ingredients" ]
opportunities = [ "Untapped catering potential","Growing local tech startup community","Unexplored online presence and order capabilities","Upcoming annual food fair" ]
threats = [ "Competition from cheaper pizza businesses nearby","There's nearby street construction that will impact foot traffic","Rising cost of cheese will increase the cost of pizzas","No immediate local regulatory changes but it's election season" ]

customer_comments = """
Customer 1: The seats look really raggedy.
Customer 2: The garlic pizza is the best on this earth.
Customer 3: I've noticed that there's a new server every time I visit, and they're clueless.
Customer 4: Why aren't there calzones?
Customer 5: I love the garlic pizza and can't get it anywhere else.
Customer 6: The garlic pizza is exceptional.
Customer 7: I prefer a calzone's portable nature as compared with pizza.
Customer 8: Why is the pizza so expensive?
Customer 9: There's no way to do online ordering.
Customer 10: Why is the seating so uncomfortable and dirty?
"""

pluginDT = kernel.import_semantic_skill_from_directory(pluginsDirectory, "DesignThinking");
my_result = await kernel.run_async(pluginDT["Empathize"], input_str=customer_comments)

display(Markdown("## ✨ The categorized observations from the 'Empathize' phase of design thinking\n"))

print(json.dumps(json.loads(str(my_result)), indent=2))

# COMMAND ----------

my_result = await kernel.run_async(pluginDT["Empathize"], pluginDT["Define"], input_str = customer_comments)

display(Markdown("## ✨ The categorized observations from the 'Empathize' + 'Define' phases of design thinking\n"+str(my_result)))

# COMMAND ----------

my_result = await kernel.run_async(pluginDT["Empathize"], pluginDT["Define"], pluginDT["Ideate"], pluginDT["PrototypeWithPaper"], input_str=customer_comments)

display(Markdown("## ✨ The categorized observations from the 'Empathize' + 'Define' + 'Ideate' + 'Prototype' + phases of design thinking\n"+str(my_result)))

# COMMAND ----------

sk_prompt = """
A 40-year old man who has just finished his shift at work and comes into the bar. They are in a bad mood.

They are given an experience like:
{{$input}}

Summarize their possible reactions to this experience.
"""
test_function = kernel.create_semantic_function(prompt_template=sk_prompt,
                                                    description="Simulates reaction to an experience.",
                                                    max_tokens=1000,
                                                    temperature=0.1,
                                                    top_p=0.5)
sk_input="""
A simple loyalty card that includes details such as the rewards for each level of loyalty, how to earn points, and how to redeem rewards is given to every person visiting the bar.
"""

test_result = await kernel.run_async(test_function, input_str=sk_input) 

display(Markdown("### ✨ " + str(test_result)))

# COMMAND ----------

"""🔖 Reminder: We haven't explicitly used the 🧲 similarity engine — we'll be doing that next!"""

# COMMAND ----------



# COMMAND ----------

"""
So are you feeling like a chef? 
An AI chef? 
Feels pretty good. 
You've been cooking, you've been organizing, it looks 
pretty good. 
You're plating things and realize, well, what if I just ordered out? 
What if I want a frozen dinner that's not bad? 
Well, right! 
You want to borrow someone else's work through a plugin of some form, extend 
what you're doing, and make you even more amazing. 
So let's reach into the freezer and try out a 
kind of frozen dinner, a design thinking plugin. 
You figure out the SWOT stuff, let's go grab another one, work with it, and 
keep cooking so that you become the best AI chef ever. 
Let's do that. 
So we're gonna tap into the design thinking plugin, get 
excited. 
Design thinking is the five-step process, empathize, define, ideate, 
prototype, test. 
We'll get into it. 
We're going to dig into a plugin to take advantage of this incredible 
plugin that I hate to say this so boastfully, but 
I made it and I was excited when I made it. 
But again, if you don't like it, you can change it because it's in 
the files. 
So design thinking, it's automated, it's pretty amazing. 
It's a plugin and the plugins are important because as you take your 
AI capabilities and and make them into transportable plugins 
packaged so that someone else can use them, you'll be 
able to use them in all kinds of places. 
Have you tried out the chat GPT plugins? 
That's pretty cool. 
And at Microsoft build, you heard the EVP of AI, 
Kevin Scott talk about more ways to use AI plugins across the 
Microsoft universe. 
And just in general, it's a great way to share recipes. 
Okay, let's do a quick inventory. 
Inventory. 
One, you know how to make a kernel. 
Fantastic. 
Two, you can make semantic and native functions. 
You can do a lot with these. 
And number three, you're able to use the business thinking, log 
in, and you're able to process SWATs in ways 
you could never imagine. 
Well, I'm not sure if you imagined them before you took this course, but 
I'm kind of bewildered at how amazing this stuff is. 
And now we're going to do design thinking plug-in stuff. 
 
And here you are. 
Just a reminder, here we are. 
Okay. 
So first off, what I want to do every time 
we want to get a kernel ready. 
Let's get a kernel. 
Get a kernel ready. 
I like work down so much. 
Okay, and I'm gonna bring in the boilerplate that you already know. 
Let's get that kernel ready. 
Let's get the message we like. A kernel is now ready. 
Very good. 
And next up, what we're gonna do is we're going 
to do what is important, which is to start from customer feedback and 
do design thinking. 
So let's start back with the customer, shall 
we? 
How do we do that? 
Well, first off, we wanna bring all those slots back into 
view just because they're useful, bringing in the slot questions 
and the responses. 
And what I'm gonna do next is gather a bunch of customer feedback. 
Let's get that, shall we? 
Let's get some customer comments. 
Okay, so this is, again, this is our business 
snapshot as questions and answers. 
This is the customer comments, 10 comments that 
are sometimes positive, sometimes negative, that's 
how business works, right? 
And what I wanna do now is access the design thinking plugin, 
which now you know is a folder somewhere. 
Let's sort of get our bearings, shall we? 
It is a folder somewhere. 
And let me show you where it is. 
Right? So there's a plugins SK, there's design thinking, there's a define 
and empathize SK prompt.txt. 
What do they look like? Go ahead and open your file browser. 
And if you want to look at the empathize prompt, 
it looks something like this. 
It is following our anonymized comments from customers, 
takes an input, and then convert to JSON, 
the list of sentiments. 
These models are very good at calculating computing sentiments. 
So they're good at that. 
That's the Empathize plugin, does some magic. 
And it's gonna take these customer comments. 
It's not gonna take these swats, we're gonna use them later. 
And we're going to run this input with that design thinking plugin. 
 
We're gonna access the plugin directory. 
And then what we're gonna do is we're going to do our favorite 
thing of importing the design thinking plugin. and 
we're going to run the kernel with the empathize function. 
 
And we're gonna take the customer comments over here, all 
over here. 
And then we're going to have it run the empathy analysis 
from this escape prompt file inside this place 
in the file directory. 
And note that if I change this and save it, 
it changes the prompt. 
Do it yourself. 
And you should do it. Let me show you it 
running so we can look at the food and pick 
it out around on our dish. 
So what did it do? 
It categorized different types of complaints 
about seat condition, praise for the garlic pizza. 
Someone likes it. 
Frustration because always a new person is serving them, doesn't 
know the layout of the restaurant. 
It's like a neutral response, like why aren't there calzones? 
And lastly, there's no online ordering. 
The pizza shop owner hasn't done any digital transformation yet. 
So what do we have here? 
We have a design thinking, instant empathize, magical AI, 
generative moment that took in these comments and 
it generated a sentiment map. 
So now that we have that, you want to remember 
what is design thinking. 
And design thinking is a simple set of steps. 
There are five steps. 
Most people argue that there's no steps in design thinking, that 
it's a set of activities you can sort of do in any order, really. 
But we did empathize, and we used large language model 
AI to summarize the feedback. 
Now once you have this feedback, you can then convert it 
into input to defining the problem. 
You know, you can never solve a problem 
unless you understand it well, as you define it. 
So let's rerun the plugin with the original information. 
It's calculating that. Okay. 
These are the responses. 
And now I'm going to send it into the define plugin. 
Here we go. 
Okay. 
So, okay. 
My result equals, wait, I can't help but love typing 
because that way I can feel it with you. 
So just bear with me. 
I'm going to use the empathize plugin. 
I'm going to feed that output into plugin define, okay? 
 
And then I am going to give as an input string, I'm 
going to give the feedback, customer comments, okay? 
And then I am going to output them. 
Let's get a fancy plated statement there. 
So what's going to happen is now I'm not just calling the empathize component of 
design thinking. 
I'm calling empathize and define it's going to do this and chain them together the 
way the kernel processes things sequentially and what 
happens is. 
Let's run that. 
It's going to take the empathy from the customer feedback it's 
going to feed it into define which is going 
to hypothesize what kind of problems exist so. 
So first off, great garlic pizza, okay, and well, 
great chef probably, and great ingredients, 
high turnover rate, insufficient training, 
absence of calzones, well, maybe we need to 
sort of rethink what we serve the customers. 
And so this example of taking design thinking to number 
one, take the feedback, and then define the problem. 
But design thinking is actually about innovation, 
and innovation is about making new kinds of ideas. 
And so you might take this knowledge you gathered and push it 
into an ideation engine to ask questions about what you could make 
as a solution. 
And then you want to prototype it and 
you also want to test it with real customers. 
This is empathize, define, ideate, prototype, test, 
five different components of design thinking. 
Well, you can do it with the design thinking plugin as well. 
Well, let's show you it running with four phases, all 
chained together. 
We're gonna empathize. 
We're going to define the problem. 
We're gonna ideate solutions and we're going to suggest prototyped, you know, 
some of these people paper prototypes of things. 
We're gonna prototype things out of paper as a next step. 
We're gonna stream this all in into a kernel. 
And what happens is something, I think once you do this kind of work, 
this kind of business strategy work within large language models, I don't 
know about you, but I am bewildered. 
And I'm also excited because a lot of bad ideas can get thrown out 
quickly. 
You know this word hallucination sounds pretty bad, 
but it's actually what humans do. 
They make mistakes. 
So when you're playing with business thinking, design 
thinking, these two plugins, you'll discover that, hey, it actually 
gets it wrong sometimes because often these kinds 
of questions don't have a single correct answer anyways. 
 
So this is what's suggested. 
So to improve seating comfort and cleanliness, 
create a paper prototype of the restaurant seating area. 
Well, that's how to use paper, right? 
Offer discounts, make paper coupons, train servers, 
make a simple training manual, cheat sheet out of paper, 
make a loyalty card with a punch cards. 
This is like 10 ideas that I think if you were to 
go out and pay for them, you could pay for them, 
but here you got them essentially close to free. 
I mean, tokens do cost money, but it's kind of amazing. 
Now you might say, hold it, John, you're getting off the hook 
because you didn't let me test the idea. 
Well, it turns out that more and more services 
are emerging where you could basically give a 
large language model a 
role like a 40 year old person with a certain situation and 
suggest how they would respond to this prototype. 
All right, let's do some of that, shall we? 
So let's make a prompt. 
We make a prompt to test the output. 
Let's say I wanna be a 40 year old man who 
has just finished his shift at work and comes into 
the bar. 
They are in a bad mood. 
Right. 
Close that. 
And they are given an experience like input. 
We're gonna give the experience and then summarize their possible 
reactions to this experience. 
Okay, so now we're gonna call this a test function. 
We're gonna do this inline just for convenience here. 
We're gonna add the configuration information and let's take 
one of the examples above. 
Let's choose one of these things here. 
Let's say someone walks in. 
And they find a simple loyalty. 
So this is one thing we could do. 
Let's pretend we've made a loyalty program. 
So a simple loyalty card that includes details, 
other importance, memoirs, is given to every 
person visiting the bar. 
So that's a potential paper prototype. 
Let's now test this. 
Let's run this prompt through the kernel and let's print 
out the results. 
So we're basically taking a prototype idea and we're going 
to test it on a synthetic person basically. 
And of course, there's like a sort of a special 
message from OpenAI that I can't predict it, but I 
did ask in the prompt, please go ahead. 
So however, some possible reactions could be, 
I might feel indifferent because I don't care about your loyalty card, or 
I may feel annoyed because I just go into the bar, or I might be 
interested because I might wanna be someone who takes advantage of being 
a regular customer. 
So these are really useful responses that you 
can imagine applying to any of the prototyped 
ideas and simple example of how to use 
this design thinking plugin, 
it's different functions, and just really helping to 
test ideas around the small business owner, pizza 
shop owner, construction company, you name it. 
So let's remember what's happened so far in this lesson, which is, 
I want to keep saying this over and over, which is 
there's something we haven't done yet. 
What is it? 
Well, we're showing off the completion engine. 
The completion engine is generating things. They're 
pretty cool. 
We haven't done any semantic similarity. 
We're going to do that next and just wait for how that feels. 
That's a different way to do things. 
Let's go into it. 

"""

# COMMAND ----------


