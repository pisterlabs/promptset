# Databricks notebook source
# MAGIC %md
# MAGIC 🧑‍🍳 L6 - A kitchen that responds to your “I’m hungry” is more than feasible

# COMMAND ----------

# MAGIC %md
# MAGIC Inventory:
# MAGIC
# MAGIC 1. Kernel
# MAGIC 2. Semantic (and Native) functions -- you can do a lot with these
# MAGIC 3. BusinessThinking plugin --> SWOTs in ways you could never imagine
# MAGIC 4. DesignThinking plugin --> you did that. Congrats
# MAGIC 5. Use the similarity engine to your heart's content 🧲
# MAGIC 6. THE BIG ONE!!!!!

# COMMAND ----------

# MAGIC %md
# MAGIC # 🔥 Let's make a kernel one more time!

# COMMAND ----------

import semantic_kernel as sk
from semantic_kernel.connectors.ai.open_ai import OpenAIChatCompletion, OpenAITextEmbedding
from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion, AzureTextEmbedding
from IPython.display import display, Markdown

kernel = sk.Kernel()

useAzureOpenAI = False

if useAzureOpenAI:
    deployment, api_key, endpoint = sk.azure_openai_settings_from_dot_env()
    kernel.add_text_completion_service("azureopenaicompletion", AzureChatCompletion(deployment, endpoint, api_key))
    kernel.add_text_embedding_generation_service("azureopenaiembedding", AzureTextEmbedding("text-embedding-ada-002", api_key, endpoint))
else:
    api_key, org_id = sk.openai_settings_from_dot_env()
    kernel.add_text_completion_service("openaicompletion", OpenAIChatCompletion("gpt-3.5-turbo-0301", api_key, org_id))
    kernel.add_text_embedding_generation_service("openaiembedding", OpenAITextEmbedding("text-embedding-ada-002", api_key, org_id))
print("I did it boss!")

# COMMAND ----------

# MAGIC %md
# MAGIC We want to have a vat of plugins ... and then find the right plugin to fit the goal ...
# MAGIC
# MAGIC **Note**: You can find more about the predefined plugins used below [here](https://learn.microsoft.com/en-us/semantic-kernel/ai-orchestration/out-of-the-box-plugins?tabs=Csharp).
# MAGIC

# COMMAND ----------

from semantic_kernel.planning import ActionPlanner

planner = ActionPlanner(kernel)

from semantic_kernel.core_skills import FileIOSkill, MathSkill, TextSkill, TimeSkill
kernel.import_skill(MathSkill(), "math")
kernel.import_skill(FileIOSkill(), "fileIO")
kernel.import_skill(TimeSkill(), "time")
kernel.import_skill(TextSkill(), "text")

print("Adding the tools for the kernel to do math, to read/write files, to tell the time, and to play with text.")

# COMMAND ----------

ask = "What is the sum of 110 and 990?"

print(f"🧲 Finding the most similar function available to get that done...")
plan = await planner.create_plan_async(goal=ask)
print(f"🧲 The best single function to use is `{plan._skill_name}.{plan._function.name}`")


# COMMAND ----------

ask = "What is today?"
print(f"🧲 Finding the most similar function available to get that done...")
plan = await planner.create_plan_async(goal=ask)
print(f"🧲 The best single function to use is `{plan._skill_name}.{plan._function.name}`")



# COMMAND ----------

ask = "How do I write the word 'text' to a file?"
print(f"🧲 Finding the most similar function available to get that done...")
plan = await planner.create_plan_async(goal=ask)
print(f"🧲 The best single function to use is `{plan._skill_name}.{plan._function.name}`")



# COMMAND ----------

# MAGIC %md
# MAGIC Note: The next two cells will sometimes return an error. The LLM response is variable and at times can't be successfully parsed by the planner or the LLM will make up new functions. If this happens, try resetting the jupyter notebook kernel and running it again.

# COMMAND ----------

from semantic_kernel.planning import SequentialPlanner
from semantic_kernel.core_skills.text_skill import TextSkill
from semantic_kernel.planning.sequential_planner.sequential_planner_config import SequentialPlannerConfig

plugins_directory = "./plugins-sk"
writer_plugin = kernel.import_semantic_skill_from_directory(plugins_directory, "LiterateFriend")

# create an instance of sequential planner, and exclude the TextSkill from the list of functions that it can use.
# (excluding functions that ActionPlanner imports to the kernel instance above - it uses 'this' as skillName)
planner = SequentialPlanner(kernel, SequentialPlannerConfig(excluded_skills=["this"]))

ask = """
Tomorrow is Valentine's day. I need to come up with a poem. Translate the poem to French.
"""

plan = await planner.create_plan_async(goal=ask)

result = await plan.invoke_async()

for index, step in enumerate(plan._steps):
    print(f"✅ Step {index+1} used function `{step._function.name}`")

trace_resultp = True

display(Markdown(f"## ✨ Generated result from the ask: {ask}\n\n---\n" + str(result)))


# COMMAND ----------

# MAGIC %md
# MAGIC Add tracing.

# COMMAND ----------

from semantic_kernel.planning import SequentialPlanner
from semantic_kernel.core_skills.text_skill import TextSkill
from semantic_kernel.planning.sequential_planner.sequential_planner_config import SequentialPlannerConfig

plugins_directory = "./plugins-sk"
writer_plugin = kernel.import_semantic_skill_from_directory(plugins_directory, "LiterateFriend")

planner = SequentialPlanner(kernel, SequentialPlannerConfig(excluded_skills=["this"]))

ask = """
Tomorrow is Valentine's day. I need to come up with a poem. Translate the poem to French.
"""

plan = await planner.create_plan_async(goal=ask)
planner = SequentialPlanner(kernel, SequentialPlannerConfig(excluded_skills=["this"]))
result = await plan.invoke_async()

for index, step in enumerate(plan._steps):
    print(f"✅ Step {index+1} used function `{step._function.name}`")

trace_resultp = True

if trace_resultp:
    print("Longform trace:\n")
    for index, step in enumerate(plan._steps):
        print("Step:", index)
        print("Description:",step.description)
        print("Function:", step.skill_name + "." + step._function.name)
        print("Input vars:", step._parameters._variables)
        print("Output vars:", step._outputs)
        if len(step._outputs) > 0:
            print( "  Output:\n", str.replace(result[step._outputs[0]],"\n", "\n  "))

display(Markdown(f"## ✨ Generated result from the ask: {ask}\n\n---\n" + str(result)))


# COMMAND ----------

# MAGIC %md
# MAGIC # 🔖 There are a variety of limitations to using the planner in August of 2023 in terms of number of tokens required and model preference that we can expect to slowly vanish over time. For simple tasks, this Planner-based approach is unusually powerful. It takes full advantage of both COMPLETION and SIMILARITY in a truly magical way.
# MAGIC
# MAGIC ![](./assets/twodimensions.png)

# COMMAND ----------

# MAGIC %md
# MAGIC So we're coming close to the end. 
# MAGIC I know you're going to miss me, but you might not miss 
# MAGIC this kitchen metaphor, but I hope it works for you because everybody 
# MAGIC wants to be a chef. 
# MAGIC You're an AI chef. 
# MAGIC Are you feeling it? 
# MAGIC Okay, good. 
# MAGIC Keep coming with me. 
# MAGIC So what we're going to do in this section is to 
# MAGIC do something really hard. 
# MAGIC We're going to imagine a world where you just say I'm hungry 
# MAGIC and the LLM is able to complete that meal, use the 
# MAGIC plugins you've created and voila. 
# MAGIC You didn't have to do anything at all. 
# MAGIC You just kind of wished it. 
# MAGIC Let's jump into how that works. 
# MAGIC It's pretty amazing. 
# MAGIC Think of this small business owner. 
# MAGIC I like the two buckets. 
# MAGIC The bucket of time is leaking. 
# MAGIC The bucket of money is leaking. 
# MAGIC You know, do they have time to think about how 
# MAGIC to solve their business problems? 
# MAGIC Heck no. 
# MAGIC Now you have the tools to be able to give them interesting solutions, 
# MAGIC but at the same time, if they could just say, 
# MAGIC I wish I could do X and just have the 
# MAGIC AI help the business owner with as little effort as possible. 
# MAGIC For instance, let's say that the wish were something like, I 
# MAGIC wish I was $10 richer. 
# MAGIC I will need to blank. 
# MAGIC You know about the completion engine now, right? 
# MAGIC It'll complete it, but it's going to hallucinate because it's making stuff 
# MAGIC up. 
# MAGIC What do you do? 
# MAGIC You use retrieval augmented generation to 
# MAGIC kind of find similar things from your knowledge base. 
# MAGIC And then you might have different things that plugins could do, 
# MAGIC like write marketing copy or send an email. 
# MAGIC You can make plugins that do all kinds of things, 
# MAGIC native or semantic. 
# MAGIC And once you do that, what happens is each 
# MAGIC plugin will have a description of some form. 
# MAGIC And all you have to do is use that similarity engine to 
# MAGIC find the different tools in your tool bucket. 
# MAGIC Pause for a second. 
# MAGIC Anybody who has like a drawer full of 
# MAGIC kitchen tools knows that it's hard to find the whatchamacallit, but 
# MAGIC what if the AI can just say, I need something that opens 
# MAGIC cans and then voila, the tool appears. 
# MAGIC That's what happens when you have the similarity engine. 
# MAGIC You can magnetize all your plugins. 
# MAGIC If you had 2000 plugins, you as a human do not want to 
# MAGIC like use every plugin yourself. 
# MAGIC You want the AI to find the right plugin. 
# MAGIC So the third step would be finding the 
# MAGIC relevant plugins and then use those to be 
# MAGIC used in the completion response. 
# MAGIC That's kind of abstract. 
# MAGIC Remember, VAT of plugin, pull out the similar ones, 
# MAGIC use it for completion, push them to the kernel. 
# MAGIC And when you have this thing called a planner, 
# MAGIC the planner does that for you. 
# MAGIC And luckily in Semantec Kernel, we love plugins. 
# MAGIC And now you're learning that we love planners because if you 
# MAGIC got a lot of plugins, you'll need a planner. 
# MAGIC So let's do our quick inventory exercise, which 
# MAGIC you know I like, I'm a Wes Anderson fan. 
# MAGIC Okay, so we did design thinking, and then you just did 
# MAGIC a lot of good stuff. 
# MAGIC if you were able to use the similarity engine to your heart's 
# MAGIC content. 
# MAGIC Wow, you feel good. 
# MAGIC You look so good. 
# MAGIC Okay, that's you. 
# MAGIC Now we're gonna go into this notebook, which 
# MAGIC is the big one. 
# MAGIC First off, let's make a kernel, shall we? 
# MAGIC Let's make a kernel. 
# MAGIC This is the, you know how much I like the fire. 
# MAGIC The fire is what it feels like, because it basically is, you know, 
# MAGIC some GPUs, NPUs. 
# MAGIC Let's make a kernel one more time. 
# MAGIC You're going to miss me because we're not going to make kernels together. 
# MAGIC But you know, we're making kernels like, you know, I 
# MAGIC mean, it's okay. 
# MAGIC Let's just keep making kernels together. 
# MAGIC We can imagine that we're still making kernels together. 
# MAGIC You're doing great, by the way. 
# MAGIC So you know, it's not easy to do what you're doing. 
# MAGIC Okay, we just made the kernel. 
# MAGIC We ran it. 
# MAGIC You know how much, you know how much I like the print message, you know, 
# MAGIC I'm pretty old school print. 
# MAGIC And I, I did it boss, nothing like, you know, nothing like your co-pilot 
# MAGIC calling you boss, feel a little superior, nice job AI, nice 
# MAGIC job computational engine. 
# MAGIC Okay, next thing we're gonna do is remember the, let's take 
# MAGIC notes here for a second. 
# MAGIC We want to have a VAT of plugins and then find the 
# MAGIC right plugin to fit the goal. 
# MAGIC Right? 
# MAGIC So how do we do that? 
# MAGIC Well, there's different kinds of planners. 
# MAGIC Remember the planners, different kind of planners. 
# MAGIC And the reality is there is a super simple planner that people 
# MAGIC kind of make fun of. 
# MAGIC It's called the Action Planner. 
# MAGIC I admit that I had some kind of prejudice for the Action 
# MAGIC Planner and it fully appreciate it. 
# MAGIC But now I do because the Action Planner 
# MAGIC is just not that smart. 
# MAGIC In the Action Planner, you create it from the kernel. 
# MAGIC You give it a bunch of skills, plugins, sorry. 
# MAGIC If you notice Semantic Kernel has like this, 
# MAGIC like a complex where there's skills, but just call them plugins, work 
# MAGIC with me here. 
# MAGIC It's gonna shift over to that. 
# MAGIC But what I'm doing right now is I'm adding the tools for the kernel 
# MAGIC to do math, to read files, to tell 
# MAGIC the time and to play with text. 
# MAGIC Okay, so basically I didn't really do much besides 
# MAGIC add those tools into essentially a vat of 
# MAGIC plugins like I promised. 
# MAGIC And next up, let's do the thing that most people say, I 
# MAGIC was listening to a podcast where people were shading GPT-4, 
# MAGIC  
# MAGIC saying like, oh, well, you know, GPT-4 
# MAGIC is really stupid because it can't add. 
# MAGIC Well, I mean, like, I couldn't add when I was a kid, basically. 
# MAGIC So let's get over that, right? I mean, if 
# MAGIC you gave it the tool to add, you're going to give people a calculator. 
# MAGIC What happens? 
# MAGIC They start calculating, right? 
# MAGIC So we're going to get it to do math. 
# MAGIC And what we're going to do is we're going to use the planner to 
# MAGIC create a one, mind you, a one function, a single 
# MAGIC function that's gonna be pulled out of a vat of 
# MAGIC plugins to use. 
# MAGIC And what it's doing is it's taking this ask, and this is basically looking 
# MAGIC through all the available plugins it has available to 
# MAGIC it, skills, functions, et cetera, okay? 
# MAGIC And what I'm gonna do is I'm gonna ask it to tell me 
# MAGIC what that function is, right? 
# MAGIC All right, let's add some more print statements around this. 
# MAGIC Because programming is pretty abstract unless it tells you 
# MAGIC what you're doing. 
# MAGIC So let's run this. 
# MAGIC And so it's finding the most similar function available 
# MAGIC to get that done. 
# MAGIC What did it do? 
# MAGIC Wow, it knew that if I'm trying to get the sum of two numbers, it 
# MAGIC found in the math plugin, the addition function. 
# MAGIC How did it find it? 
# MAGIC Well, remember we made a description for each function. 
# MAGIC It's just comparing this question into the 
# MAGIC function description. 
# MAGIC Not a surprise, is it? 
# MAGIC Like, let's say like a, what is today? 
# MAGIC It's gonna look through the available plugins and it 
# MAGIC found in the time plugin, the today function. 
# MAGIC Do you see how that's working? 
# MAGIC It, you know, if you totally do something very complex, 
# MAGIC like what is the way to get to San Jose 
# MAGIC when the traffic is really bad? 
# MAGIC Now, this might require many plugins to work in concert, 
# MAGIC but as you can see, it's like, no, I really can't do that, boss, you know? 
# MAGIC  
# MAGIC So for simple things, how do I write the word text to a file? 
# MAGIC It's probably going to find in the file IO skill, it 
# MAGIC found the write function. 
# MAGIC Pretty cool, right? 
# MAGIC Again, it's very limited. 
# MAGIC It's no insult to you, computer. 
# MAGIC It's just not that smart. 
# MAGIC But when you can do that, a simple planner, 
# MAGIC you can imagine a planner that is much more powerful. 
# MAGIC And so the action planner is good for 
# MAGIC like a very basic searching through, find just one function. 
# MAGIC But what if I wanted to do a multi-step plan that's automatically 
# MAGIC generated, right? 
# MAGIC Let's do that. 
# MAGIC So what we're gonna do is we're going to pull in the 
# MAGIC sequential planner. 
# MAGIC The sequential planner is our gen two planner. There's 
# MAGIC a gen three planner that is, it's been ported from C-sharp. so it'll be coming in 
# MAGIC the repo shortly. 
# MAGIC Again, this is all open source, so you can get access to the latest 
# MAGIC and greatest as it comes out, fresh out of our kitchens 
# MAGIC to go into your kitchen. 
# MAGIC And all I'm gonna do is I'm going to bring in the literate friend 
# MAGIC plugin that I have. 
# MAGIC The literate friend plugin has a few functions. 
# MAGIC One, it can write poetry, it can translate, but I'm 
# MAGIC gonna hold onto that. 
# MAGIC Look at that, I'm still like, what a plugin. 
# MAGIC And then what I'm gonna do is I'm gonna make a sequential planner. 
# MAGIC Remember before I made an action planner, which wasn't too smart? 
# MAGIC Well, it's a sequential planner. 
# MAGIC It is. 
# MAGIC I would say like, not just a little, quite a lot smarter. 
# MAGIC I want it to do the following. 
# MAGIC Tomorrow is Valentine's day. 
# MAGIC I need to come up with a poem. 
# MAGIC Translate the poem to French. 
# MAGIC And so this is essentially gonna require two 
# MAGIC plugins, essentially one that can write the poem and 
# MAGIC the other that can translate. 
# MAGIC And I'm gonna do, I'm gonna call the planner. 
# MAGIC Oops, I forgot to close the string. 
# MAGIC That's red, right? 
# MAGIC Okay, there we go. All set? 
# MAGIC Good. 
# MAGIC All right, so we're gonna create plan, async. 
# MAGIC You know, we're built, we're architected in C-sharp where people 
# MAGIC ask, why is there a wait and like async everywhere? 
# MAGIC  
# MAGIC You know, this is enterprise software, 
# MAGIC people doing stuff. 
# MAGIC So I apologize, but in the end, you will thank us for all 
# MAGIC of our attendance to things that can happen asynchronously 
# MAGIC because we live in a network world, right? 
# MAGIC Okay, so that's gonna do that. 
# MAGIC And let's basically, let's print out the plan steps. 
# MAGIC I have that over here in my plated form. 
# MAGIC So let's see what happens. 
# MAGIC I'm going to bring in the literate friend plugin. 
# MAGIC It's got three functions. 
# MAGIC One is able to write a poem, one is able to translate. 
# MAGIC I think the other one is to summarize something, 
# MAGIC but there's three of them in there. 
# MAGIC I'm gonna ask it to make a plan to address this ask and let's see 
# MAGIC what happens. 
# MAGIC So if things work out the way we want, it's gonna realize that I 
# MAGIC need to write a poem and I need to translate it and 
# MAGIC so it pulled out two functions to use and you're like 
# MAGIC well great well can you use them absolutely so what I want 
# MAGIC to do is see what happens what happens when I have it 
# MAGIC actually tell me what it created and it 
# MAGIC says tomorrow's the Valentine's. 
# MAGIC I need to come up with a poem, that's my ask. 
# MAGIC It made a poem and then it translated it to French. 
# MAGIC Now let's do that in super slow motion, shall 
# MAGIC we? 
# MAGIC Let's print out the results step-by-step. 
# MAGIC And that's quite beautiful. 
# MAGIC Over here, okay. 
# MAGIC So I have a little trace results. 
# MAGIC You can look at the code later, but I'm gonna step through the 
# MAGIC plan and look at different things inside it 
# MAGIC and look at the input variables and output 
# MAGIC variables as they change. 
# MAGIC That is the weirdest part. 
# MAGIC So I'm gonna run that, and what you'll be able to see is that as 
# MAGIC the kernel takes the plan, the plan has already built a way to take the 
# MAGIC output from one and stuff it into the input of another, the 
# MAGIC plan has already built a way to take the output from one and 
# MAGIC stuff it into the input of another. 
# MAGIC Now watch this move here. 
# MAGIC The poem has been created and the poem's been created and it figured out 
# MAGIC to add a parameter French. 
# MAGIC Wow, isn't that amazing? 
# MAGIC So it basically plucked out the fact that I needed to make 
# MAGIC it in French and it took the poem output and there you have it. 
# MAGIC That is an automatically generated thing. 
# MAGIC Now, you may not think this is a big deal, 
# MAGIC but it's kind of a big deal because I did not have 
# MAGIC to tell the system to use those two plugins. 
# MAGIC I just gave it a box of plugins and it 
# MAGIC just went and pulled out the ones that need it. 
# MAGIC And number two, it created a plan, a multi-step plan to affect a 
# MAGIC more complex outcome. 
# MAGIC And wait for it. 
# MAGIC How does this work? 
# MAGIC I forget your magical piece of, oops, I forget. 
# MAGIC Let's see here. 
# MAGIC Markdown. 
# MAGIC I forget that there's two dimensions. 
# MAGIC There is completion. 
# MAGIC The completion is generating the plan. 
# MAGIC Similarity is pulling in context for the completion 
# MAGIC to be more right. 
# MAGIC And it's also pulling out the right plugins through the 
# MAGIC descriptions to be able to execute a plan. 
# MAGIC And what does this mean? 
# MAGIC It means that in the future, you won't 
# MAGIC go into kitchen and make dish one, dish two, dish three. 
# MAGIC You'll just say I'm hungry and it'll make you maybe five dishes if it 
# MAGIC has the right ingredients to do that. 
# MAGIC It'll make it the ones that you want. 
# MAGIC Because it's doing completion, it has similarity, understands 
# MAGIC what you have done in the past, and 
# MAGIC it can find the tools in the kitchen to go 
# MAGIC ahead and cook that and also the ingredients too. 
# MAGIC Okay, so that was a lot covered in a short amount of time. 
# MAGIC You must be exhausted by now. 
# MAGIC But I encourage you to play around with 
# MAGIC the different parameters here. 
# MAGIC Add more functions. 
# MAGIC Now that you know about the packaged format of functions, 
# MAGIC semantic functions, go ahead and throw 
# MAGIC in some native functions, which are by nature deterministic. 
# MAGIC Be sure to label all the variables, label all the descriptions, 
# MAGIC because as you now know, this similarity engine thing 
# MAGIC drives a lot of things. 
# MAGIC So it's the magic that lets the AI navigate your 
# MAGIC data to be able to augment the completion they go hand in 
# MAGIC hand left hand right hand together pretty amazing 
# MAGIC now we're gonna go to the conclusion of this course I'm gonna 
# MAGIC miss you but wait till you see what's up if 
# MAGIC you stick around for the next lesson 
# MAGIC

# COMMAND ----------



# COMMAND ----------



# COMMAND ----------



# COMMAND ----------


