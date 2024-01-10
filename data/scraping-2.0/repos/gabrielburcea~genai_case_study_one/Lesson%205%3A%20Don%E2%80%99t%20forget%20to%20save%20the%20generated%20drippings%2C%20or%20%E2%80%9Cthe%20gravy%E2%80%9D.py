# Databricks notebook source
# MAGIC %md
# MAGIC Inventory:
# MAGIC
# MAGIC Kernel
# MAGIC Semantic (and Native) functions -- you can do a lot with these
# MAGIC BusinessThinking plugin --> SWOTs in ways you could never imagine
# MAGIC DesignThinking plugin --> you did that. Congrats
# MAGIC ... next up ... you did all that COMPLETION ✨✨✨✨✨✨✨✨✨✨✨✨✨✨✨✨✨✨

# COMMAND ----------

"""
Embeddings
"""

# COMMAND ----------

from IPython.display import display, Markdown
import semantic_kernel as sk
from semantic_kernel.connectors.ai.open_ai import OpenAIChatCompletion, OpenAITextEmbedding
from semantic_kernel.connectors.memory.chroma import ChromaMemoryStore

kernel = sk.Kernel()

api_key, org_id = sk.openai_settings_from_dot_env()
kernel.add_text_completion_service("openai-completion", OpenAIChatCompletion("gpt-3.5-turbo-0301", api_key, org_id))
kernel.add_text_embedding_generation_service("openai-embedding", OpenAITextEmbedding("text-embedding-ada-002", api_key, org_id))

kernel.register_memory_store(memory_store=ChromaMemoryStore(persist_directory='mymemories'))
print("Made two new services attached to the kernel and made a Chroma memory store that's persistent.")

# COMMAND ----------

import shutil

### ONLY DELETE THE DIRECTORY IF YOU WANT TO CLEAR THE MEMORY
### OTHERWISE, SET delete_dir = True

delete_dir = False

if (delete_dir):
    dir_path = "mymemories"
    shutil.rmtree(dir_path)
    kernel.register_memory_store(memory_store=ChromaMemoryStore(persist_directory=dir_path))
    print("⚠️ Memory cleared and reset")

# COMMAND ----------

strength_questions = ["What unique recipes or ingredients does the pizza shop use?","What are the skills and experience of the staff?","Does the pizza shop have a strong reputation in the local area?","Are there any unique features of the shop or its location that attract customers?", "Does the pizza shop have a strong reputation in the local area?", "Are there any unique features of the shop or its location that attract customers?"]
weakness_questions = ["What are the operational challenges of the pizza shop? (e.g., slow service, high staff turnover)","Are there financial constraints that limit growth or improvements?","Are there any gaps in the product offering?","Are there customer complaints or negative reviews that need to be addressed?"]
opportunities_questions = ["Is there potential for new products or services (e.g., catering, delivery)?","Are there under-served customer segments or market areas?","Can new technologies or systems enhance the business operations?","Are there partnerships or local events that can be leveraged for marketing?"]
threats_questions = ["Who are the major competitors and what are they offering?","Are there potential negative impacts due to changes in the local area (e.g., construction, closure of nearby businesses)?","Are there economic or industry trends that could impact the business negatively (e.g., increased ingredient costs)?","Is there any risk due to changes in regulations or legislation (e.g., health and safety, employment)?"]

strengths = [ "Unique garlic pizza recipe that wins top awards","Owner trained in Sicily at some of the best pizzerias","Strong local reputation","Prime location on university campus" ]
weaknesses = [ "High staff turnover","Floods in the area damaged the seating areas that are in need of repair","Absence of popular calzones from menu","Negative reviews from younger demographic for lack of hip ingredients" ]
opportunities = [ "Untapped catering potential","Growing local tech startup community","Unexplored online presence and order capabilities","Upcoming annual food fair" ]
threats = [ "Competition from cheaper pizza businesses nearby","There's nearby street construction that will impact foot traffic","Rising cost of cheese will increase the cost of pizzas","No immediate local regulatory changes but it's election season" ]

print("✅ SWOT analysis for the pizza shop is resident in native memory")

memoryCollectionName = "SWOT"

for i in range(len(strengths)):
    await kernel.memory.save_information_async(memoryCollectionName, id=f"strength-{i}", text=f"Internal business strength (S in SWOT) that makes customers happy and satisfied Q&A: Q: {strength_questions[i]} A: {strengths[i]}")
for i in range(len(weaknesses)):
    await kernel.memory.save_information_async(memoryCollectionName, id=f"weakness-{i}", text=f"Internal business weakness (W in SWOT) that makes customers unhappy and dissatisfied Q&A: Q: {weakness_questions[i]} A: {weaknesses[i]}")
for i in range(len(opportunities)):
    await kernel.memory.save_information_async(memoryCollectionName, id=f"opportunity-{i}", text=f"External opportunity (O in SWOT) for the business to gain entirely new customers Q&A: Q: {opportunities_questions[i]} A: {opportunities[i]}")
for i in range(len(threats)):
    await kernel.memory.save_information_async(memoryCollectionName, id=f"threat-{i}", text=f"External threat (T in SWOT) to the business that impacts its survival Q&A: Q: {threats_questions[i]} A: {threats[i]}")

print("😶‍🌫️ Embeddings for SWOT have been generated")

# COMMAND ----------

potential_question = "What are the easiest ways to make more money?"
counter = 0

memories = await kernel.memory.search_async(memoryCollectionName, potential_question, limit=5, min_relevance_score=0.5)

display(Markdown(f"### ❓ Potential question: {potential_question}"))

for memory in memories:
    if counter == 0:
        related_memory = memory.text
    counter += 1
    print(f"  > 🧲 Similarity result {counter}:\n  >> ID: {memory.id}\n  Text: {memory.text}  Relevance: {memory.relevance}\n")

# COMMAND ----------

what_if_scenario = "How can the business owner save time?"
counter = 0

gathered_context = []
max_memories = 3
memories = await kernel.memory.search_async(memoryCollectionName, what_if_scenario, limit=max_memories, min_relevance_score=0.77)

print(f"✨ Leveraging information available to address '{what_if_scenario}'...")

for memory in memories:
    if counter == 0:
        related_memory = memory.text
    counter += 1
    gathered_context.append(memory.text + "\n")
    print(f"  > 🧲 Hit {counter}: {memory.id} ")

skillsDirectory = "./plugins-sk"
print(f"✨ Synthesizing human-readable business-style presentation...")
pluginFC = kernel.import_semantic_skill_from_directory(skillsDirectory, "FriendlyConsultant");

my_context = kernel.create_new_context()
my_context['input'] = what_if_scenario
my_context['context'] = "\n".join(gathered_context)

preso_result = await kernel.run_async(pluginFC["Presentation"], input_context=my_context)

display(Markdown("# ✨ Generated presentation ...\n"+str(preso_result)))



# COMMAND ----------

# MAGIC %md
# MAGIC ## 🔖 Reminder: You have now accessed both the COMPLETION and SIMILARITY engines. 
# MAGIC
# MAGIC Huzzah! In doing so, you've unlocked the popular "RAG" (Retrieval Augmented Generation) pattern.
# MAGIC
# MAGIC
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC So the secret to great cooking, especially in the AI era, is don't 
# MAGIC forget to collect all the drippings. 
# MAGIC Turn that into gravy. 
# MAGIC That may sound a little bit disgusting, I know, 
# MAGIC but think of all the things you're generating, auto-completing with 
# MAGIC this magical LLM. 
# MAGIC It's all over the place. 
# MAGIC You have so much of it. 
# MAGIC What if you could harvest that, reuse it, make it a powerful 
# MAGIC partner to the completion engine. 
# MAGIC What if you could use the similarity engine 
# MAGIC that we talked about to be able to play with their gravy? 
# MAGIC Sounds a little bit abstract, let's get concrete and let's work 
# MAGIC with our gravy. 
# MAGIC Let's review inventory, design plugin, you did that. 
# MAGIC Did that, congrats. 
# MAGIC This is where we are right now. 
# MAGIC okay now what we're going to do is next up 
# MAGIC you did all that completion let's sort of make sure that sparkles here 
# MAGIC generate you generate 
# MAGIC it generate it generate it generate it generate it generate it generate 
# MAGIC it and it's kind of like drippings from the meals you've made 
# MAGIC and don't forget that you've used a lot of tokens to make all the 
# MAGIC information wouldn't it be nice if you could like soak it all 
# MAGIC up and 
# MAGIC reuse it somehow think about those two dimensions remember that there's 
# MAGIC the two engines there is the semantic completion engine you've used 
# MAGIC it a lot and now you want to use the semantic similarity 
# MAGIC engine and that is slightly different but so 
# MAGIC similar that you may not notice it as a AI 
# MAGIC chef that was in training like six months ago and 
# MAGIC now I can I can make a pretty good AI meal. 
# MAGIC I've been doing this for so long and I realized wait 
# MAGIC a lot of completion but. I'm getting a lot of similarity now. 
# MAGIC And what does that mean? 
# MAGIC It's hidden in this word that I find, if 
# MAGIC you are a super duper machine learning scientist, 
# MAGIC I apologize, but what the heck was 
# MAGIC this word embeddings? 
# MAGIC This word, I don't know, like this word, so 
# MAGIC much stuff in there. 
# MAGIC You can make a kernel with an embedding model. 
# MAGIC So what happened here? 
# MAGIC I got my hidden key stuff. 
# MAGIC I made a text completion service, but look, I made something 
# MAGIC called an embedding service. 
# MAGIC I use the text embedding service. 
# MAGIC Now, what happens when you do this is you do kind 
# MAGIC of like a double take, like we'd already had a service, completion 
# MAGIC service. 
# MAGIC Well, now you're adding another service. 
# MAGIC It is an embedding service. 
# MAGIC It's the magical machine that takes text and converts it to long vectors 
# MAGIC of numbers. 
# MAGIC And when you do this, you have two powers. 
# MAGIC You have the completion power and the similarity power. 
# MAGIC Now, what you can do with semantic kernel is there's 
# MAGIC different ways to store this information. 
# MAGIC And what we're gonna do is we're going to draw from the 
# MAGIC Chroma memory store. 
# MAGIC There's also a volatile memory store. 
# MAGIC There's a Pinecone, Weaviate, Azure Cognitive Search, a 
# MAGIC bunch of cool ways to kind of like hold on to whatever 
# MAGIC you generate with embeddings. 
# MAGIC And the way you add it is you do the following. 
# MAGIC Kernel dot register memory store, memory store equals 
# MAGIC Chroma memory store store. 
# MAGIC And you can close that and just run it, 
# MAGIC but what happens is the data that goes 
# MAGIC into that vector database goes away unless you 
# MAGIC say persist directory equals and give it a name. 
# MAGIC I'm using my memories to hold on to it okay made two nervous 
# MAGIC touch the kernel and made a chroma memory store that's persistent 
# MAGIC  
# MAGIC okay let's see this works look at that no errors didn't feel 
# MAGIC good so I did three things there I made a completion service. 
# MAGIC I made a similarity service also called embeddings 
# MAGIC and I attached a memory store to the 
# MAGIC kernel so I can hold on to vectors that I generate from 
# MAGIC the embedding conversion process. 
# MAGIC Just a bit of a caveat because we want this 
# MAGIC notebook to be useful to you. 
# MAGIC Let's say you run this and your memory store 
# MAGIC which is going to be stored in my 
# MAGIC memories directory starts to give you errors. 
# MAGIC So this is a. I'm gonna, let's see, delete dir equals true. 
# MAGIC If I run this, it just deleted the memory store folder. 
# MAGIC You might say like, wait, bring it back. 
# MAGIC Well, first off, let's just go back here and run 
# MAGIC this code above and no worries, it's there. 
# MAGIC And let's just ignore this code, walk over it, look aside, look 
# MAGIC aside. 
# MAGIC Okay, how you feel? 
# MAGIC We're here now. 
# MAGIC Okay, next we're gonna do is we're gonna put stuff into the memory store 
# MAGIC after the embedding vector has been 
# MAGIC created. 
# MAGIC You ready? 
# MAGIC Okay, so first off, let's get some data. 
# MAGIC You know, I like data, this data from the SWATs. I like the 
# MAGIC question and answer pairs because those 
# MAGIC are generally useful as information. 
# MAGIC Right now they're stored in native memory. 
# MAGIC They haven't gone anywhere. 
# MAGIC Okay, so let's now put all of those strengths, weaknesses, 
# MAGIC opportunities, and threats into memory. 
# MAGIC So I'm gonna add them to memory collection named SWAT. 
# MAGIC I'm gonna loop over the different arrays of strings and let's just 
# MAGIC neatly put them all into, there we go. 
# MAGIC So now it's sitting in the vector store. 
# MAGIC Fantastic. 
# MAGIC Okay. 
# MAGIC Now what? 
# MAGIC Let's now use the magnet. 
# MAGIC You know how much I love the magnet symbol? 
# MAGIC Magnet. 
# MAGIC Let's use a magnet. 
# MAGIC Okay. 
# MAGIC So I'm now going to look at this SWAT. 
# MAGIC The SWAT's all in vector memory, and I'm now 
# MAGIC going to ask questions of it. 
# MAGIC So, what are the easiest ways to make more 
# MAGIC money is the question I'm going to ask, and. 
# MAGIC I'm going to do the same kind of memory search async, I'm going 
# MAGIC to pluck out the different memory results, I'm also going 
# MAGIC to let you see the relevance score, remember 0 to 
# MAGIC 1, 1 is like perfect match, 0 is no 
# MAGIC match. 
# MAGIC Let's run that. 
# MAGIC And so now, it compares what are the easiest ways to 
# MAGIC make more money to what's in the vector store. 
# MAGIC And this is the first one that's coming up. 
# MAGIC It's saying catering potential. 
# MAGIC It's saying the annual food fair is coming. 
# MAGIC And so you see, it's basically sorted the most similar 
# MAGIC item to the query. 
# MAGIC It's kind of amazing, isn't it? 
# MAGIC Like, let's change that. 
# MAGIC Go ahead and change this. It's kind of like an amazing feeling. 
# MAGIC What are the easiest ways to save money? 
# MAGIC Let's see what it does with that one. 
# MAGIC It says partnerships. 
# MAGIC It says, worry about your competition. 
# MAGIC The cheese, don't forget the cheese. 
# MAGIC And so again, this is a magical machine now that takes 
# MAGIC your gravy drippings and uses it. 
# MAGIC And this kind of, remember left hand, right hand? 
# MAGIC This is your left hand doing amazing things. 
# MAGIC Okay, so let's go into a super long example. 
# MAGIC Now, I think you're kind of tired of that long example. 
# MAGIC So let me give you something a little 
# MAGIC easier because typing this is kind of hard. 
# MAGIC Here we go. 
# MAGIC Okay, let's read this code here for a second. 
# MAGIC All right, let's have a what if scenario. 
# MAGIC Well, how can the business owner save time? 
# MAGIC It's going to do the memory search. 
# MAGIC It's going to find the most similar memories. 
# MAGIC I'm gonna use a plugin, a plugin from the 
# MAGIC friendly consultant folder, plugin collection. 
# MAGIC And I'm gonna ask it to give me a presentation. 
# MAGIC I've made a plugin to make a presentation about anything I 
# MAGIC ask it to do. 
# MAGIC And long story short, set the context. 
# MAGIC And I ask it to run. 
# MAGIC Let's see how this works. 
# MAGIC So first off, it's used a similarity engine 
# MAGIC to find the most similar pieces of context. 
# MAGIC It's going to take all of that and give it to 
# MAGIC the prompt that is going to generate the presentation. 
# MAGIC So this is that example of retrieval augmented 
# MAGIC generation. 
# MAGIC The generated information is taken from 
# MAGIC the actual information stored in the vector database. 
# MAGIC So there you have it. 
# MAGIC This is a professional presentation from a consultant. 
# MAGIC The question is, how can the business owner save time? 
# MAGIC No problem, boss. 
# MAGIC Here are the three concerns. 
# MAGIC Here's how to address them individually. 
# MAGIC And this is what I brought to you. 
# MAGIC Kind of amazing? 
# MAGIC No, yes, no. 
# MAGIC And again, remember, you can change everything in these notebooks, 
# MAGIC like not just this one with other ones, and you can do entirely different 
# MAGIC analyses. 
# MAGIC Okay, so I wanna summarize here before 
# MAGIC we go to the next section, but I wanna 
# MAGIC congratulate you because you've now unlocked this very 
# MAGIC popular acronym called RAG. 
# MAGIC and you've accessed both the completion and similarity engines. 
# MAGIC Congratulations. 
# MAGIC You were able to bind similar things in the vector 
# MAGIC database. 
# MAGIC You were able to give them to a completion prompt and generate 
# MAGIC something on point. 
# MAGIC No hallucination needed. 
# MAGIC Doesn't that feel good? 
# MAGIC And now that you've done this, we're going to take you into the next chapter, which 
# MAGIC is all about something at a 
# MAGIC whole different level. 
# MAGIC Once you master plugins and similarity completion, 
# MAGIC all this world, world, you suddenly discover that it's 
# MAGIC time to make a plan, 
# MAGIC make an AI plan, generate a way to solve a goal instead 
# MAGIC of just sift through your plugins by hand. 
# MAGIC Have the AI go look through your giant Lego made of plugins. 
# MAGIC You don't have to go look through them. 
# MAGIC The AI can look through them. 
# MAGIC What's that like? 
# MAGIC Let's jump into plans. 
# MAGIC
# MAGIC
