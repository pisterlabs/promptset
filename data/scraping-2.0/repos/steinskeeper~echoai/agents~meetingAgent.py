import re
from bson.objectid import ObjectId
import asyncio
import json
from pymongo import MongoClient
import time
import openai
from nanoid import generate as generate_id
openai.api_key = ""
client = MongoClient("mongodb://localhost:27017/")
from langchain.tools import DuckDuckGoSearchResults
db = client["echoai"]
import pprint
pp = pprint.PrettyPrinter(indent=4)
import whisper
model = whisper.load_model("small", "cpu")
import os
from langchain.utilities import BingSearchAPIWrapper
os.environ["BING_SUBSCRIPTION_KEY"] = ""
os.environ["BING_SEARCH_URL"] = "https://api.bing.microsoft.com/v7.0/search"
meetingTranscript = """
Developer Standup Transcript - August 19, 2023

Team Lead (Alex): Good morning, everyone. Let's kick off this standup. Who wants to start?

Developer 1 (Sarah): Morning, team. Yesterday, I was working on the user authentication module. I encountered an issue with the third-party library integration. I've reached out to their support and hope to have a solution by today.

Alex: Thanks, Sarah. Keep us posted on the progress. Next?

Developer 2 (John): Hi all. I made significant progress on the front-end redesign. The new layout is almost complete, and I've started optimizing the code for better performance. I'll be pairing up with Sarah once her authentication piece is ready.

Alex: Great work, John. Collaboration is key. Anyone else?

Developer 3 (Emily): Morning, team. I've been tackling the database performance bottleneck. After some profiling, I identified a few queries that need optimization. I've already made the necessary adjustments, and preliminary tests show improved response times.

Alex: Excellent, Emily. Performance improvements are always appreciated. Any blockers?

Developer 4 (Michael): Hey everyone. I've been exploring the integration with the analytics service. I hit a snag due to the differences in data structures, so I'll need to coordinate with Emily to align the database changes.

Alex: Thanks for flagging that, Michael. Collaboration between teams is crucial. Anything else before we move on?

Developer 5 (Olivia): Morning, team. I've been researching ways to streamline our deployment process. I've come across some new tools that could help automate our deployments and reduce manual errors. I'll share my findings with everyone once I've done some testing.

Alex: That sounds promising, Olivia. Looking forward to your insights. Now, onto the topic of the day. As we've discussed in our recent architecture meetings, there's been increasing interest in migrating to a microservices architecture. This would allow us to decouple our monolithic application into smaller, more manageable services. We've seen the benefits in terms of scalability, agility, and fault isolation.

Sarah: That's an exciting direction. It would certainly help us scale different components independently and enable faster development cycles.

John: Absolutely. I can see how it would allow us to make updates to specific features without affecting the entire application.

Emily: From a database perspective, this would also mean we could choose the most appropriate database technology for each service's needs, optimizing performance further.

Michael: One concern I have is the increased complexity of managing multiple services and their interactions. We'd need to establish strong communication patterns and possibly adopt new tools for monitoring and debugging.

Olivia: I've been researching tools that can assist with service discovery, load balancing, and monitoring in a microservices environment. It's definitely a shift in mindset, but with the right tools and practices, I believe we can manage the complexity effectively.

Alex: Those are all valid points. Before we proceed, I'll set up a more focused architecture discussion to delve deeper into the migration process, potential challenges, and the tools we'll need. In the meantime, I encourage everyone to continue researching and thinking about how our current components can be broken down into microservices.

Sarah: Sounds like a plan, Alex. Looking forward to diving into this exciting phase. I believe we should do a bit of research on Svelte and see if we can migrate our frontend to Svelte while comparing Svelte vs React

Alex: Great. Let's get back to our tasks for now, and we'll regroup soon to dive into the details. Keep up the good work, team. Standup adjourned.
"""
noSpeakerstranscript = """
Developer Standup Transcript - August 19, 2023

Team,

In our standup meeting today, we commenced by discussing our ongoing tasks and progress. Sarah shared her work on the user authentication module, highlighting an issue encountered during the integration of a third-party library. She's actively engaged with the library's support team and expects a resolution today.

Moving on, John reported significant advancement in the front-end redesign. He's made substantial progress in optimizing the code for better performance. John plans to collaborate with Sarah, who's handling the authentication component.

Emily then shared her efforts to address a database performance bottleneck. Through profiling, she identified specific queries requiring optimization. Her proactive adjustments have already resulted in noticeable improvements in response times.

Our next update revolved around integration with the analytics service. Michael discussed challenges arising due to disparities in data structures. Effective collaboration between Emily and Michael will ensure the necessary alignment of database changes for successful integration.

The focus then shifted towards streamlining our deployment process. Olivia delved into research on automation tools to minimize manual errors in deployments. She expressed intentions to share her findings with the team after conducting thorough testing.

Now, turning to the main topic of our discussion today: microservices architecture. In recent architecture meetings, the team expressed growing interest in migrating towards this approach. The motivation behind this transition is to break down our monolithic application into smaller, manageable services. Such a shift offers benefits including enhanced scalability, greater agility, and improved fault isolation.

As we delved into this topic, Sarah shared her excitement about the prospect of scaling different components independently and enabling faster development cycles. John highlighted how microservices would facilitate updates to specific features without affecting the entire application.

Considering the database aspect, Emily explained how adopting microservices would permit the selection of database technologies tailored to the needs of each service, thereby optimizing overall performance.

However, amidst these advantages, Michael raised a valid concern regarding the potential complexity in managing multiple services and their interactions. This complexity demands the establishment of robust communication patterns and possibly the adoption of new monitoring and debugging tools.

Addressing this concern, Olivia spoke about her research into tools designed to assist with service discovery, load balancing, and monitoring within a microservices environment. She acknowledged the paradigm shift required but expressed confidence in effectively managing complexity with the right tools and practices.

Acknowledging these viewpoints, it was decided to initiate a more comprehensive architecture discussion, focused on the migration process, potential challenges, and required tools. As we conclude, team members were encouraged to continue researching and contemplating how to decompose our current components into microservices.

In summary, our standup covered a range of ongoing tasks and updates, leading to an insightful conversation about adopting a microservices architecture. This shift presents various opportunities and challenges that we're ready to address as a dedicated team. With our collective efforts, Team, we're poised to embrace this new phase of development. Keep up the excellent work. Standup adjourned.
"""

danmeeting = """
Yajat : Hey, danny boi, Good morning. 
Danny: Hi Yajat, Good morning. What are we doing today?
Yajat : So like we discussed yesterday, we have to start working on a new feature for the client . The requirement is a solution to convert invoices into text and then load this into the payables database.
Danny: Oh we can use OCR technology to do that. I think we have plenty open source libraries to start from.
Yajat: That's great, let's get started on finding the best open source solution. On a different line of thought I was wondering if we could use a paid API service for the same?
Danny: We can try to do that aswell. But I think we will have to run it by Vibha before we do that since she will be the one signing off the finances to buy the API
Yajat: Isn't Vibha on this call?
Danny: Maybe not, I think she'll join later.
Yajat: All right, I'll ask her about this.

Danny: Sounds like a plan. I'll start researching open source OCR libraries. I remember Tesseract being a popular one. I'll also explore a few others to see which fits our requirements the best.

Yajat: Perfect. Meanwhile, I'll look into paid API options. I remember seeing some cloud-based OCR services that might be a good fit. I'll compile a list of potential options along with their pricing plans.

Danny: Great, once we have both options evaluated, we can present them to Vibha for her input. Also, let's keep the client updated with our findings.

Yajat: Sounds good. I had a different question, Will we be using Python or JavaScript to parse the output?

Danny: We'll have to look into that aswell, because python has great libraries but js is highly scalable

Yajat: Okay, I will look into what we can use. Give me some time and let's connect back after a while and present what we have to Vibha so that she is up to speed.

Danny : Works. I will see you in sometime.

Yajat: Awesome. Bye

Danny: Bye
"""
async def invokeAgents(meetingid, filename):
    print("Meeting Agent called")
    transcript = await transcriptGenerator(filename)
    transcript = danmeeting
    rep = await summaryGenerator(transcript)
    todo = await todoGenerator(transcript)
    keyTakeways = await keyTakewaysGenerator(transcript)
    absentia = await absentiaGenerator(transcript)

    references = await referencesGenerator(transcript)
    emails = await emailSuggestions(transcript)
    meetings = await meetingSuggestions(transcript)
    result = {
        "transcript": transcript,
        "summary": rep["summary"],
        "meetingName": rep["title"],
        "todo": todo,
        "keyTakeways": keyTakeways,
        "absentia": absentia,
        "references": references,
        "emails": emails,
    }
    pp.pprint(result)
    db.meetings.update_one({"_id": ObjectId(meetingid)}, {"$set": {"agent": result}})
    print("Meeting Agent completed")



async def transcriptGenerator(filename):
    print("Transcript Generator invoked")
    result = model.transcribe('./uploads/'+filename)
    speech = result["text"]
    print(speech)
    return speech


async def summaryGenerator(transcript):
    print("Summary Generator invoked")
    prompt = '''
    You are a summariser.
    You are given a transcript from a meeting summarise it.
    The transcript is:
    ''' + transcript + '''

    The summary must be clear, concise, and accurate.
    '''
    
    # create a chat completion
    summary = openai.ChatCompletion.create(
        model="gpt-3.5-turbo", messages=[{"role": "user", "content": prompt}]).choices[0].message.content
    print(summary)
    promptfortitle = '''
    You are a meeting name generator.
    You are given a transcript from a meeting, generate a relevant title for the meeting.
    The transcript is:
    ''' + transcript + '''

    The name must be clear, concise, and accurate.
    '''
    title = openai.ChatCompletion.create(
        model="gpt-3.5-turbo", messages=[{"role": "user", "content": promptfortitle}]).choices[0].message.content
    
    rep = {
        "title": title,
        "summary": summary
    }

    return rep


async def todoGenerator(transcript):
    print("Todo Generator invoked")
    prompt = '''Transcript :  '''+transcript +'''
    You are a todo generator. You must always follow the rules
    Task : Analyze the provided meeting transcript and generate todo tasks for every person. 
    Rules :
    1. The output must be an JSON array of strings containing all the tasks, for example ["task1", "task2"].
    2. Don't add comma at the end of the array. It should be a valid JSON array.
    '''
    todo = openai.ChatCompletion.create(
        model="gpt-3.5-turbo", messages=[{"role": "user", "content": prompt}]).choices[0].message.content
    print(todo)
    print(json.loads(todo))
    return json.loads(todo)

async def keyTakewaysGenerator(transcript):
    print("Key Takeaways Generator invoked")
    prompt = '''Transcript :  '''+transcript +'''
    You are a key-takeaway generator. You must always follow the rules
    Task : Analyze the provided meeting transcript and generate key-takeaway points. 
    Rules :
    1. The output must be an JSON array containing all the key-takeaways.
    2. Generate only the array part of the JSON, for example ["key-takeaway1", "key-takeaway2"].
    3. Don't add comma at the end of the array. It should be a valid JSON array.
    '''
    takeaway = openai.ChatCompletion.create(
        model="gpt-3.5-turbo", messages=[{"role": "user", "content": prompt}]).choices[0].message.content
    print(takeaway)
    print(json.loads(takeaway))
    return json.loads(takeaway)
    


async def absentiaGenerator(transcript):
    print("Absentia Generator invoked")
    prompt = '''Transcript :  '''+transcript +'''
    You are a key-takeaway generator. You must always follow the rules
    Task : Analyze the provided meeting transcript and generate key-takeaway for each person that took part in the meeting or were mentioned in the meeting.
    Rules :
    1. The output must be an JSON array in the format :  [{name: 'name of the person', content : 'key takeaways for this person'}].
    2. Generate only the array part of the JSON.
    3. Don't add comma at the end of the array. It should be a valid JSON array.
    '''
    takeaway = openai.ChatCompletion.create(
        model="gpt-3.5-turbo", messages=[{"role": "user", "content": prompt}]).choices[0].message.content
    print(takeaway)
    print(json.loads(takeaway))
    return json.loads(takeaway)

async def referencesGenerator(transcript):
    print("References Generator invoked")
    prompt = '''Transcript :  '''+transcript +'''
    You are a topic extractor. You must always follow the rules
    Task : Analyze the provided meeting transcript and give the most prominent and niche topics that can be helpful in order to get further insights after the meeting. 
    Rules :
    1. The output must be an JSON array containing all the keywords.
    2. Generate only the array part of the JSON, for example ["topic1", "topic2"].
    3. Generate only the most importat topics that would need further references to the team. For example things that were discussed in the meeting that would need further research.
    4. Do not include one word generic topics. Instead it should be phrases.
    5. Do not include more than 5 topics.
    6. Don't add comma at the end of the array. It should be a valid JSON array.
    '''
    takeaway = openai.ChatCompletion.create(
        model="gpt-3.5-turbo", messages=[{"role": "user", "content": prompt}]).choices[0].message.content
    print(takeaway)
    pp.pprint(json.loads(takeaway))
    search = BingSearchAPIWrapper()
    res = []
    for i in json.loads(takeaway):
        x = (search.run(i,1))
        for m in x:
            res.append({
                "title": m["title"],
                "link" : m["link"],
                "keyword": i
            })
        
    pp.pprint(res)
    y = {
        "type": "REFERENCE",
        "links": []
    }
    for j in res:
        y["links"].append(i)
    return y


async def emailSuggestions(transcript):
    print("Email Suggestion Generator invoked")
    prompt = '''Transcript :  '''+transcript +'''
    You are an email writer. You must always follow the rules
    Task : Analyze the provided meeting transcript and generate follow up emails where ever required. 
    Rules :
    1. The output must be an JSON array in the format :  [{reciever: 'name of the person the email is addressed to', content : 'email content', title: 'subject of the email', reason: 'why the email needs to be sent}].
    2. Generate only the emails that are required to be sent.
    3. If there are no emails to be sent, generate an empty array.
    4. Don't add comma at the end of the array. It should be a valid JSON array.
    '''
    takeaway = openai.ChatCompletion.create(
        model="gpt-3.5-turbo", messages=[{"role": "user", "content": prompt}]).choices[0].message.content
    print(takeaway)
    print(json.loads(takeaway))
    fin = {
        "type": "EMAIL",
        "emails": []
    }
    for i in json.loads(takeaway):
        fin["emails"].append({
            "reciever": i["reciever"],
            "emailContent": i["content"],
            "emailSubject": i["title"],
            "emailCTA": i["reason"],
            "actionCompleted" : False,
        })
    return json.loads(takeaway)



async def meetingSuggestions(transcript):
    print("Meeting Suggestion Generator invoked")
    prompt = '''Transcript :  '''+transcript +'''
    You are an meeting scheduler. You must always follow the rules
    Task : Analyze the provided meeting transcript and schedule follow up meeting where ever required. 
    Rules :
    1. The output must be an JSON array in the format :  [{reason : 'purpose of the meeting', title: 'title of the meeting'}].
    2. Generate only the meetings that are required to be scheduled.
    3. If there are no meetings to be scheduled, generate an empty array.
    4. Don't add comma at the end of the array. It should be a valid JSON array.
    '''
    takeaway = openai.ChatCompletion.create(
        model="gpt-3.5-turbo", messages=[{"role": "user", "content": prompt}]).choices[0].message.content
    print(takeaway)
    print(json.loads(takeaway))
    return json.loads(takeaway)

