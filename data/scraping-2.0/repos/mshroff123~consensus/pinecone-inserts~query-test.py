import json
import openai
import os
import pinecone
import pandas as pd
from langchain.document_loaders import DataFrameLoader
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.chains.summarize import load_summarize_chain
from langchain.prompts import PromptTemplate
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.text_splitter import CharacterTextSplitter
from langchain.docstore.document import Document
import datetime
from tqdm.auto import tqdm


def get_key_claims_hash(comments, llm_resp_array):
    # init pinecone

    batch_size = 128

    index_name = 'semantic-search-relevant-comments'

    # only create index if it doesn't exist
    if index_name not in pinecone.list_indexes():
        pinecone.create_index(
            name=index_name,
            dimension=1536,
            metric='cosine'
        )

    # now connect to the index
    print('done creating index')
    index = pinecone.Index(index_name)
    print('connected to index')

    # upload comments to pinecone
    for i in tqdm(range(0, len(comments), batch_size)):
        # find end of batch
        i_end = min(i + batch_size, len(comments))
        # create IDs batch
        ids = [str(x) for x in range(i, i_end)]
        # create metadata batch
        metadatas = [{'text': text} for text in comments[i:i_end]]
        # create embeddings
        # xc = model.encode(questions[i:i_end])
        openai_embed = OpenAIEmbeddings(openai_api_key=openai.api_key)

        xc = openai_embed.embed_documents(comments[i:i_end])
        # create records list for upsert
        records = []
        for i in range(len(ids)):
            record = (ids[i], xc[i], metadatas[i])
            records.append(record)

        # upsert to Pinecone
        print("uploading")
        index.upsert(vectors=records)

    # semantic search comments now on key claims
    key_claims_hash = {}
    for claim in llm_resp_array:
        # get 5 most relevant comments
        openai_embed = OpenAIEmbeddings(openai_api_key=openai.api_key)
        xq = openai_embed.embed_query(claim)
        xc = index.query(xq, top_k=5, include_metadata=True)
        for result in xc['matches']:
            if result['score'] < .5:
                break
            key_claims_hash[claim] = result['metadata']['text']
            # query dynamo db table to get all related questions

    pinecone.delete_index(index_name)
    print("FINAL TIME")
    e = datetime.datetime.now()
    print(e)

    print(key_claims_hash)



if __name__ == "__main__":
    openai.api_key = "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"
    os.environ["OPENAI_API_KEY"] = "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"


    # TODO implement
    # model = SentenceTransformer('./model/')

    # init pinecone
    PINECONE_API_KEY = '974f9758-d34f-4083-b82d-a05e3b1742ae'
    PINECONE_ENV = 'us-central1-gcp'

    pinecone.init(
        api_key=PINECONE_API_KEY,
        environment=PINECONE_ENV
    )

    index = pinecone.Index("semantic-search-6998")

    # Sample Query
    query = "Best exercises for strengthening my wrists?"

    # create the query vector
    # xq = model.encode(query).tolist()

    # now query
    e = datetime.datetime.now()
    questions = []

    openai_embed = OpenAIEmbeddings(openai_api_key=openai.api_key)

    xq = openai_embed.embed_query(query)

    # now query
    e = datetime.datetime.now()
    questions = []
    xc = index.query(xq, top_k=3, include_metadata=True)
    # xc = index.query(xq, top_k=3, include_metadata=True)

    print(e)

    ex = datetime.datetime.now()

    '''
    for result in xc['matches']:
        response = questions_table.query(
            IndexName='title-index',
            KeyConditionExpression=Key('title').eq(result['metadata']['text'])
        )
    
        # query dynamo db table to get all related questions
    
        for item in response['Items']:
            questions.append(item['id'])
    '''
    # query dynamo db to get all comments

    comments = ['Standing Neider presses, Turkish Getups, Clean and presses, Power Cleans, Standing Dumbbell shoulder presses, 1-arm Dumbbell bench presses  Definitely all my favorites.  Edit: Save the power cleans, and clean and presses for a bit later down the line...', 'Squats, deadlifts and bench presses are the three exercises you should never go without. This is because they are compound exercises, in that they work many different muscles rather than isolating a single one.  [This is a pretty good resource](http://www.sport-fitness-advisor.com/dumbbellexercises.html) for dumbbell exercises. The animated pictures really helped when I was first starting out.', "I like to work out the shoulders a lot.  You'll notice other muscle areas stemming from shoulder workouts.  I noticed my traps increasing in size even though I, stupidly, didn't ever work out my traps or back.    I worked my anterior delts until they were freakishly strong... I'm of relatively small stature, but I could lift 60lb dumbells straight out in front of me.  I couldn't hold it out there but I could put it up with decent form.   The strength of my shoulders was an awesome foundation for many, many of my other upper-body workouts.  ", 'I use FitnessBliss to manage my [workout routines](http://www.fitnessbliss.com)', "http://www.amazon.com/Starting-Strength-2nd-Mark-Rippetoe/dp/0976805421/ref=sr_1_1?s=books&amp;ie=UTF8&amp;qid=1290392860&amp;sr=1-1  This is all you need. Doing isolation exercises like most people will recommend will only slow down your progress. Just do big, compound lifts and you'll gain strength faster than you thought possible. When you stop gaining strength every time you go to the gym, you can expand what you're doing.", "http://www.dumbbell-exercises.com/  I'm a fan of this site.  I only have dumbbells and bodyweight to work with and still managed to tailor a full body workout.  http://www.reddit.com/r/Fitness/comments/e71s9/good_protein_mix_for_someone_who_wants_to_bulk_up/c16ecjl", 'Step 1: Go to Gym  Step 2: Pick up weights  Step 3: Run like hell out of the gym before they catch you.  Step 4: Enjoy your free weights and cardio exercise ', 'Dumbbell Chest Press, Shoulder Press, Romanian deadlifts, Shoulder shrugs, obviously Bicep curls and Tricep extensions.  Flies, one armed row (not sure what its called), lat raises.  I only use one machine for upper body in the gym. With free weights and a bench/bar I could do my entire workout (with slight modifications).  Free weights are good because they encourage correct form and isolate less.', "if your a gym noob and JUST starting out gym/weight training, i would suggest staying off the free weight exercises, at least till your muscles have adjusted to the motion of the weights. do machine exercises first as this will allow it.  i have seen so many people injure themselves from DB presses and the like. don't be one of them pls.   once you have gone past the memory muscle thing, start with the light DB presses and then ease into the idea of free weights.  ask more if you have further questions.", "Read a beginner's program, it makes everything a LOT less confusing. Starting Strength, Westside for Skinny Bastards, something like that. You do those for a few months and it gives you a good base of strength, good form, and a basic gym education. Then you can decide where you want to go from there once you have some knowledge", "Muscle-ups. I can only do a few of them and my form isn't that great but it just feels cool to be able to get all the way up on the bar and just hold it there for a second.", 'Chinups/Pullups', 'I like turtles', 'Sled.', 'squat', 'DB press', "Without wanting to just go along with the general consensus here, I have to say chins and deads.  I'm a novice, so the most important criteria for an exercise are form and safety.   Chin form is piss-simple and baby-safe—failure results in simply landing on your feet.  And when it comes to weight exercises I feel like deadlift form is second to bench press for simplicity, but no. 1 for safety—failure results in either not being able to lift the weight from the ground in the first place or being forced to drop it to the ground.  Plus they're both compound movements, which is great if your goal is full-body strength and fitness.", 'I have a semi-romantic relationship with dips.', "OHP. I don't care if I suck at it, it makes me feel like Hercules ", 'Benchpress and pull-ups/chin-ups &amp;  dips', 'Squat and OHP. ', 'Incline/decline bench always feels great. Cable flyes too. Feels like a get bigger after every session.', 'Barbell curls, standing in the squat rack (for safety).', 'Now that I can finally do them, chin ups. Still tough for me but getting progressively better.', 'Deadlift. It builds character and strong bones. Got Deadlift ?', 'Every work out I do is just one day closer to deadlift day. ', 'Deadlifts.', 'I like to do a deadlift and then hold it for 20 seconds at the top. When I set that shit down and stand back up, I feel the euphoria of the gods.', 'Just did hang cleans for the first time yesterday, and they are a ton of fun. Absolutely destroyed my traps', 'Calf raises, that burn', "Deadlift and bench press.   I don't care how bro-ish it sounds BP just makes me feel powerful. ", 'Power clean, so much fun', 'Easily Deadlifts', 'Dumbbell chest presses, pull-ups.', 'At that age I would say bodyweight workouts  A calorie deficit is the only way you’re really going to lose much weight. So make that your main focus and exercise complementary', 'As your vacation starts in 3 weeks, pretty much irrelevant if that was your hope. If you wanted progress 6 months a go is when to start.', 'Have a look at our recommended workouts in the wiki: https://thefitness.wiki/routines/strength-training-muscle-building/  The wiki also covers how to setup your diet for fat loss.', "The best workouts are the ones you will actually do. Do you have access to a gym? Do you like to run?  At the end of the day, however, working out won't get you the results (fat loss) you want; you need to track calories and weight and eat less. See https://thefitness.wiki/weight-loss-101/", "3 day a week sounds the best of those 3 options to me, and you can start supplementing in other exercises if you need to. make sure you do 3 or 4 sets of 8-12 reps, if you get past 12, up the weight. And you won't look like arnold or have to be in a weight lifting competition (i know it was a joke) if you do starting strength. It is STARTING STRENGTH, literally the basic compound exercises which help you learn and gain some muscle. If you keep your diet the same, you won't get huge or anything, just stronger.", 'why not follow a program like starting strength or strong lifts? You seem like your just trying to re-invent the wheel', "Push ups and pull ups. Simple and they work a ton of muscles. There's a good reason the military makes you do a ton of pushups. ", 'Horizontal pushing: One handed pushup progression or pseudo planche pushup progression  Vertical pushing: dip progression or handstand pushup progression  Horizontal pulling: one handed inverted row progression or front lever row progression  Vertical pulling: one handed chinup or pullup progression.   Legs: sprints, pistol squat progression, plyometrics   Other:  back lever, plank, side plank, any ring movement', "For what goal? Strength/muscle building? Not many to be honest. But for cardio or just general fitness... Burpees, squat jumps, paratroopers, push ups and all their varieties, all the crunch varieties, the list goes on forever. If you want the best cardio work out with some strength component to it, just do like 30 seconds of 5 of these exercises in a row, rest 30 seconds, then repeat till you're only getting a couple of reps per 30 second rep.", 'Anything high tension like one arm push-ups, pull-ups, pistol squats, and muscle-ups. They tend to give you far more bang for the buck then pumping out endless push-ups and crunches. ', 'http://www.nerdfitness.com/blog/2009/12/09/beginner-body-weight-workout-burn-fat-build-muscle/', '/r/bodyweightfitness might be able to help.', 'Seconding the recommendations for /r/Fitness and Starting Strength.  Also, check out CrossFit.  Even without joining a "box", the philosophy is sound and the workouts are excellent.', "You should check out /r/Fitness, they've got information.  If you're just starting to get into lifting, you should look into Starting Strength (SS).", "Weight loss and fat loss are two different things. Exercise can actually make you gain weight via muscle/bone gains. You can't spot-reduce fat, or really tone one area vs another. You can only gain muscle from working them, or lose fat in the areas your genes want you to at that point. Generally, everyone is different, but the last place you gained a spot of fat is the first place you'll lose it and vice-versa.  Fat loss is mostly diet. If you get that squared away, then will exercise help more. You sound like you're getting a handle on this. Can't go wrong with protein and veggies, avoiding large servings of processed foods and sugar/starch/white flour.  As for an exercise program: Cardio burns calories while you're doing it, and then mostly stops. Lifting weight burns fewer calories during the gym session, but if you lift (heavy), you burn a TON of calories for the 48 hour recovery period (especially the larger leg/hip muscles). Mostly in your sleep, as that's where a body does the most healing in general. Increasing muscle mass helps burn a little more, but its mostly the recovery period thing for calorie burning. So for fat loss, many people combine cardio and lifting, lifting as many times per week as recovery will allow. Generally each muscle can be worked every other day with heavy lifting.   If you want to hit the gym and lift really heavy for maximum fat loss, try [Starting Strength](http://startingstrength.com/) or [The New Rules of Lifting for Women](http://www.amazon.com/New-Rules-Lifting-Women-Goddess/dp/1583332944). If you still want to work out at home, then also look into [Ross Enamait's ideas](http://rosstraining.com/blog/), particularly his article section and his book Never Gymless. Good ideas on what to do and what gear to get. Lots of ideas on how to make some of your own gear cheaply, as well  If you're worried about bulking up from lifting heavy weights, read [this article](http://www.stumptuous.com/lies-in-the-gym). Will put your mind at ease", 'Congratulations on getting rid of all the junk food and soda in your life, I am sure you will notice the benefits immediately. Keep choosing healthy foods making sure you hit your protein/carb/fat needs while eating smaller portions than you would before.  Cardio is a great activity for your heart and for your body but it is not required for weight loss; in any case, go out of your way to walk places rather than drive. Dieting and fitness do not have to be immediate changes; continue to slowly change little aspects and it will eventually add up', 'Congrats for deciding to get healthy!  1) How much cardio should you do? How much can you do? See how long you can run on that treadmill going 5 miles per hour. Then slow it down to a walk. Then see how long you can do it again! Get comfortable jogging. If you decide you like it, try a couch to 5k. Seriously, google "Couch to 5k."  2) Great, you have some hand weights! Here are some things you can do:  Arms: Dumbbell curls: http://www.youtube.com/watch?v=ggSmQiAfyd0 Legs: Lunges: http://www.youtube.com/watch?v=dJ95qwNaD78 Butt: Body weight squats: http://www.youtube.com/watch?v=Up_48p-BMug  Also something to remember: when it comes to overall fitness, don\'t think of anything as a "problem area." No amount of ab exercise can get you washboard abs until you have basically no body fat. Concentrate on making yourself stronger, and you\'ll be happy with the results. ', 'The easy answer is "whatever exercise you will actually *do* 5 days a week."  The hard answer is to do a 1-2 hours of strength training a week, 1-2 hours of cardio a week, and an 0.5-1 hours of flexibility training a week. It\'s up to you whether you want to make each of those it\'s own workout or whether you prefer to do some of everything each day.', "Sorry, when you start to loss weight, you're going to loss your D's.  They'll probably become C's.  Which is still glorious :P  For cardio I'd just do a 20 minute walk 5-7 days a week to start.  Strength I'd do bodyweight exercise, I like convict conditioning personally.. boils down to progressive movements for squats, push ups, pull ups, leg raises, etc.  Lowest step is manageable for almost anyone.  Only equipment necessary is a pullup bar, could get a door mount one.", "There's a good selection of starter stuff in the [FAQ](http://www.reddit.com/help/faqs/Fitness#Exercise1), I'd start there and look through the recommendations and see if anything sounds interesting/fun.", 'Most girls I see at the gym working on their cardio are on the elliptical machines. They make you sweat your ass off and render a nice butt in return.  ', 'Why use a treadmill when you can go for a run. I bet if you run around where you live you will be able to find some interesting places / routes.  BONUS FUN: running *really* late at night (even better after rain), everything is quiet, still and cool.... aaaah  Best rule for foods ever: the less processed it is, the better it is.  Obviously not hard and fast but it helps you when your in the supermarket choosing what to have for dinner.', '**READ THE FAQ**   Mods need to make the FAQ a permanent post. Most self.fitness posts are sounding more and more like Livejournal entries.', "You can not lose fat in just the areas you want. I've lost about 15 pounds in the past few months and went from a DD to a D. If it is going to happen, it will happen.  ", 'Deadlift. Makes me sad that I failed a 1 rep 10kg less than my PR because I was sleep deprived and had been away from the gym for a few days.', 'All exercises involving dumbbells are my favorite, except anything involving legs.', "Cleans by far are my favorite, just because it can be used for everything. Great movement overall.   I can't say I really hate anything in the gym accept steady state cardio...I've learned to love all kinds of exercises and their purposes...cardio though...unless it's sprints, it gets so boring to me. ", 'used to hate squat and deadlift, dont know why i love em now', 'Kayak row gives me an actual high if I push myself right', 'Deadlift.', "If I love any exercise, I stop doing it, as it tends to mean it's not very effective.   The exercise I hate the most is safety squat bar squats, and holy crap are they effective.", 'pull ups', 'standing barbell OHP, weighted pullups on good days, db lateral raises.', 'Deadlifts is the only correct answer, but I also find standing ab whele rollouts makes me feel like a beast', "I've been loving front squats lately. They just feel so right! And deadlifts of course. Bent over rows with snatch grip. And incline bench. ", 'Smith machine quarter squats.', 'Freestanding Handstand holds. ', 'Deadlift and Rows. ', 'Pull ups. I can barely do two, but I feel a boss when do them. =P', 'Weighted pull-ups and dips.', 'Favourite exercises are squats and OHP ', 'Heavy squats absolutely suck but I feel like $1,000,000 if I nail a new 5x5 PR', 'Read the wiki.   Work out TDEE, eat accordingly, start a compound total body routine.', "I'd say two instrumental things that got me into a regular routine for working out were   1. making it part of my every day routine. I go to class, then I go to the gym, then I go home for dinner etc. When it's part of your routine it's harder to not workout and get out of your program. (also seems to work best for me if I go somewhere for a workout, as working out at home always becomes me skipping workouts regularly) 2. Do research on what to do for a program. If you don't know what you're doing it's easier to just not do anything.  Get on a schedule and try to keep up with it for a while and you'll see great results, especially if you're eating well.", 'This submission has been removed.    Violation:    ## [Rule 0: No threads that are answered by the Wiki, Searching r/Fitness, or Google](https://www.reddit.com/r/Fitness/wiki/rules#wiki_rule_.230)    ##### Please review the [Getting Started page](https://thefitness.wiki/getting-started-with-fitness/) from our Wiki.    ***    IMPORTANT: **This is not an automatic removal by a bot.** Your thread was reviewed and removed by command from a human moderator. Please click the link above and read the full text of the rule if you have any questions about the removal.', 'Something else to keep in mind (this comes from judging by your name): there are no "hacks" or cheat codes or short cuts. You have to find a program that seems like you\'re willing to stick to and do it. If you switch programs a few times or make minor tweaks, that\'s fine. Just know that consistency and effort are the only 2 things that will make the most difference.', '1.) determine what your goal is.  2.) find a program that will lead you to that goal  3.) Commit to it, and begin.  Finally, realize that this is a marathon, not a sprint. Your current situation did not develop overnight, so neither will any positive changes. View your commitment as a lifestyle change, not a temporary fix.', 'This thead has been locked. Reason: code1554-savage-22', 'Read the wiki... ', 'The Bro Split', "Read the wiki. If you can't even put in the work to do the initial research, then you're not ready at all. Come back when you're older."]
    comments_df = pd.DataFrame({'body': comments})
    loader = DataFrameLoader(comments_df, page_content_column="body")
    documents = loader.load()

    text_splitter = CharacterTextSplitter(chunk_size=10000, chunk_overlap=500)
    texts = text_splitter.split_documents(documents)
    docs = []
    counter = 0
    content = ""
    content_array = []
    for text in texts:
        if counter < 20:
            counter += 1
            content += text.page_content
            content += '\n'
        else:
            counter = 0
            content_array.append(content)
            content = ""
            content += text.page_content
            content += '\n'
    content_array.append(content)
    docs = [Document(page_content=t) for t in content_array]
    # query langchain for questions to get the desired result
    prompt_template_p1 = f"Given the following Reddit comments in response to the search query {query}, "


    #prompt_template_p2 = """identify the most frequently occuring key claims (2-6 claims) found in the comments and output it as a list of key claims."
    #prompt_template_p2 = """list the key claims for each comment in the format [claim1,claim2, etc.]"""
    #prompt_template_p2 = """list the key claims for each comment. If a
    #comment makes an irrelevant or troll claim in response to the question ignore it.

    prompt_template_p2 = """identify the most frequently occurring key claims (2-6 claims) found in the comments
    that directly answer the search query. Output a list of key claims."
    
    ```{text}```
    
    """

    prompt_template_formatted = prompt_template_p1 + prompt_template_p2
    PROMPT = PromptTemplate(template=prompt_template_formatted, input_variables=["text"])



    combine_template_p1 = f"Given the relevant key claims extracted from reddit comments made in response to the search query {query} , "
    combine_template_p2 = """consolidate and identify the relevant and unique claims that are mentioned most frequently in the list of claims.
    Provide a list of between 1 to 7 key distinct and unique claims that directly answer the question and rank them based on frequency of occurrence.
    Your output should be an array of strings where each item in the array is a string of the claims you found. An example is below. 
    Only return the array and nothing else
    
    EXAMPLE OUTPUT STRUCTURE:
        ["Claim 1", "Claim 2", ...]
        
    Claims extracted from Reddit Comments in the OUTPUT STRUCTURE given (only the array and nothing else):
    ```{text}```
    """

    combine_template =  combine_template_p1 + combine_template_p2

    combine_prompt = PromptTemplate(
        input_variables=["text"],
        template=combine_template,
    )

    chain = load_summarize_chain(llm=OpenAI(temperature=0, batch_size=60) ,chain_type="map_reduce", map_prompt=PROMPT, combine_prompt=combine_prompt)
    llm_resp = chain.run(docs)
    start_array_index = 0
    end_array_index = 0

    for i in range(len(llm_resp)):
        if llm_resp[i] == '[':
            start_array_index = i
        if llm_resp[i] == ']':
            end_array_index = i
    llm_resp = llm_resp[start_array_index + 1: end_array_index]

    llm_resp_array = llm_resp.split(',')
    print(get_key_claims_hash(comments, llm_resp_array))
