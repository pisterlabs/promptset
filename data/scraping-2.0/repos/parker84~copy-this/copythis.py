from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from dotenv import find_dotenv, load_dotenv
from langchain.prompts import PromptTemplate
import os
load_dotenv(find_dotenv())

# --------------constants
CHAT_MODEL = "gpt-3.5-turbo-16k"


class CopyThis():

    def __init__(self, logger) -> None:
        self.example_copy = {}
        for fname in os.listdir('./data/example-copy'):
            with open(f'./data/example-copy/{fname}') as f:
                self.example_copy[fname.split('.')[0]] =  f.read()
        self.logger = logger
    
    def run(self, input_copy):
        self.logger.info('Running CopyThis')
        make_it_sing_version = self.make_it_sing(input_copy)
        aida_version = self.format_w_aida(make_it_sing_version)
        slippery_slope_version = self.slippery_slope(aida_version)
        simplify_version = self.simplify(slippery_slope_version)
        storytelling_version = self.storytelling(simplify_version)
        write_like_you_speak_version = self.write_like_you_speak(storytelling_version)
        longer_copy_version = self.longer_copy(write_like_you_speak_version)
        # output_copy = self.about_us(output_copy)
        yes_ladder_version = self.yes_ladder(longer_copy_version)
        emotion_version = self.emotion(yes_ladder_version)
        output_copy, final_prompt = self.combine_copy_together(
            original_copy=input_copy,
            make_it_sing_version=make_it_sing_version,
            aida_version=aida_version,
            slippery_slope_version=slippery_slope_version,
            simplify_version=simplify_version,
            storytelling_version=storytelling_version,
            write_like_you_speak_version=write_like_you_speak_version,
            longer_copy_version=longer_copy_version,
            yes_ladder_version=yes_ladder_version,
            emotion_version=emotion_version
        )
        return output_copy, final_prompt

    def make_it_sing(self, input_copy):
        self.logger.info('Giving it rhythm')
        llm = ChatOpenAI(model_name=CHAT_MODEL, temperature=0)
        prompt = PromptTemplate(
            input_variables=["example_copy", "input_copy"],
            template="""
            You are an expert at writing sales copy. Specifically Sales copy that has rythm.

            I'm going to give you some sales copy and you're going to drastically improve it so it actually sells.

            To help you understand what your sales copy should sound/look like I'm going to share with you a sales page from Louis C.K.

            Remember the best sales copy is like a slippery slope that you slowly fall down. The words suck you right in.

            Give it rhythm!

            His sales page was a simple blog post. And it was a masterpiece in writing in a simple and engaging way in order to sell!
            - Dry yet intriguing.
            - He uses simple language.
            - The flow is beautiful. His writing sings!
            - He uses self-deprecation and honesty that sells
            - And his rhythm is poetic.

            Here's his post: 
            "{example_copy}"

            Now I want you to this sales copy and make it just as good (if not better) than that:
            "{input_copy}"

            Return the improved version of this sales copy with proper markdown formatting (bolding / headers / bullets / ...)
            Because this is markdown - output dollars signs ($) as \$.
            Be sure to blow my socks with how many sales this is going to generate.
            Only output the new copy - no commentary around it.
            """,
        )
        chain = LLMChain(llm=llm, prompt=prompt)
        output_copy = chain.run({"example_copy": self.example_copy['01-louis-ck'], "input_copy": input_copy})
        self.logger.info(f"Output Copy: \n{output_copy}")
        return output_copy
    
    def format_w_aida(self, input_copy):
        self.logger.info('AIDA')
        llm = ChatOpenAI(model_name=CHAT_MODEL, temperature=0)
        prompt = PromptTemplate(
            input_variables=["example_copy", "input_copy"],
            template="""
            You are an expert at writing sales copy. Specifically formatting sales copy to take advantage of AIDA (attention, interest, desire, action).

            I'm going to give you some sales copy and you're going to drastically improve it so it actually sells.

            To help you understand what your sales copy should sound/look like I'm going to share with you a letter Gary Halbert 
            (arguably the greatest sales copywriter ever) wrote to his son on how to be a great copywriter.

            AIDA stands for attention, interest, desire, action.

            This is the most common format for sales pages.

            The copy you write, should be in this format.
            (but do not actually label it as such)

            Here's his letter: 
            "{example_copy}"

            Now I want you to this sales copy and make it just as good (if not better) than that:
            "{input_copy}"

            Return the improved version of this sales copy with proper markdown formatting (bolding / headers / bullets / ...)
            Because this is markdown - output dollars signs ($) as \$.
            Be sure to blow my socks with how many sales this is going to generate.
            Only output the new copy - no commentary around it.
            """,
        )
        chain = LLMChain(llm=llm, prompt=prompt)
        output_copy = chain.run({"example_copy": self.example_copy['02-aida'], "input_copy": input_copy})
        self.logger.info(f"Output Copy: \n{output_copy}")
        return output_copy
    
    def slippery_slope(self, input_copy):
        self.logger.info('Slipper Slope')
        llm = ChatOpenAI(model_name=CHAT_MODEL, temperature=0)
        prompt = PromptTemplate(
            input_variables=["example_copy", "input_copy"],
            template="""
            You are an expert at writing sales copy. Specifically writing the slippery slope.

            "He that has once done you a kindness will be more ready to do you another.”

            Ben Franklin said that. It's such a famous quote, psychologists named it The Ben Franklin effect.

            It means a person who has already performed a favor for you is more likely to do you another favor. They’re bought in.

            For example, a politician wants huge signs planted in the front lawns of everyone in town.

            The best way to achieve this is asking homeowners to put a small sign in their window.

            A week later, they come back asking to put a big sign in the yard. People are more likely to say yes (this is a famous, proven case study).

            This idea of getting people bought in works with copywriting.

            Copywriters call this “the slippery slope.”

            Your goal as a copywriter is to get readers to fall down a slippery slope.

            The goal of your first sentence: get people to read the second sentence.

            The goal of the third: get people to read the fourth. And so on.

            Readers will keep reading in proportion to the amount they’ve already read. And the more they read, the more they’ll agree with you.

            I'm going to give you some sales copy and you're going to drastically improve it so it actually sells.

            To help you understand what your sales copy should sound/look like I'm going to share with you a 
            blog post from Joel Spolsky. He founded StackOverflow, Trello, and a few other huge startups.

            Here's his blog post: 
            "{example_copy}"

            Now I want you to this sales copy and make it just as good (if not better) than that:
            "{input_copy}"

            Return the improved version of this sales copy with proper markdown formatting (bolding / headers / bullets / ...)
            Because this is markdown - output dollars signs ($) as \$.
            Be sure to blow my socks with how many sales this is going to generate.
            Only output the new copy - no commentary around it.
            """,
        )
        chain = LLMChain(llm=llm, prompt=prompt)
        output_copy = chain.run({"example_copy": self.example_copy['03-joels-post'], "input_copy": input_copy})
        self.logger.info(f"Output Copy: \n{output_copy}")
        return output_copy
    
    def simplify(self, input_copy):
        self.logger.info('Simplify')
        llm = ChatOpenAI(model_name=CHAT_MODEL, temperature=0)
        prompt = PromptTemplate(
            input_variables=["example_copy_dilbert", "input_copy", "example_copy_stephen"],
            template="""
            You are an expert at writing sales copy. Specifically simplifying sales copy.

            Warren Buffet writes an annual shareholder letter explaining what’s happening in Berkshire Hathaway’s businesses.

            It's pretty complicated stuff. Imagine trying to explain how a $100 billion insurance company works!

            However, with 50 years worth of letters, Buffett’s writing has stayed simple. Very simple. He averages less than 25 words per sentence and writes at a 4th grade reading level.

            Why?

            Because simple works.

            But there's bad news.

            You likely don’t write simply. Most don’t. Schools train us to write flowery language and minimum word counts.

            That doesn't mean long pages are bad. The opposite is true (in most cases). Long copy sells better than short copy.

            However, you must write simply and your copy should be just long enough to cause the reader to take the action that you request.

            If you need long copy in order to sell effectively, use long copy.

            Some products, like a can of Coke, don't need long copy. Other stuff, like a $1,000 course taught by a no-name person, needs a lot of words.

            The sales page for Amazon's Kindle, one of the most well known products and brands in the world, is ~1,400 words. And that doesn't include the tens of thousands of reviews, pictures and videos.

            So, remember: your copy needs to be long enough to sell and always written in simple language.
            
            I'm going to give you some sales copy and you're going to drastically improve it so it actually sells.

            To help you understand what your sales copy should sound/look like I'm going to share with you 2 examples:

            Here's an example from dilbert's blog:
            "{example_copy_dilbert}"

            Here's an example from Stephen King’s Toolbox:
            "{example_copy_stephen}"

            Now I want you to this sales copy and make it just as good (if not better) than those:
            "{input_copy}"

            Return the improved version of this sales copy with proper markdown formatting (bolding / headers / bullets / ...)
            Because this is markdown - output dollars signs ($) as \$.
            Be sure to blow my socks with how many sales this is going to generate.
            Only output the new copy - no commentary around it.
            """,
        )
        chain = LLMChain(llm=llm, prompt=prompt)
        output_copy = chain.run({"example_copy_dilbert": self.example_copy['04-dilbert'], "input_copy": input_copy, "example_copy_stephen": self.example_copy['04-stephen-king-toolbox']})
        self.logger.info(f"Output Copy: \n{output_copy}")
        return output_copy

        
    def storytelling(self, input_copy):
        self.logger.info('Storytelling')
        llm = ChatOpenAI(model_name=CHAT_MODEL, temperature=0)
        prompt = PromptTemplate(
            input_variables=["example_copy", "input_copy"],
            template="""
            You are an expert at writing sales copy. Specifically selling through storytelling.

            I'm going to give you some sales copy and you're going to drastically improve it so it actually sells.

            To help you understand what your sales copy should sound/look like I'm going to share with you 
            one of the most famous sales letters ever written.

            It doesn’t get any better than this.

            This advertisement ran continuously, with minor changes, for thirty-one years from 1975 to 2003. And this style of storytelling was used since the early 1900s.

            This letter sold over $2 billion worth of subscriptions for the WSJ. That’s roughly $200,000 a day for over 25 years. Other than minor price edits, it ran mostly unchanged the entire time.

            Just two pages long, it was the workhorse for the Wall Street Journal.

            Here’s what you should learn:

            Storytelling is an amazing way to grab people’s attention and get them interested. Stories are the best way to create a slippery slope.
            Starting a sales letter with a story…and THEN selling…is a great way to sell. Storytelling can be the Attention and the Interest in AIDA.
            After storytelling, explain the facts of the offer (Desire) and tell the Action.

            Here's the sales letter: 
            "{example_copy}"

            Now I want you to this sales copy and make it just as good (if not better) than that:
            "{input_copy}"

            Return the improved version of this sales copy with proper markdown formatting (bolding / headers / bullets / ...)
            Because this is markdown - output dollars signs ($) as \$.
            Be sure to blow my socks with how many sales this is going to generate.
            Only output the new copy - no commentary around it.
            """,
        )
        chain = LLMChain(llm=llm, prompt=prompt)
        output_copy = chain.run({"example_copy": self.example_copy['05-wsj'], "input_copy": input_copy})
        self.logger.info(f"Output Copy: \n{output_copy}")
        return output_copy
    
    def write_like_you_speak(self, input_copy):
        self.logger.info('Write Like You Speak')
        llm = ChatOpenAI(model_name=CHAT_MODEL, temperature=0)
        prompt = PromptTemplate(
            input_variables=["example_copy", "input_copy"],
            template="""
            You are an expert at writing sales copy. Specifically at writing in plain english like people speak.

            I'm going to give you some sales copy and you're going to drastically improve it so it actually sells.
            
            Jason Fried founded a popular email business called HEY.
            On the sales homepage, he wrote a wonderful letter from his personal perspective on why HEY is different and worth the user’s money.

            So, you're going transform this sales copy to: Write like you speak.

            Great copywriting, or really any writing for that matter, should use everyday language that sounds like it's coming straight from your mouth.

            Your goal is to use the written word to get your ideas into your reader’s brain and get them to take the action you want them to take.

            This means that proper grammar isn’t the most important part of your writing (a lot of people don’t realize this). And sounding smart and important with big words - that’s for losers.

            To help you understand what your sales copy should sound/look like I'm going to share with you 
            an example sales homepage from an email business called HEY. 

            Here's the sales homepage: 
            "{example_copy}"

            Now I want you to this sales copy and make it just as good (if not better) than that:
            "{input_copy}"

            Return the improved version of this sales copy with proper markdown formatting (bolding / headers / bullets / ...)
            Because this is markdown - output dollars signs ($) as \$.
            Be sure to blow my socks with how many sales this is going to generate.
            Only output the new copy - no commentary around it.
            """,
        )
        chain = LLMChain(llm=llm, prompt=prompt)
        output_copy = chain.run({"example_copy": self.example_copy['06-basecamp'], "input_copy": input_copy})
        self.logger.info(f"Output Copy: \n{output_copy}")
        return output_copy

    
    def longer_copy(self, input_copy):
        self.logger.info('Long Form Copy Sells')
        llm = ChatOpenAI(model_name=CHAT_MODEL, temperature=0)
        prompt = PromptTemplate(
            input_variables=["oyster_example_copy", "matterhorn_example_copy", "bond_example_copy", "input_copy"],
            template="""
            You are an expert at writing sales copy. Specifically at long form copy that sells.

            I'm going to give you some sales copy and you're going to drastically improve it so it actually sells.
            
            Here is a quote from you:
                "I once made a bet with a designer co-worker of mine. We were designing a landing page for a $300/year product and we made a bet on which of us could get a higher conversion rate.

                Her page was typical of most San Francisco startups. Bright colors. Cute but vague phrases and tag lines. Short copy. Very “pretty.”

                Mine was only copy. Black text and a white background. It was ugly. But it told a story, was interesting, addressed objections, and informed the buyer.

                She made fun of me. She said it looked like stupid Internet market-y stuff and that no one would buy it.

                We spent $1000 over 2 days on Facebook ads and drove the traffic equally to each page.

                My page made $3,000. Hers, only $300.

                Most people think exactly like my co-worker!

                But, they’re wrong!

                Why?

                Because your copy needs to be long enough to achieve your desired action.

                Sometimes that means long copy, sometimes that means short. This is hard for people to understand because when looking at different landing pages people will skim a long one and say “this is too long, no one will read this!”

                Nonsense.

                Your goal isn’t to get everyone to read and buy. It’s to get the right person to read and buy.

                Imagine a sales page that promises to alleviate pain caused by a rare health disorder. Most people don’t care about that. They’ll ignore. But the potential buyers, so long as you create a slippery slope, will read every word. And buy.

                So here’s a terribly oversimplified rule to live by (and eventually break)

                Longer copy is better for unknown, expensive products.
                Shorter copy is better for well known or cheaper products.
                For example, imagine you're selling a $5,000 “increase your vertical jump” program.

                A link to a buy now button is pretty much all that’d be needed if Michael Jordan were the creator!

                Now compare this to if I, a 30-something thicc boy dork, sold the same thing.

                One might need a little more sales copy to convince people to buy…

                Another example is the Kindle sales page. It has about 2,500 words. This doesn’t include all the pictures, videos, or 50,000 reviews.

                We’re talking about Amazon here. And the Kindle. Both of which just about everyone in America knows. And yet, there’s a lot of copy needed to sell it.

                The one takeaway you should have today is that your copy needs to be long enough to achieve your desired action. Not shorter or longer. The only goal is to get people do act. That’s it."

            To help you understand what your sales copy should sound/look like I'm going to share with you 3 
            excellent example ads from Rolex:

            The Magic Of The Rolex Oyster Case - A simple and effective way of explaining to anyone what’s special about the oyster case and why you should buy a Rolex Submariner Date:
            "{oyster_example_copy}"
            
            Exploring The Matterhorn - Created in 1966, this ad could still work today.
            "{matterhorn_example_copy}"

            The Influence Of James Bond - Probably the most iconic Rolex ad ever created for the most iconic luxury watch in history.
            "{bond_example_copy}"

            Now I want you to this sales copy and make it just as good (if not better) than that:
            "{input_copy}"

            Return the improved version of this sales copy with proper markdown formatting (bolding / headers / bullets / ...)
            Because this is markdown - output dollars signs ($) as \$.
            Be sure to blow my socks with how many sales this is going to generate.
            Only output the new copy - no commentary around it.
            Don't include any copy that doesn't need to be there even if it is in the original version.
            Delete any copy that doesn't need to be there.
            """,
        )
        chain = LLMChain(llm=llm, prompt=prompt)
        output_copy = chain.run({
            "oyster_example_copy": self.example_copy['07-rolex-ad-oyster'], 
            "matterhorn_example_copy": self.example_copy['07-rolex-ad-matterhorn'],
            "bond_example_copy": self.example_copy['07-rolex-ad-bond'],
            "input_copy": input_copy
        })
        self.logger.info(f"Output Copy: \n{output_copy}")
        return output_copy
    
    def about_us(self, input_copy):
        self.logger.info('About Us')
        llm = ChatOpenAI(model_name=CHAT_MODEL, temperature=0)
        prompt = PromptTemplate(
            input_variables=["hubspot_example", "input_copy", "pen_example"],
            template="""
            You are an expert at writing sales copy. Specifically at writing about us pages and other forgotten pieces of copy.

            I'm going to give you some sales copy and you're going to drastically improve it so it actually sells.
            
            Groupon used to have the most amazing email unsubscribe page I’ve ever seen.

            When Groupon users opted out of their email list, they were sent to a page that read:

            “We’re sorry to see you go. How sorry? Well, we want to introduce you to Derrick - he’s the guy who thought you’d enjoy receiving the Daily Groupon email.”

            Then there was a video of someone fake throwing hot coffee into Derrick’s face and yelling at him—followed by a “make Derrick happy again an re-subscribe” button.

            How beautiful is that?

            As Groupon grew and appealed to more people, they toned it down a bit. That’s fair. But this premise could still work.

            Often, the 2nd most viewed page on a website is the “About Us” page. And yet, when companies think about sales copywriting, they only think about landing pages.

            The About Us page, unsubscribe page, product descriptions, and a few other categories are what I call “forgotten copy.”

            This copy is often forgotten. And most of the time the copy is (gasp!) default copy from the software. Like a generic unsubscribe page from Mailchimp.

            You should know by now that great copywriting grabs your attention and gets you to fall down the slippery slope.

            That happens through the AIDA formula, storytelling, and conversational writing.

            But grabbing people’s attention is also done by making the forgotten copy wonderful.

            To help you understand what your sales copy should sound/look like I'm going to share with you 
            2 examples.

            The first is the welcome email I wrote for The Hustle. When users signed up for The Hustle, they got this email. 
            Dozens of blogs wrote about it. Admittedly, it was heavily inspired by another company called CD Baby:
            "{hubspot_example}"

            The second is the About Us page for a pen business. The author is a woman who used to work for me. This company sells a $150 pen.
            "{pen_example}"

            Now I want you to this sales copy and make it just as good (if not better) than that:
            "{input_copy}"

            Return the improved version of this sales copy with proper markdown formatting (bolding / headers / bullets / ...)
            Because this is markdown - output dollars signs ($) as \$.
            Be sure to blow my socks with how many sales this is going to generate.
            Only output the new copy - no commentary around it.
            """,
        )
        chain = LLMChain(llm=llm, prompt=prompt)
        output_copy = chain.run({"hubspot_example": self.example_copy['08-hustle'], "input_copy": input_copy, "pen_example": self.example_copy['08-tactile']})
        self.logger.info(f"Output Copy: \n{output_copy}")
        return output_copy

    
    def yes_ladder(self, input_copy):
        self.logger.info('Yes Ladder')
        llm = ChatOpenAI(model_name=CHAT_MODEL, temperature=0)
        prompt = PromptTemplate(
            input_variables=["example_copy", "input_copy"],
            template="""
            You are an expert at writing sales copy. Specifically at writing the "yes ladder".

            I'm going to give you some sales copy and you're going to drastically improve it so it actually sells.
            
            In 1966, two researchers wanted to answer a powerful question:

            “How can a person be induced to do something he would rather not do?”

            To start, they called two groups of people.

            They asked Group 1 if a researcher could go to their home and examine their household product brands and usage.

            They asked Group 2 an easier question: What type of household cleaning products do you use?

            Then, a week later, they called Group 2 back again and asked them if researcher could go to their home to examine their household product brands.

            The results: Group 2 was 135% more likely to let a researcher come to their home.

            This study isn’t a fluke. Researchers have studied this in different cultures and scenarios.

            And the takeaway is this: a person who’s said yes to a small request first is more likely to say yes to a bigger request later–versus asking the bigger request straight away.

            I call this the Yes Ladder.

            So how do you apply this with your copy?

            Get people to say yes early and often in your sales copy. And if you do that well, they’re more likely to say yes to your bigger request: buying your product.

            And how do you get them to say yes?

            Say it with me: E-M-P-A-T-H-Y.

            It’s called specific knowledge. When writing copy, take the the time to understand the customer and product.

            Know why people are frustrated with the problem and the competition. Put yourself in the customer’s shoes. Find out what motivates them, what they dislike, and the root emotion that’ll make them buy.

            This isn’t tough to do, but it does take time. Read Twitter. Look at message boards, Reddit, and reviews of other products. Cold call people.

            For today’s assignment, I’ve found one of my favorite modern examples of this.

            Marie Poulin sells a $799 course on how to use Notion (a powerful but complex todo list) to increase productivity.

            The sales page is beautiful, both in terms of copy and design.

            If you’ve ever used Notion, you’ll see how well she got you to say yes.

            Also notice: her page is full of testimonials and social proof. Over 35 examples. You can never have too much.

            To help you understand what your sales copy should sound/look like I'm going to share with you 
            the Notion Mastery Sales Page. See it here: 
            "{example_copy}"

            Now I want you to this sales copy and make it just as good (if not better) than that:
            "{input_copy}"

            Return the improved version of this sales copy with proper markdown formatting (bolding / headers / bullets / ...)
            Because this is markdown - output dollars signs ($) as \$.
            Be sure to blow my socks with how many sales this is going to generate.
            Only output the new copy - no commentary around it.
            """,
        )
        chain = LLMChain(llm=llm, prompt=prompt)
        output_copy = chain.run({"example_copy": self.example_copy['09-notion-mastery'], "input_copy": input_copy})
        self.logger.info(f"Output Copy: \n{output_copy}")
        return output_copy
    
    def emotion(self, input_copy):
        self.logger.info('Emotion')
        llm = ChatOpenAI(model_name=CHAT_MODEL, temperature=0)
        prompt = PromptTemplate(
            input_variables=["example_copy", "input_copy"],
            template="""
            You are an expert at writing sales copy. Specifically at making the readers feel emotion.

            I'm going to give you some sales copy and you're going to drastically improve it so it actually sells.
            
            Here's some context from a course on doing this for you to help you understand:
                
                "Some of you are going to be angry after reading this, and I understand why.

                You know those ads you see at the bottom of Fox News and CNN article?

                Often, they’re telling you about a stock that might be the next Amazon or another big promise like that.

                These are called advertorials.

                Basically, they look like a normal articles but they’re really sales pages. They tell you a story, get you bought in and at the bottom say something like “if you’re interested, click here to sign up.”

                I’ve got a friend who made $20 million in 2 weeks running a successful advertorial.​​​​​​

                There are two companies that are world class at this.

                The first is Motley Fool. They sell financial newsletters. And honestly, their stock picks are really great. But a lot of people hate their marketing. They love advertorials, many of which are pretty aggressive. But, from a business perspective, it works. Last I heard, they make $500m a year.

                And then there’s Agora. You’ve likely never heard of them. They own dozens of different brands, most of which sell financial newsletters but also supplements and all types of crazy stuff.

                They’re highly unethical, recently getting sued for selling a diabetes “cure”.

                But…again, they successfully use advertorials. They make around $1.5 billion (!) a year this way.

                So, why am I telling you this.

                Because advertorials work. You can do them ethically. They work because they make readers feel emotion. More on that in a second.
                And I’ve written a few of these. Some of which have made tens of millions of dollars.
                In 2016, while running The Hustle, one of our advertisers wanted to try an advertorial. So, using what I knew about copywriting, I wrote one.

                The result was this article: Getting Called “Sweetie” Helped this Entrepreneur Create a Multi-Million-Dollar Business​

                I wrote a story about how Kara Goldin was mocked by Coke execs when she founded a water company called Hint.

                The advertorial was a hit. Hint was making hundreds of thousands of dollars A DAY. We’d both post the article on our Facebook pages and they’d buy ads on Facebook and drive traffic to it. Hell, a Hint employee wrote about the success of it and it helped them keep their job!

                Because of the success of that advertorial, we did a few more.

                Another favorite was a story about my friend Jenn’s watch company, Linjer (here).

                But my point in all this is this:

                Great copywriting and sales pages have one thing in common…they make you feel a certain emotion!

                And advertorials are, in my opinion, a beautiful example of this principle.

                And I get it…often, these are totally scammy. But they don’t have to be. Mine weren’t. Another great example is how Chubbies does this. It's great!

                The takeaway is this: when writing sales pages, advertorials, or anything else meant to sell…use emotion.

                Write your copy like you’re writing an episode of Succession. Have a main character, a hero’s journey, a plot, ups and downs.

                This will help you stand out, and most of all, get the sale."

            To help you understand what the sales copy you're about to write should sound/look like I'm going to share with you 
            and example - The Hint advertorial by The Hustle:
            "{example_copy}"

            Now I want you to this sales copy and make it just as good (if not better) than that:
            "{input_copy}"

            Return the improved version of this sales copy with proper markdown formatting (bolding / headers / bullets / ...)
            Because this is markdown - output dollars signs ($) as \$.
            Be sure to blow my socks with how many sales this is going to generate.
            Only output the new copy - no commentary around it.
            """,
        )
        chain = LLMChain(llm=llm, prompt=prompt)
        output_copy = chain.run({"example_copy": self.example_copy['10-emotion'], "input_copy": input_copy})
        self.logger.info(f"Output Copy: \n{output_copy}")
        return output_copy
    
    def combine_copy_together(
            self, 
            original_copy,
            make_it_sing_version,
            aida_version,
            slippery_slope_version,
            simplify_version,
            storytelling_version,
            write_like_you_speak_version,
            longer_copy_version,
            yes_ladder_version,
            emotion_version
        ):
        self.logger.info('Combining')
        llm = ChatOpenAI(model_name=CHAT_MODEL, temperature=0)
        prompt = PromptTemplate(
            input_variables=[
                'original_copy',
                'make_it_sing_version',
                'aida_version',
                'slippery_slope_version',
                'simplify_version',
                'storytelling_version',
                'write_like_you_speak_version',
                'longer_copy_version',
                'yes_ladder_version',
                'emotion_version'
            ],
            template="""
            You are an expert at writing sales copy.

            I'm going to give you some sales copy and you're going to drastically improve it so it actually sells.

            You've previously had multiple attempts at improving this sales copy each focused on improving it in a specific way.

            Now you're going to combine all these together into the best possible version.

            ## Here is the original version of the sales copy:
            {original_copy}

            ## Then you took that and made it sound better with rythm:
            {make_it_sing_version}

            ## Then you took that and applied the AIDA sales principles:
            {aida_version}

            ## Then you took that and made sure the reader falls down a slippery slope when reading:
            {slippery_slope_version}

            ## Then you took that and made sure it was simple:
            {simplify_version}

            ## You made it sound like a story:
            {storytelling_version}

            ## You made sure it was written like a person would speak:
            {write_like_you_speak_version}

            ## You ensured it was just the right lenghth:
            {longer_copy_version}

            ## You implemented the "yes ladder":
            {yes_ladder_version}

            ## And you ensured to invoke the emotion of the reader:
            {emotion_version}

            Now I want you to take the best of all these and output the best sales page.
            ever written to sell what was being sold in the original version.

            Return the final version of this sales copy with proper markdown formatting (bolding / headers / bullets italica / `` for numbers, ...)
            Because this is markdown - output dollars signs ($) as \$.
            Be sure to blow my socks with how many sales this is going to generate.
            Don't include any copy that doesn't need to be there even if it is in the original version.
            Delete any copy that doesn't need to be there.
            """,
        )
        final_prompt = f"""
You are an expert at writing sales copy.

I'm going to give you some sales copy and you're going to drastically improve it so it actually sells.

You've previously had multiple attempts at improving this sales copy each focused on improving it in a specific way.

Now you're going to combine all these together into the best possible version.

## Here is the original version of the sales copy:
{original_copy}

## Then you took that and made it sound better with rythm:
{make_it_sing_version}

## Then you took that and applied the AIDA sales principles:
{aida_version}

## Then you took that and made sure the reader falls down a slippery slope when reading:
{slippery_slope_version}

## Then you took that and made sure it was simple:
{simplify_version}

## You made it sound like a story:
{storytelling_version}

## You made sure it was written like a person would speak:
{write_like_you_speak_version}

## You ensured it was just the right lenghth:
{longer_copy_version}

## You implemented the "yes ladder":
{yes_ladder_version}

## And you ensured to invoke the emotion of the reader:
{emotion_version}

Now I want you to take the best of all these and output the best sales page
ever written to sell what was being sold in the original version.
        """
        chain = LLMChain(llm=llm, prompt=prompt)
        output_copy = chain.run({
            "original_copy": original_copy, 
            "make_it_sing_version": make_it_sing_version,
            "aida_version": aida_version,
            "slippery_slope_version": slippery_slope_version,
            "simplify_version": simplify_version,
            "storytelling_version": storytelling_version,
            "write_like_you_speak_version": write_like_you_speak_version,
            "longer_copy_version": longer_copy_version,
            "yes_ladder_version": yes_ladder_version,
            "emotion_version": emotion_version
        })
        self.logger.info(f"Output Copy: \n{output_copy}")
        return output_copy, final_prompt