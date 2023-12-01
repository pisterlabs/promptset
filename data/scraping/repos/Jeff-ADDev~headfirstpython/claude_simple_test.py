import os, requests, sys, argparse
import openpyxl
from typing import List
from openpyxl import load_workbook
from openpyxl.worksheet.table import Table, TableStyleInfo
from openpyxl.styles import Alignment
from openpyxl.utils import get_column_letter
from colorama import init, Fore, Back, Style
from dotenv import load_dotenv
from datetime import datetime
from anthropic import Anthropic, HUMAN_PROMPT, AI_PROMPT

start_time = datetime.now()
start_time_format = start_time.strftime("%m/%d/%Y, %H:%M:%S")
date_file_info = start_time.strftime("%Y_%m_%d")
create_date = start_time.strftime("%m/%d/%Y")

load_dotenv()
init() # Colorama   

claudekey = os.getenv("CLAUDE_KEY")

def terminal_update(message, data, bold):
    if bold:
        print(Back.GREEN + Fore.BLACK + Style.BRIGHT + f"  {message}: " + Back.BLUE + Fore.BLACK + Style.BRIGHT + f" {data} " + Style.RESET_ALL, end="\r")
    else:
        print(Fore.GREEN + Style.BRIGHT + f"  {message}: " + Fore.BLUE + Style.NORMAL + f" {data} " + Style.RESET_ALL, end="\r")

def try_claude():
    terminal_update("Trying Cluade", " - ", False)
    anthropic = Anthropic(
        # defaults to os.environ.get("ANTHROPIC_API_KEY")
        api_key=claudekey,
    )
    completion = anthropic.completions.create(
        model="claude-2",
        max_tokens_to_sample=20000,
        prompt=f"{HUMAN_PROMPT}" + 
        """
        When a boy is taken in for an operation, the surgeon says "I can not do the surgery because this is my son."
        How is this possible?
        """ 
        + f"{AI_PROMPT}",
    )

    print(completion.completion)

def count_tokens(text):
    terminal_update("Claude Counting Tokens", " - ", False)
    anthropic = Anthropic(
        # defaults to os.environ.get("ANTHROPIC_API_KEY")
        api_key=claudekey,
    )
    print(f"\nTokens - {anthropic.count_tokens(text)}")


def main(args):
    #if args.label:
    #    project_label = args.label 
    
    try_claude()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Create Excel Sheet for Project Reporting")
    parser.add_argument("-l", "--label", help="Label for the project")
    parser.add_argument("-c", "--console", help="Enable Console Output", action="store_true")
    args = parser.parse_args()
    main(args)
    text1 = """
        The followwing text is from a meeting of multiple people.  I am asking for a summary of all the main points talked about.  Please also provide some detail behind those points and any action items that were discussed. Here is the start of the text.  Please create the ouput in a markdown file format and include the markdown
        format in the text.
        This is the discovery with the R team in breaking down the work. Hey guys. Hey. Hey. Thanks Shane for joining us. No problem. Looking forward to it. Where's your smile man? You should be happy. I am happy. I was smiling earlier until you got on. That happens all the time when you see my face. All right. So one, I just wanted to put time together to let you guys talk about how you want to break down the work, how to go through stuff. Marco asked some great questions when we were going through onboarding post sprint planning, Julie. So, but also to real quick Shane. Something that came up, and this is more for Henry. Marco and I were talking about the 204. So Marco's interested in 204. Kurt's interested in it. I know Henry, you had mentioned it. Actually, Shane might want to be doing this. The Azure 204, that's the dev certification on Azure. Okay. I'm actually interested in it as well. Maybe we look at doing it together as a team. I mean, I went through some of the course material and looked at it, but I'm not big on certifications. I mean, I use all the tooling anyways. It's just looking at it. I mean, it's a good track though for what we're doing if somebody needs to look at all the tooling. And I'm not saying to go get the certification. I'm saying using it as a baseline for what we go through and understand and learn. Yeah, I could see it being a good, like a book club sort of thing too. If we wanted to, you know, there are probably books on that cert versus Microsoft videos. I mean, it depends if you like books or videos. Well, and I like Marco's. Oh, sorry. Go ahead, Shane. No, I think it's good to encourage people to get certs. I think it forces you to study and stay on track and have a goal to work towards because I know I was that way in Salesforce. Like, I didn't really start really learning until I forced myself to get the certs because it forced me to learn it. Like, have a goal, set a date, have a goal, just do it. Some people like it. I know Henry could go through it and learn and be fine without a cert too. So leave it open. I think you go through the material. One of the things Marco would suggest is like, hey, let's kick it off and start doing it together as a team. When the Python book club ends, then we can go and we've gone through stuff and we could actually be kind of a leadership piece or leaders within kicking it outside maybe. No, I think it's going to open up to all the developers or whoever wants to do it. I think it's good to get everyone on the same page with it. So anyways, I sidetracked the meeting. I'm sorry, I needed to eat. So mid-bite. Katie, yesterday in our offsite meeting and stuff, we were talking about the discovery and she's asking, you know, how we were going to go through and how we're going to kind of go through this. So I invited her to this meeting just to listen. And how we were trying to plan through and go through it. So that's why she's here. Unless you've got all the answers and got this thing all solved and built already for us, Katie. So guys, how do we want to go through the high level stuff with Julie again? You guys aware of with the Figma board? How do you guys want to start to talk about and breaking this down and attacking it? I think for me, what would be most useful is to understand what we first want to launch with. Like, what is the MVP? Right. So I think I remember hearing that that was review monitoring. But is that all of review monitoring? Is it a subset of review monitoring? Is it even still review monitoring? I don't know. But just like narrowing in on sequencing, getting an idea of like, I have a good sense from our meeting last, was it last week or whatever, whenever we had that, where we talked about like the Figma architecture diagram and things like that, of what the whole thing, what the whole vision is. But just the sequencing, I really don't have a good sense for. Yeah, the answer is yes. Review monitoring. We have a lot of that stuff in place already. So that gets us to doing some enhancements that are display enhancements for reviews that are on sites already. We already have a review feed in the marketing platform. So adding some filtering capabilities that will set us up for the response. The review widgets are new. So providing those widgets, the clients can implement them on their site. That's something new. And then we need thoughts on the switchover from BirdEye to Yet. So that would be part of that review monitoring. I will say that some of those conversations are still in progress with the product team. We just need to make sure that everybody's aligned, that we could do this review monitoring work and actually launch it in production. We're thinking that that's not going to be a problem, but we just have to kind of finalize that decision and that thinking. But when we talk about what's next after that, the review response and the generation, that's all part of that product relaunch, really. Monitoring is too, but those are new features and functionalities that we would be adding. So we would want to package that with the whole product relaunch. So we would need to look into some feature toggling and stuff on that stuff. Okay. So some of the more foundational stuff then is BirdEye to Yet. Sounds like maybe that's kind of one of the very first things, right? And then all of review monitoring piece, kind of like stable, doesn't necessarily need to go behind a feature flag because a lot of it's already out there. Enhancements. Sorry, Marco. Just the only thing to point out with that is that there is still a change. That review monitoring will be added to our local search plan. So that's where the alignment piece needs to come from product to make sure that that's communicated to sales, that we're all ready for that instead of just doing like a big bang product relaunch. Okay. That's kind of like the sensitivity around that. What plan was that being added to? Can you say that again? Local search advanced. Okay. Yeah. So right now, not all of our local search advanced clients have review marketing. If they do, they're paying extra for it. So this monitoring piece would become available to them as part of the local search advanced plan. Are we, from a product point of view, are we planning on having a big bang approach, like we shut off BirdEye and everyone's on Yext, or is it going to be like some people are going to be in BirdEye, some people are going to be on Yext for a transition period? Yeah, I think that's still part of what, that's part of the conversation I think we're still trying to figure out. Part of that could affect our design too, because if we decide we're going to pull out the database and make it all separate, we'll have to think about what systems currently interact with the CDR database in those reviews, and those will have to either have some sort of way to switch back and forth. It's going to add some complexity. I think we'll need to know that strategy sooner than later. Cool. Good point. I don't know if you know what AllTouch is, the current BirdEye solution out on the websites. We used some reviews out of there. Does TapClick interact with BirdEye in any way with that data? I'm not sure about that either, but that would be my only questions. Hope we can dig into that. Julia, are we mainly going to you as we start looking at discovery and questions, or do we think Zach as well, like both of you? Both of us. So, Zach, just so we all level set here, Zach really is the product manager for review marketing. He's the one who's been working on all of this. He's been, I'm just here to help get this moving through our team. He hasn't worked with any of you guys, so we're getting it moving through the team. And helping to keep it on task from a product perspective across all the teams. So, do you think he's going to want the level of visibility like what we were asking with Toscan, and like you like to see with the entries and exits on work? No, I don't think that he needs to have that.he should be able to count on people to run the work through the team. So he should be able to count on me for stuff that's running through the R team. He should be able to count on Justin for stuff for the platform and Mike for stuff for the strategy center. I think as far as getting started, I think we just need to go. I mean, we had this sprint fucked up for time, so Marco and Shane, you know, get together and just start going over those flows, the epics, see what's out there, start doing our research and start asking lots of questions as we dive into it and reach out to you guys. Yeah, I mean, from my perspective, I'll tell you guys what I'm looking for and what product is gonna be looking for. They're gonna be looking for a plan on how we get to launch for this. So some like milestone with target timing, same like, I know not everybody was involved with the site builder, but that kind of same thing, target dates, understanding that those targets can grow or shrink over time. But because this is a product relaunch, there's a whole bunch of stuff that's happening outside of the tech world too. There's sales updates, there's marketing updates, there's pricing, like everything. So without having targets to work toward, it would be really, really difficult. So I think like, I guess what I'm looking for next is some timings on when we think we would have some plans and approaches put together for keeping in mind we could do monitoring first and then roll into response and generation. Maybe those could be, Jeff and I know you're already kind of working toward that. Those could be our three big buckets and have some timings kind of associated with that. Well, and let's talk about timings on it because from a reporting out kind of aspect, the way that I plan to approach that, on the epics, there's a story point field. I'm gonna use that story point field as an estimate field. So like, for example, say we think something's gonna take two sprints. I'm gonna say roughly 30 points as a sprint. So just a big ass swag as to what you think it's gonna take for an epic to be completed. But then what I'm gonna watch is within that epic, as we define work and issues within an epic, that starts to narrow the cone of uncertainty for me as far as what that really looks like effort wise. Because then I can say, okay, we've got 15 epics in there and point wise, they add up to 150 points. Where does that kind of fit versus what we originally thought swag. It's really just a swag on the epic piece, that piece. So it's really kind of first phase estimate. And then as we define the work, you don't have to worry about the estimate piece. It's just defining the issues within there. And I'll be able to gather that information and say kind of how that's tracking. So are you looking at us to come up with a technical plan for sequencing how it, like the order that it makes sense to build this stuff in, and then kind of come up with a plan for a release plan after that in terms of what features we want to roll out when, or is there already an idea of what the minimum set of features looks like before we can put it in front of you, like clients? That question make sense? I'm not sure if I fully understand your question, but I think, oh, go ahead. Like earlier, you had mentioned that, like what we want to come out of this is like estimates in terms of like timeline of how long we think it's gonna take to roll out, right? But I mean, are you looking for the whole kit and caboodle, like everything in that diagram? Yeah. Okay. Yeah. Gotcha. Yeah, so, and keeping in mind, so, I mean, I don't think, Jeff, correct me if I'm wrong, I don't think you guys need to worry about the timelines. I think Jeff's probably gonna handle that, but he just needs your help with these like T-shirt size estimates on how long you think something might take. And then he can work his magical from there. He already has a framework set up. But then sequencing would be cool too, how you guys wanna come through that work and attack it. Okay. Just cause thinking through it, like we're gonna find opportunities to pilot bits and pieces of this as while other pieces are in flight, you know? So I don't know if we want to necessarily wait to release everything all at once, you know, or flip the switch all at once. I think that that goes back to what we were talking about around review and monitoring, but definitely we're thinking would be the thing that could go. Now, review response and generation are probably gonna be tied together. Just cause that really is part of the new tiers and stuff that are gonna be packaged and relaunched. So. Cool. At least that's the plan right now. Yeah, it depends. I mean, a lot of it probably depends on like whether we wanna collect early client feedback and do some like preview testing of certain features and things like that. I don't know. Henry, deep in thought? Yeah, I'm just thinking. My other board open looking at notes and stuff. I mean, we're at the really, we just need to get in and start diving through it. I mean, I don't have so many questions for my first six. I need to get started that day. I just need to do a deep dive on those flows and the ethics and then a lot of questions will start flowing out of that too, but. It's cool. I mean, we already got reviews on that. I mean, we already got reviews on that. I mean, we already got reviews on that. I mean, we already got reviews on that. I mean, we already got reviews on that. I mean, we already got reviews on websites. We've got reviews in the platform. So I mean, it feels relatively easy to just figure out where you guys switch them over. I mean, we could even start ingesting UX reviews and have stuff in parallel and have IDs and have a nice clean switchover plan too once we get into it. But I don't know enough to say yet. Yeah. It doesn't sound like it'd be that bad from just on the surface. I think one of the big wild cards is gonna be the UX work that needs to happen. As you have seen from that Sigma diagram, Alex's face is plastered all over that thing. And the UX is also gonna help inform what's needed from a backend perspective. So I don't know how Alex's work is getting managed, but we probably need to wrap some delivery timeframes around his work so that he can, like that's gonna automatically feed into the progress on the dev side. Yeah, so we'll probably involve him a lot too in some of these questions once we start getting into it almost, you know, almost a few years back to at least get a high level of what we think is gonna be on some of those. I mean, day to day, there's only so, UX will offer whatever they offer and we can't provide much beyond that from data and things, but. Do we want a confluence site for outstanding questions and stuff like that to get that out there and a central area to work from on this? Well, there's also all the ethics that are signed up. I don't know if you wanna drop the questions in those ethics. So it's okay. We just do it that way. Yeah. Now, the only other thing I would just point out is that those ethics were just asked, like, you know, creating ethics and trying to break down the work. Like, don't feel tied to that. New ethics can be spun up. If you're like, we don't need one for this, combine them. Like, we were just trying to make progress and be able to get you guys as much information as we could ahead of time. Are all the requirements in those along with that, like that flow, or there's still a lot of things that we think we'll discover requirement-wise as we get into it? Like how much have you guys typed into it? There's a lot of requirements in there and then there's some open questions, like, related to UX and stuff that need to be figured out. So, you know, they're all pretty much filled out. Now, I'm sure that there's plenty of holes. Yeah. As you know, Henry, as we work through things, so. Yep. I definitely have some questions related to specific ethics, yeah. Review widgets and things like that. Mm-hmm. If you guys create ethics, can you guys let me know? Because one of the things I do, I put a label in it so I can pull that stuff in and that's what I drive the reporting off of. Yeah. Or if you create an epic, just as simple as putting a label on it for review marketing and it'll grab my attention right away. So if you can remember that. Would it help if we put together, like, what are the things that we need to do with regularly, how can you implement reviews? That's really key. Not so much about work expenditure, but what are the ethics that you're trying to turn in into an action plan? Sure. Oh my gosh! What are the ethics? How does one go about that? However, what are the ethics? I don't understand that how do you takelike high-level estimates and plan for supporting bird eye ingestion and yaks ingestion in parallel versus just doing a clean cut over if that decision hasn't been made yet? Yeah, that would be helpful. That's a really good idea, Marco, because that might help make the decision. Right, yeah. Yeah, I like that. Great idea. Yeah, options are always really good. Okay. I know the other thing we talked about was other teams kind of, you know, like if there's a review marketing UI, Fusion or somebody might pick that up. Same with, well, I guess it's not Fusion, whatever the squads are. Same with strategy center staff. So I don't, I don't know how you guys get involved, involve those tech leads or something to get the t-shirt sizes. I don't know how that works. To get, well, yeah, we would need to pull in somebody from those teams probably into these planning meetings to talk about those pieces or like a subset of it. Just ask the team to send a representative, like who feels comfortable estimating that. Yeah. Have you all done story mapping before? I'm just wondering if that might be an approach to refining this, you know, pulling in tech leads from Pixel Perfect, Squid Squad, Data, and just going through the whole thing and story mapping it out, you know, and then you can start figuring out what could be in the different releases. Yeah, I think I have done this as a part of like Safe Agile PI planning, which is like, are you familiar with like Safe Agile? It's just like, it's like Waterfall Agile. It's like Enterprise Agile, where every six sprints you... I am certainly not suggesting that at all. No, no, no, I'm just saying it reminds me of it where you like, you're basically sticky noting and kind of like trying to, I don't know, maybe I'm missing, I don't know, maybe I'm not understanding exactly what you mean by story mapping, but it looks a lot like it to me. I don't think we need to get the other teams involved yet. I mean, we don't know for sure who's going to do the work, so I think we, if it's another team and they bring it in, they estimate it. I'm just thinking from a t-shirt size perspective, Shane, because we're still looking for that high-level plan or view. Well, yeah, but I think like between Marko knows the platform pretty well and Salesforce, I know Salesforce, so I think we could do a t-shirt size on anything outside of... Yeah, we'll ultimately have stories for that. I mean, it's only a couple UIs from what I've seen. I mean, there's not a ton. I mean, there's, you know, the listing the reviews and responding, so on. There's not a ton there. I mean, cards for those, we're going to have the APIs and interfaces for that. Salesforce, there's going to be some mechanism to subscribe to that queue to create cases and stuff, so... If we have more questions, we need another tech lead to just bring them in for a conversation. T-shirt size, I don't think it has to be that formal or complicated. Yep. I love not complicated. How do you guys want to structure going through the work? Do you want me to set stuff up to get all of us together, or do you guys want to do it yourselves? Just leave us alone, let us start working on it. I can do that. That's why I said you got the whole sprint. Yeah, I think we just start with ARR 2.470, the reviews reset, update the data source ingestion process, start breaking down and estimating that, two different approaches, whether we need to support in parallel or not, and then we just follow the logical lines down and start. Yeah. Sure. One quick thing related to that, Marcel, so the review assessment, 2472, that's in the monitoring flow, but I could see that as something that could come a little later, just as an FYI, if we're talking in sequencing the work, because that's really going to be tied to the other two tiers, the new tiers, the local search premium and the managed service tier, so that could come a little later, like be bundled up with review response and generation. Okay, yeah. I mean, we'll just try to estimate and come up with a plan as closely as possible and figure out how we want to sequence, I guess. For the review widget stuff, 2473, there was, we talked about the embed code. Is that for clients that have microsites and stuff built by us? I mean... That's for clients that have their own website. Okay, that's what I thought, yeah. Because otherwise, what would that look like if they have a website? Do they just configure the widget and click a button and work with their strategist? Do they not even configure the widget themselves? If they have a microsite or website, that is an existing process. It would get turned on on that. It would be part of the monitoring package and be turned on on their website or microsite with us. So sometimes, I guess the biggest use case would be if a client has a microsite with us, but then they also have their client-owned website, then that might be the case where they would also display those widgets on their client-owned website. I don't believe that there's any clients that have a website with us, as well as their own client-owned website. But in the case that they might have a microsite with us and then have their own website, they would just leverage the styling that we would set up in the site builder. But I think that probably the most often use case would be that they would maybe need to configure some colors, and that's all in the Epic, and they would have to do that from the marketing platform. Gotcha. Yeah, that one, I definitely have a lot of questions around, and I think a lot of it will depend on how customizable it is, what kind of features need to be in place. Does it automatically pull a theme from their microsite theme if they have one? It's stuff like that that's really going to probably impact the t-shirt size. Yep. Yeah, so if you're going to that Epic, we have some details on there, so as you have the questions, just reach out, and I'll help you answer them. Okay. Yep. Yeah, so you guys can pull Zach and I into conversations or schedule a meeting with us, and we can go through your questions as you have them. Okay. Open up the chat, and we can try to do that too. Sorry, go ahead. I may have missed this, but what format are we using? Are we just going to start setting up essentially, or? Yep. Okay. And then what I'll do, I'll get... Sorry, go ahead, Julie. I was just going to say the Epics I already created are all linked from the Figma file too. And also too, like this reporting summary doc that I'm talking about that I'll drive from, I'll find a location to start putting that stuff out there, so we can see a picture. You guys can see all the Epics, all the issues, and you can have access to what I'm using as well. The other thing I just want to throw out here, I know that Zach has mentioned this before too, but there's a dream that we would have some form of this ready for the getbacks that happen at the end of March. Shit, we'll have it ready by Christmas. What are you talking about, Julie? That's great. That'd be even better. But I think some UI stuff might hold you up there. But yeah, so anyhow, I just wanted to throw that target out. What else was I going to say? I can't remember. I don't know why. Oh, I know what it was. So is it safe to say that maybe by the end of this sprint, we'll have some plans with options and timings or just looking to level set some expectations? And I know Todd is back from a two-week vacation today, so he'll probably be asking. Well, definitely no more by the end of the sprint. I think we can definitely share what we have. So I'm just going to say, as they continue to unpack and put this together, that knowledge will just get more firm as it continues on. But yeah, by the end of the sprint, I would hope we'd have some kind of idea around that kind of stuff. Are you asking, is there a review, marketing, sprint goal that we're comfortable with?around setting. Yeah. I am asking that. Yeah. Thank you. All right. So here's what I'm going to say. When you take this and you put an estimate out there, so let's say, and this will be my message to Todd and Julie and how I'm going to take it. That estimate we put on sprints, to me it's probably a plus or minus 50 to 75%. So it gives us a swag of time. What's really going to drive a more firm definition is when we start breaking down into issues and stories and start seeing that stuff pointed. Anytime we start to put points on stories, that's what I put as the line in the sand that that's a commitment from engineering as to a size and complexity. And I'll just say complexity. I hate to even say size, so. Yeah, we're not, we're just looking for target timeframes. Knowing that we give you a target for end of March for get back, we're just looking to see, is that target that we put out there, like just insane? And that would never be true. Are we going to be done by Christmas? So we're just looking for targets. We all understand that those targets should grow and change. We also understand even after estimating cars as you're moving through the work, there's likely other things to shake out that would add to those points as well. But we got to start somewhere with a target, right? And then just communicate it over time, how it changes. I'm aligned with you, Jeff. Yeah, yeah, we've talked about it. And I want to narrow it down as much as I can and give as accurate as an answer as I can. And we'll just go based on what we have information-wise. I usually just always say target, target, target, estimate, estimate, estimate. Yeah. Well, too, I mean, putting this report together also gives data to back us up. Yeah, for sure. Agree. Shane, you still don't look happy. I'm good. I guess I'm ready to be done talking and start working. All right, well, get your ass to work. So our spring goal for review marketing, back to that, is... Well, I think we can definitely say like review monitoring is the first piece. So maybe we'll get further than this, but we'll have a rough estimate around the review monitoring piece. And we'll have like the ethics broken down and kind of like finalizing the structure that we think makes the most sense and figure out the sequencing. And I mean, I think we can throw in like the piece about estimating both approaches for either supporting BirdEye in parallel and rolling out piecemeal or not. Maybe we'll go deeper into review marketing. Maybe we'll keep it a little higher level and move into other things. I don't know. Like we can go as deep as we want into that. Sounds good. Shane, do you wanna be in our standups at 8.45 or at least invited so you can pop in from time to time? Yeah, invite me. Okay. It's becoming a party. Okay, well, I would say, yeah, let's get into it. I mean, I think we're getting there. Okay, well, I would say, yeah, let's get into it. Do we wanna just like pop open JIRA and have someone like screen share while we talk through or like look at the diagram and start to figure it out? I got one other quick question, tool-wise. So Henry and I talked a little bit yesterday about Figma Jam or FigJam, or would it be worth a subscription to a handful of you guys to use that? I don't think we need to talk about that now. They're invited to the board of that one, so we can just work out of that for now. Okay, that works. Do you guys need us or do you wanna go off into your world and get started and then reach out to us? I think they're telling us to get the hell off of this and let us go to work. Every minute we sit here and just kinda stare at each other and one last minute we can actually dive into the work. You know what? I was just asking if you needed us if you were diving into the work. I'm guessing no, so that was my night. We probably have lots of questions. We'll have lots of questions. I'm imagining there'll be lots of calling you in randomly. I think we'll just get started and we'll come up with a plan between the three of us and then start, if you're available, bring people in as we have questions. We can probably group our questions and try to hit some sort of critical mass before we pull folks in so that we're not randomly pulling people in for five minutes at a time. 
        """