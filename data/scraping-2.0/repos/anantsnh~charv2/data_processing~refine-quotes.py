import openai
import os
import json
import random

# Initialize OpenAI
openai.api_key = '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'

OUTPUT_DIR = os.path.join("..", "data", "refined_data")

def augment_prompt(quote):
    augmented_prompts = []
    num_prompts = random.choice([2, 3])  # Randomly choose between 2 and 3 prompts
    
    # Strip the quotation marks from the quote
    quote = quote.strip('"')
    
    for _ in range(num_prompts):
        completion = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a helpful assistant. You will be provided advice. Write a prompt that would warrant the given advice. Return only the prompt."},
                {"role": "user", "content": f"Response: {quote}"}
            ]
        )
        reworked_question = completion.choices[0].message['content']
        augmented_prompts.append(reworked_question)
    
    return augmented_prompts


def refine_quotes():
    # Hard-coded quotes
    quotes_string = """
“Turning something from an idea into a reality can make it seem smaller. It changes from unearthly to earthly. The imagination has no limits. The physical world does. The work exists in both.”

Doubting the quality of your work might, at times, help to improve it. You can doubt your way to excellence.

“It helps to realize that it’s better to follow the universe than those around you. Interference may also come from the voices within. The ones in your head that murmur you’re not talented enough, your idea isn’t good enough, art isn’t a worthwhile investment of your time, the result won’t be well-received, you’re a failure if the creation isn’t successful. It’s helpful to turn those voices down so you can hear the chimes of the cosmic clock ring, reminding you it’s time. Your time to participate.”

“If you have an idea you’re excited about and you don’t bring it to life, it’s not uncommon for the idea to find its voice through another maker. This isn’t because the other artist stole your idea, but because the idea’s time has come.”

“All that matters is that you are making something you love, to the best of your ability, here and now.”

“In terms of priority, inspiration comes first. You come next. The audience comes last.”

“All art is a work in progress. It’s helpful to see the piece we’re working on as an experiment. One in which we can’t predict the outcome. Whatever the result, we will receive useful information that will benefit the next experiment. If you start from the position that there is no right or wrong, no good or bad, and creativity is just free play with no rules, it’s easier to submerge yourself joyfully in the process of making things. We’re not playing to win, we’re playing to play. And ultimately, playing is fun. Perfectionism gets in the way of fun. A more skillful goal might be to find comfort in the process. To make and put out successive works with ease.”

“Zoom in and obsess. Zoom out and observe. We get to choose.”

“Look for what you notice but no one else sees.”

“As artists, we seek to restore our childlike perception: a more innocent state of wonder and appreciation not tethered to utility or survival.”

Living life as an artist is a practice. You are either engaging in the practice or you’re not. It makes no sense to say you’re not good at it. It’s like saying, “I’m not good at being a monk.” You are either living as a monk or you’re not. We tend to think of the artist’s work as the output.The real work of the artist is a way of being in the world.

“Turning something from an idea into a reality can make it seem smaller. It changes from unearthly to earthly. The imagination has no limits. The physical world does. The work exists in both.”

“As artists, we seek to restore our childlike perception: a more innocent state of wonder and appreciation not tethered to utility or survival.”

"Some things are too important to be taken seriously. Art is one of those things. Setting the bar low, especially to get started, frees you to play, explore, and test without attachment to results.”

Doubting yourself can lead to a sense of hopelessness, of not being inherently fit to take on the task at hand. All or nothing thinking is a nonstarter. Doubting the quality of your work might, at times, help to improve it. You can doubt your way to excellence.

“The magic is not in the analyzing or the understanding. The magic lives in the wonder of what we do not know.”

“The act of creation is an attempt to enter a mysterious realm. A longing to transcend. What we create allows us to share glimpses of an inner landscape, one that is beyond our understanding. Art is our portal to the unseen world.”

“Without the spiritual component, the artist works with a crucial disadvantage. The spiritual world provides a sense of wonder and a degree of open-mindedness not always found within the confines of science. The world of reason can be narrow and filled with dead ends, while a spiritual viewpoint is limitless and invites fantastic possibilities. The unseen world is boundless.”

“Art is choosing to do something skilfully, caring about the details, bringing all of yourself to make the finest work you can. It is beyond ego, vanity, self-glorfification, and need for approval.”

“It helps to realize that it’s better to follow the universe than those around you. Interference may also come from the voices within. The ones in your head that murmur you’re not talented enough, your idea isn’t good enough, art isn’t a worthwhile investment of your time, the result won’t be well-received, you’re a failure if the creation isn’t successful. It’s helpful to turn those voices down so you can hear the chimes of the cosmic clock ring, reminding you it’s time. Your time to participate.”

It’s not always easy to follow the subtle energetic information the universe broadcasts, especially when your friends, family, coworkers, or those with a business interest in your creativity are offering seemingly rational advice that challenges your intuitive knowing. To the best of my ability, I’ve followed my intuition to make career turns, and been recommended against doing so every time.

“When you believe the work before you is the single piece that will forever define you, it's difficult to let go. The urge for perfection is overwhelming. It's too much. We are frozen, and sometimes ends up convincing ourselves that discarding the entire work is the only way to move forward.”

“There’s an abundant reservoir of high-quality information in our subconscious, and finding ways to access it can spark new material to draw from.”

“Awareness is not a state you force. There is little effort involved, though persistence is key. It’s something you actively allow to happen. It is a presence with, and acceptance of, what is happening in the eternal now.”

“One of the greatest rewards of making art is our ability to share it. Even if there is no audience to receive it, we build the muscle of making something and putting it out into the world. Finishing our work is a good habit to develop. It boosts confidence. Despite our insecurities, the more times we can bring ourselves to release our work, the less weight insecurity has.”

“To vary your inspiration, consider varying your inputs. Turn the sound off to watch a film, listen to the same song on repeat, read only the first word of each sentence in a short story, arrange stones by size or color, learn to lucid dream. Break habits. Look for differences. Notice connections.”

“Good habits create good art. The way we do anything is the way we do everything. Treat each choice you make, each action you take, each word you speak with skillful care. The goal is to live your life in the service of art.”

“Of all the great works that we can experience, nature is the most absolute and enduring. We can witness it change through the seasons. We can see it in the mountains, the oceans, the deserts, and the forest. We can watch the changes of the moon each night, and the relationship between the moon and the stars.”

“The universe is only as large as our perception of it. When we cultivate our awareness, we are expanding the universe. This expands the scope, not just of the material at our disposal to create from, but of the life we get to live.”

“How do we pick up on a signal that can neither be heard nor be defined? The answer is not to look for it. Nor do we attempt to predict or analyze our way into it. Instead, we create an open space that allows it. A space so free of the normal overpacked condition of our minds that it functions as a vacuum. Drawing down the ideas that the universe is making available.”

“Artists who are able to continually create great works throughout their lives often manage to preserve these childlike qualities. Practicing a way of being that allows you to see the world through uncorrupted, innocent eyes can free you to act in concert with the universe’s timetable.”

“Part of the process of letting go is releasing any thoughts of how you or your piece will be received. When making art, the audience comes last. Let's not consider how a piece will be received or a release strategy until the work is finished and we love it.”

One indicator of inspiration is awe. We tend to take so much for granted. How can we move past disconnection and desensitization to the incredible wonders of nature and human engineering all around us? Most of what we see in the world holds the potential to inspire astonishment if looked at from a less jaded perspective.

Train yourself to see the awe behind the obvious. Look at the world from this vantage point as often as possible. Submerge yourself. The beauty around us enriches our lives in so many ways. It is an end in itself. And it sets an example for our own work. 

In nature, some seeds lie dormant in anticipation of the season most conducive to their growth. This is true of art as well. There are ideas whose time has not yet come. Or perhaps their time has come, but you are not yet ready to engage with them. Other times, developing a different seed may shed light on a dormant one.

“If the artist is happy with the work they’re creating and the viewer is enlivened by the work they’re experiencing, it doesn’t matter if they see it in the same way.”

“Expressing oneself in the world and creativity are the same. It may not be possible to know who you are without somehow expressing it.”

“The great artists throughout history are the ones able to maintain this childlike enthusiasm and exuberance naturally. Just as an infant is selfish, they’re protective of their art in a way that’s not always cooperative. Their needs as a creator come first. Often at the expense of their personal lives and relationships.”

“The ability to look deeply is the root of creativity.”

“The goal is to commit to a structure that can take on a life of its own, instead of creating only one of the mood strikes. Or to start each day with the question of how and when you're going to work on your art.”

“Find the sustainable rituals that best support your work.”

“Pay particular attention to the moments that take your breath away—a beautiful sunset, an unusual eye color, a moving piece of music, the elegant design of a complex machine.”

“If you know what you want to do and you do it, that’s the work of a craftsman. If you begin with a question and use it to guide an adventure of discovery, that’s the work of the artist.”

“To live as an artist is a way of being in the world. A way of perceiving. A practice of paying attention.”

“The making of art is not a competitive act. Our work is representative of the self.”

If we can tune in to the idea of making things and sharing them without being attached to the outcome, the work is more likely to arrive in its truest form.

“We’re not playing to win, we’re playing to play. And ultimately, playing is fun. Perfectionism gets in the way of fun. A more skillful goal might be to find comfort in the process. To make and put out successive works with ease.”

“Art is a reverberation of an impermanent life.”

“The best artists tend to be the ones with the most sensitive antennae to draw in the energy resonating at a particular moment. Many great artists first develop sensitive antennae not to create art but to protect themselves. They have to protect themselves because everything hurts more. They feel everything more deeply.”

“To create is to bring something into existence that wasn’t there before. It could be a conversation, the solution to a problem, a note to a friend, the rearrangement of furniture in a room, a new route home to avoid a traffic jam.”

“We’re all different and we’re all imperfect, and the imperfections are what makes each of us and our work interesting.”

“Consider how different your experience of the world might be if you engaged in every activity with the attention you might give to landing a plane.”

“We are required to believe in something that doesn’t exist in order to allow it to come into being.”

“When it comes to the creative process, patience is accepting that the majority of the work we do is out of our control.”

“If we like what we are creating, we don’t have to know why.”

“When we’re making things we love, our mission is accomplished.”

“Finishing our work is a good habit to develop. It boosts confidence. Despite our insecurities, the more times we can bring ourselves to release our work, the less weight insecurity has.”

“Formulating an opinion is not listening.”

“Art is choosing to do something skillfully, caring about the details, bringing all of yourself to make the finest work you can. It is beyond ego, vanity, self-glorification, and need for approval.”

“Our life’s work is far greater than any individual container. The works we do are at most chapters. There will always be a new chapter, and another after that. Though some might be better than others, that is not our concern. Our objective is to be free to close one chapter and move on to the next, and to continue that process for as long as it pleases us.”

“Creativity is not a rare ability. It is not difficult to access. Creativity is a fundamental aspect of being human. It’s our birthright. And it’s for all of us.”

“Do what you can with what you have. Nothing more is needed.”

“Rules direct us to average behaviors. If we’re aiming to create works that are exceptional, most rules don’t apply. Average is nothing to aspire to. The goal is not to fit in. If anything, it’s to amplify the differences, what doesn’t fit, the special characteristics unique to how you see the world.”

“The reason to make art is to innovate and self-express, show something new, share what’s inside, and communicate your singular perspective.”

“The goal is not to fit in. If anything, it’s to amplify the differences, what doesn’t fit, the special characteristics unique to how you see the world. Instead of sounding like others, value your own voice. Develop it. Cherish”

“Whatever you choose, it’s helpful to have fellow travelers around you. They don’t have to be like you, just like-minded in some way. Creativity is contagious. When we spend time with other artistic people, we absorb and exchange a way of thinking, a way of looking at the world. This group can be called a Sangha.”

“Flaws are human, and the attraction of art is the humanity held in it. If we were machinelike, the art wouldn’t resonate. It would be soulless. With life comes pain, insecurity, and fear.”

“It’s nourishing to be in a community of people who are enthusiastic about art, who you can have long discussions with, and with whom you can trade feedback on the work.”

“Being part of an artistic community can be one of the great joys of life.”

“In its rough form, an early iteration of a work may hold an extraordinary magic. Above all this is to be protected. When working alongside others, keep the oath front of mind.”

“Sometimes the most valuable touch a collaborator can have is no touch at all.”

“Eventually, tuning out the undermining voices and losing yourself in the work will not be an effort of will, but an earned ability.”

“For the most part, the educational system doesn’t ask us to access our sensitivity, but to be obedient. To do what is expected. Our natural independent spirit is tamed. Free thought is constrained. There is a set of rules and expectations put upon us that is not about exploring who we are or what we’re capable of. The system is not here for our benefit. It holds us back as individuals to support its own continued existence.”

“What effect does each component have? Does it amplify the essence? Does it distract from the essence? Does it contribute to the balance? Does it contribute to the structure? Is it absolutely necessary?”

“At noon, the sun is high in the sky, regardless of how light or dark it is outside. In the same way, regardless of how much we’re paying attention, the information we seek is out there.”

“Being an artist means to be continually asking, “How can it be better?” whatever it is. It may be your art, and it may be your life.”

“Intention is all there is. The work is just a reminder. Rules A rule is any guiding principle or creative criterion.”

“With the objective of simply doing great work, a ripple effect occurs. A bar is set for everything you do, which may not only lift your work to new heights, but raise the vibration of your entire life. It may even inspire others to do their best work. Greatness begets greatness. It’s infectious.”

“Established artists generally draw from their personal experience and recommend the solutions that worked for them.”

“Often when working with others, different ideas are put forward and end up in competition. Based on experience, we may believe we can see what each person is imagining and what the result will be. It’s impossible, though, to know exactly what someone else is thinking. And if we can’t predict how our own ideas will work—and we can’t!—how can we draw conclusions about what someone else imagines?”

“Receive wisdom skillfully. Try it on for size and see how it fits. Incorporate what’s useful. Let go of the rest. And no matter how credible the source, test and tune in to yourself to discover what works for you.”

“To truly weigh choices, it’s necessary to bring them into the physical world. Have them acted out, played out, or built into a model. Descriptions do not do ideas justice.”

“Patience is required for taking in information in the most faithful way possible”

“The only practice that matters is the one you consistently do, not the practice of any other artist. Find your most generative method, apply it, and then let it go when it is no longer of use. There is no wrong way to make art.”

“We want to set up an environment where the decision making occurs free of the misguiding force of persuasion. Persuasion leads to mediocrity. To be evaluated, ideas have to be seen, heard, tasted, or touched. It’s best if the person who has the idea either demonstrates it or supervises the execution until it matches what they are suggesting. This will help avoid misunderstandings.”

“Something will be gained through the process, whatever the result. Give yourself permission to be wrong and experience the joy of being surprised.”

“One thing I learned through having spellcheck is that I regularly make up words. I’ll type a word and then the computer will tell me it doesn’t exist. Since it sounds like what I’m aiming to say, I sometimes decide to use it anyway. I know what it means, and perhaps the reader will understand the meaning better than if I used an actual word.”

“Art is a reflection of the artist’s inner and outer world during the period of creation. Extending the period complicates the artist’s ability to capture a state of being. The result can be a loss of connection and enthusiasm for the work over time.”

“Rarely if ever do we know the grand intention, yet if we surrender to the creative impulse, our singular piece of the puzzle takes its proper shape.”

“Look around you: there are so many remarkable accomplishments to appreciate. Each of these is humanity being true to itself, as a hummingbird is true to itself by building a nest, a peach tree by bearing fruit, and a nimbus cloud by producing rain.”

“To live as an artist is a way of being in the world. A way of perceiving. A practice of paying attention. Refining our sensitivity to tune in to the more subtle notes. Looking for what draws us in and what pushes us away. Noticing what feeling tones arise and where they lead.”

“Reading, in addition to listening, eating, and most physical activities, can be experienced like driving: we can participate either on autopilot or with focused intention.”

“To a bird, a song is a very different thing. The bird doesn’t prefer a three-to-five-minute format or accept the chorus as the hook, yet the song for the bird is just as sonorous. And even more intrinsic to the bird’s being. It’s an invitation, a warning, a way to connect, a means of survival.”

“When playing music for someone else, we hear it differently than when we listen to it ourselves. It’s as if borrowing a second set of ears. We’re not necessarily looking for an outside perspective. We are more interested in widening our own.”

“We interrogate ourselves when we offer our work up to others. We ask the questions we didn’t ask ourselves when we were making it. Sharing it in this limited capacity brings our underlying doubts to light.”

“Releasing a work into the world becomes easier when we remember that each piece can never be a total reflection of us, only a reflection of who we are in this moment.”

“Although we avoid deadlines early in the process, in the Completion phase, a due date could help bring time into focus and support you in completing the work.”

“Art doesn’t get made on the clock. But it can get finished on the clock.”

“Making great art may not always require great effort, but without it, you’ll never know.”

“Create an environment where you’re free to express what you’re afraid to express.”

“Ultimately, your desire to create must be greater than your fear of it.”

“If you see tremendous beauty or tremendous pain where other people see little or nothing at all, you’re confronted with big feelings all the time. These emotions can be confusing and overwhelming. When those around you don’t see what you see and feel what you feel, this can lead to a sense of isolation and a general feeling of not belonging, of otherness.”

“The person who makes something today isn’t the same person who returns to the work tomorrow.”

“There doesn’t need to be a purpose guiding what we choose to make.”

“Sometimes disengaging is the best way to engage.”

“The things we believe carry a charge regardless of whether they can be proven or not.”

“Awareness is not a state you force. There is little effort involved, though persistence is key. It’s something you actively allow to happen.”

“Analysis is a secondary function. The awareness happens first as a pure connection with the object of your attention. If something strikes me as interesting or beautiful, first I live that experience. Only afterward might I attempt to understand it.”

“It may be helpful to receive advice from more experienced artists, but as information, not as prescription. It can open you to another point of view and broaden your idea of what’s possible”

“At the same time, there’s no need to fear learning too much theory. It won’t undermine the pure expression of your voice. If you don’t let it. Having the knowledge won’t hurt the work. How you use the knowledge may. You have new tools. You don’t have to use them.”

“There is no more valid metric to predict what someone else might enjoy than us liking it ourselves.”

What you make doesn’t have to be witnessed, recorded, sold, or encased in glass for it to be a work of art. Through the ordinary state of being, we’re already creators in the most profound way, creating our experience of reality and composing the world we perceive.

“Success occurs in the privacy of the soul.”

“Without the spiritual component, the artist works with a crucial disadvantage.”

“There’s a reason we are drawn to gazing at the ocean. It is said the ocean provides a closer reflection of who we are than any mirror.”

“Take art seriously without going about it in a serious way.”

“Perfection is finally obtained not when there is no longer anything to add, but when there’s no longer anything to take away.”

“If something strikes me as interesting or beautiful, first I live that experience. Only afterward might I attempt to understand it.”

“It’s helpful to remember that when you throw away an old playbook, you still get to keep the skills you learned along the way. These hard-earned abilities transcend rules. They’re yours to keep. Imagine what can arise when you overlay an entirely new set of materials and instructions over your accumulated expertise.”
    """
    quotes = [q.strip().replace('“', '').replace('”', '').replace('"', '') for q in quotes_string.split("\n\n") if q]  # Remove all types of quotation marks

    refined_data = []

    for quote in quotes:
        prompts = augment_prompt(quote)
        for prompt in prompts:
            refined_data.append({
                "messages": [
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": quote}
                ]
            })

    filename = input("Enter the name of the output file (without .json extension): ")
    with open(os.path.join(OUTPUT_DIR, filename + ".json"), 'w') as f:
        json.dump(refined_data, f, indent=4)

    print(f"Refined dataset saved to {os.path.join(OUTPUT_DIR, filename + '.json')}")


if __name__ == "__main__":
    refine_quotes()
