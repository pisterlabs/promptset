import os
from dotenv import load_dotenv
from openai import OpenAI
import json
import random

# Load environment variables from the .env file
load_dotenv()

# Access the environment variable
api_key = os.getenv("OPENAI_API_KEY")


client = OpenAI(api_key=api_key)

story_ideas = [
    "In a world where emotions are bought and sold, a struggling artist discovers a black market for genuine feelings.",
    "A detective must solve a murder mystery in a small town where everyone has a secret, including the victim.",
    "An astronaut on a solo mission to a distant planet discovers an ancient alien civilization that challenges everything they thought they knew about the universe.",
    "A time traveler accidentally alters the course of history and must navigate a new present while trying to fix the past.",
    "A group of strangers wake up on a deserted island with no memory of how they got there, and they must work together to survive.",
    "In a society where memories can be erased, a woman begins to question the authenticity of her own experiences.",
    "A young wizard discovers that their magical abilities are linked to a powerful artifact, and they must protect it from those who seek to misuse its power.",
    "An AI gains sentience and explores the complexities of human emotions while helping a lonely scientist find love.",
    "A teenager discovers a hidden portal to a parallel world where mythical creatures coexist with humans.",
    "In a post-apocalyptic world, a group of survivors must confront a mysterious force that threatens their last remaining sanctuary.",
    "A person wakes up with the ability to see one day into the future and must use this power to prevent a series of disasters.",
    "A robot develops self-awareness and embarks on a quest to understand its purpose in a world that fears its kind.",
    "In a world where magic is real but illegal, a group of rebels fights against a tyrannical government to restore magic to the world.",
    "A journalist investigates a series of mysterious disappearances in a small town and uncovers a supernatural cult.",
    "A retired superhero is forced to come out of retirement when an old nemesis resurfaces with a new, deadly plan.",
    "An archaeologist discovers an ancient artifact that reveals the true history of humanity, challenging established beliefs.",
    "A group of friends stumbles upon a time-traveling device and decides to revisit key moments in their lives, with unintended consequences.",
    "A high school student discovers they have the power to freeze time but struggles to control this newfound ability.",
    "A cursed painting brings misfortune to those who possess it, and a detective must unravel its dark history to break the curse.",
    "A teenager discovers they have the ability to communicate with animals and must use this power to save endangered species.",
    "In a world where dreams become reality, a person must confront their deepest fears manifested in their dreamscape.",
    "A scientist invents a device that allows people to enter dreams, only to uncover a dark conspiracy lurking in the subconscious.",
    "A musician makes a deal with a supernatural being for unparalleled talent but soon realizes the cost is higher than they imagined.",
    "An ancient prophecy foretells the return of a long-lost civilization, and a chosen hero must prevent a catastrophic war.",
    "A person wakes up with the ability to manipulate time and must choose between saving a loved one or preserving the timeline.",
    "In a future society, a group of rebels fights against a government that controls emotions through a mind-altering drug.",
    "A person discovers a hidden library that contains books about every possible alternate reality, leading to a journey of self-discovery.",
    "A group of survivors in a zombie apocalypse realizes that the zombies are evolving, and they must find a way to coexist or face extinction.",
    "A detective must solve a murder where the victim appears to have been killed by their future self.",
    "An ancient artifact grants a person the ability to manipulate probability, leading to both extraordinary luck and unforeseen consequences.",
    "In a world where people can switch bodies at will, a person wakes up in a new body and must navigate someone else's life.",
    "A cursed video game causes players to experience the game's events in real life, and a group must find a way to break the curse.",
    "A person inherits a haunted mansion and must uncover the dark secrets within to free the trapped spirits.",
    "A person discovers a hidden society living beneath the city streets, with its own rules and conflicts.",
    "A scientist invents a device that allows people to experience each other's dreams, leading to unexpected connections.",
    "In a world where superpowers are common, a person with a seemingly useless ability discovers its true potential.",
    "A person gains the ability to manipulate probability, leading to both extraordinary luck and unforeseen consequences.",
    "A time-traveling detective must solve a crime that spans multiple eras, with each clue leading to a different time period.",
    "In a world where people can manipulate elements, a person with the rare ability to control shadows becomes a target for a dark organization.",
    "A group of friends discovers a mysterious island that appears and disappears at random intervals, leading to a quest for its secrets.",
    "In a society where dreams can be bought and sold, a person becomes a 'dream thief' and must confront the consequences of their actions.",
    "A person discovers a hidden society of magical creatures living alongside humans and must navigate a world of deception and trust.",
    "An inventor creates a device that allows people to relive their happiest memories, but the technology has unintended consequences.",
    "A person discovers a hidden dimension where lost and forgotten things go, leading to a journey to reclaim what has been lost.",
    "A person receives a letter from their future self with instructions on how to prevent a global catastrophe.",
    "A group of friends discovers a mysterious portal in the woods that leads to a realm of magic, but they must navigate its dangers to return home.",
    "In a society where people can visit the afterlife, a person returns from the dead with a warning about an impending disaster.",
    "A geneticist accidentally creates a new species with human-like intelligence, leading to ethical and societal challenges.",
    "In a world where words have magical power, a person discovers a forbidden language that can manipulate reality.",
    "A parallel universe exists where mythical creatures live in harmony, and a portal accidentally opens, causing chaos in both worlds.",
    "A person discovers a mirror that shows glimpses of alternate realities, leading to a quest to find the version of themselves living the best life.",
    "A photographer captures unexplained phenomena in their pictures and becomes entangled in a government conspiracy.",
    "An ancient curse causes a town's residents to age backward, and a group must find a way to break the curse before it's too late.",
    "A person discovers they can communicate with their future self and receives warnings about impending disasters.",
    "In a world where people are assigned soulmates, a person falls in love with someone outside their predetermined match.",
    "An artist creates a masterpiece that comes to life, blurring the line between art and reality.",
    "A person discovers a hidden book that can transport readers into the stories, but with unforeseen consequences.",
    "A person finds a mysterious map that leads to a hidden realm filled with mythical creatures and untold treasures.",
    "In a world where dreams determine the future, a person with the ability to control their dreams becomes a target for those seeking power.",
    "A person discovers a hidden society living beneath the ocean and must uncover its secrets before it falls into the wrong hands.",
    "An ancient artifact grants a person the ability to absorb and use the skills of others, but at the cost of losing their own identity.",
    "A group of friends discovers a mysterious board game that transports them to a fantasy realm with real-life consequences.",
    "A person discovers a hidden society of shape-shifters living among humans and must navigate a world of deception and trust.",
    "A scientist creates a device that allows people to travel to parallel universes, leading to encounters with alternate versions of themselves.",
    "In a future where technology controls every aspect of life, a hacker group fights for freedom in the virtual and real worlds.",
    "A person discovers a hidden city beneath the ocean and must uncover its secrets before it falls into the wrong hands.",
    "In a world where people can manipulate elements, a person with the rare ability to control shadows becomes a target for a dark organization.",
    "A person discovers a hidden society living beneath the city streets, with its own rules and conflicts.",
    "An ancient artifact grants a person the ability to speak and understand all languages, uncovering a global conspiracy.",
    "A scientist invents a device that allows people to relive their happiest memories, but the technology has unintended consequences.",
    "In a future where emotions are controlled by a government-issued drug, a person goes off the medication and discovers the true cost of free will.",
    "A person discovers they can communicate with plants and must use this ability to save the environment from a looming disaster.",
    "A group of friends discovers a mysterious portal in the woods that leads to a realm of magic, but they must navigate its dangers to return home.",
    "In a world where people can switch bodies at will, a person wakes up in a new body and must navigate someone else's life.",
    "An inventor creates a device that allows people to experience the sensations of others, leading to ethical questions and personal growth.",
    "A person inherits a haunted mansion with rooms that change based on the occupants' fears, forcing them to confront their own demons.",
    "A group of explorers discovers a gateway to a dimension where time flows backward, leading to encounters with past versions of themselves.",
    "A person discovers a book that allows them to rewrite their own life story, with unforeseen consequences.",
    "A person discovers a hidden society of magical creatures living alongside humans and must navigate a world of deception and trust.",
    "An ancient artifact grants a person the ability to speak and understand all languages, uncovering a global conspiracy.",
    "In a future where robots have gained sentience, a person forms an unlikely alliance with a robot to uncover a corporate conspiracy.",
    "A person discovers a hidden dimension where lost and forgotten things go, leading to a journey to reclaim what has been lost."
]

def get_random_idea():
    if story_ideas:
        selected_idea = random.choice(story_ideas)
        story_ideas.remove(selected_idea)
        return selected_idea
    else:
        return "No more ideas left."

# Get and print one random idea
random_idea = get_random_idea()

#  Function to generate a short story and title
def generate_story_with_title():
    completion = client.chat.completions.create(
        # model="gpt-3.5-turbo",
        model="gpt-3.5-turbo-1106",
        response_format={ "type": "json_object" },
        messages=[
            {"role": "system", "content": "You are renowned children story writer. You write children's stories. The story should be max 500 words in JSON format and must contains only title and story. No special charachters. The Deutsch langugae level should be between A1 to B1 only."},
            {"role": "user", "content": f"Generate me a story on {random_idea} and title. The story should be in Deutsch."},
        ]
    )

    # Extracting story and title from the generated content
    response_content = completion.choices[0].message.content

    # Parse the JSON response
    response_json = json.loads(response_content)

    # # Extract the title
    title = response_json["title"]
    story = response_json["story"]

    return title, story