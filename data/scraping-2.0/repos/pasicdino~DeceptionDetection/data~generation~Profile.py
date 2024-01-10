import os
import random
from openai import OpenAI

# I wrote this script then I realised that you need to pay per token for the api calls, so instead I send this code to
# GPT-4, and ask it to run it and to, instead of making the api calls, ask itself the prompts. Which works.

client = OpenAI(api_key='')

jobs_list = ["Accountant", "Architect", "Artist", "Baker", "Barista", "Bartender", "Carpenter", "Chef", "Chemist",
             "Civil Engineer", "Cleaner", "Dentist", "Designer", "Doctor", "Electrician", "Engineer", "Farmer",
             "Fashion Designer", "Firefighter", "Florist", "Graphic Designer", "Hairdresser", "Journalist", "Lawyer",
             "Librarian", "Mechanic", "Musician", "Nurse", "Optometrist", "Painter", "Pharmacist", "Photographer",
             "Physiotherapist", "Pilot", "Plumber", "Police Officer", "Programmer", "Psychologist", "Real Estate Agent",
             "Receptionist", "Scientist", "Secretary", "Security Guard", "Social Worker", "Teacher", "Translator",
             "Veterinarian", "Waiter/Waitress", "Web Developer", "Writer"]

names_list = ["Aiden Smith", "Isabella Garcia", "Yu Chen", "Olga Ivanova", "Tarun Patel", "Amara Okafor",
              "Juan Martinez", "Emily Johnson", "Noah Wilson", "Sofia Rodriguez", "Liam Brown", "Mia Anderson",
              "Muhammad Khan", "Layla Hassan", "Ethan Davis", "Zoe Jones", "Lucas Baker", "Ava Lopez", "Mason Gonzalez",
              "Lily Young", "Alexander Harris", "Chloe King", "Jackson Lee", "Emma Moore", "Benjamin Clark",
              "Harper Green", "Elijah Lewis", "Mia Murphy", "Daniel Walker", "Amelia Hall", "Gabriel Adams",
              "Nora Thomas", "Logan Nelson", "Isla Wright", "Aarav Singh", "Zoe Hill", "Isaac Scott", "Aaliyah Turner",
              "Levi Campbell", "Grace Carter", "Sebastian Mitchell", "Scarlett Perez", "Caleb Roberts",
              "Victoria Phillips", "Ryan Evans", "Lily Collins", "Wyatt Stewart", "Emily Sanchez", "Oliver Morris",
              "Charlotte Nguyen"]


class Profile:
    def __init__(self):
        self.profile = None
        self.income = None
        self.job = random.choice(jobs_list)
        self.name = random.choice(names_list)
        self.bank_number = random.randint(10000000, 999999999)
        self.last_transaction = '€' + str(random.uniform(0, 500))
        self.create_profile()

    def create_profile(self):
        self.income = self.gpt(
            f'Return a random estimate of the yearly income of a {self.job} in euros. Return only the value.')
        self.profile = self.gpt(
            f'Return an example profile of a {self.job} called {self.name} of max 250 characters. Include basic '
            f'information about their personal life, like their family (including names of children, partner), '
            f'hobbies, and include 2 other examples. Do not mention income. Return only the profile.')

    def gpt(self, prompt):
        response = client.completions.create(
            model="gpt-4",
            prompt=prompt,
            max_tokens=0
        )
        return response.choices[0].text.strip()


profile = Profile()
print(profile.profile)
print(profile.income)
print(profile.job)
print(profile.name)
print(profile.bank_number)
print(profile.last_transaction)
