import openai

from src import Constants

if __name__ == '__main__':
    openai.api_key = Constants.API_KEYS_GPT3[0]
    examples = []

    examples.append("""
Table: Persons
-----------
Name | Year
-----------
Enzo | 35
null | null
Paolo | 45
John | null
----------

Task: read(name,year)[*]
Example: Enzo is 35 and Paolo is 45 years old.
""")

    examples.append("""
Table: Persons
-----------
Name | Year | Country | Salary
-----------
Enzo | 35 | ITA | 10000
Paolo | 45 | SPA | 2000
----------

Task: read(name,year)[*]
Example: Enzo is 35 and Paolo is 45 years old.
    """)

    examples.append("""
Table: Persons
-----------
Name | Year | Income
-----------
Paolo | 45 | 1500
Enzo | 35 | 1000
----------

Task: read(name,year,income)[*], compare(<,year), compare(>,income)
Example: Enzo is 35 years old and is younger than Paolo. But Paolo has an income of 1500 that is higher than the Enzo's income that is 1000.
""")

    examples.append("""
Table: Persons
-----------
Name | Surname | Year
-----------
Enzo | Rossi | 35
Paolo | Verdi | 45
----------

Task: read(name, surname)[*], compare(>,year)
Example: Paolo Verdi is older than Enzo Rossi.
    """)

    examples.append("""
Table: Persons
-----------
Name | Surname | Year | City
-----------
Enzo | Rossi | 50 | PZ
Paolo | Verdi | 30 | Rome
----------

Task: read(surname,name)[*], compare(<,year)
Example: Verdi Paolo is younger than Rossi Enzo.
    """)

    examples.append("""
Table: Universities
-----------
School | Faculty | Students
-----------
Roma 3 | Mathematics | 4500
Unibas | Computer Science | 3500
----------

Task: read(faculty,school,students)[*], compare(<,students)
Example: The faculty of Computer Science of Unibas has 3500 students which is less than the Faculty of Mathematics of Roma 3 that has 4500 students.
    """)

    examples.append("""
Table: Persons
-----------
Name | Year
-----------
Enzo | 35
Paolo | 45
John | 35
-----------

Task: read(name,year)[year=35], compare(<,year)
Example: Enzo and John are 35 and both are younger than Paolo.
 """)

#     examples.append("""
# Table: Devices
# -----------
# Model | CPU
# -----------
# Samzumg X1000 | 2.3
# iPhongy | 3.7
# Samzumg X3000 | 2.3
# Samzumg X2000 | 2.3
# ----------
#
# Task: read(model,cpu)[*], compare(=,cpu), read(model,cpu)[*], compare(<,cpu),
# Example: Samzumg X1000, Samzumg X3000 and Samzumg X2000 have the same CPU (2.3). Their CPUs are slower than iPhongy CPU that is 3.7.
#  """)

    examples.append("""
Table: Persons
-----------
Name | Year
-----------
Enzo | 25
Paolo | 25
----------

Task: read(name)[*], compare(=,year)
Example: Enzo and Paolo have the same age.
""")

    examples.append("""
Table: Persons
-----------
Name | Year
-----------
Enzo | 23
Paolo | 22
John | 35
----------

Task: compute(max,year)=35, read(name)[max]
Example: John is the oldest
""")

    examples.append("""
Table: Persons
-----------
Name | Year
-----------
Enzo | 23
Paolo | 22
John | 35
----------

Task: compute(max,year)=35, read(name,year)[max]
Example: John is the oldest, and he's 35 years old
""")

#provare read(name,year=35)
    examples.append("""
Table: Persons
-----------
Name | Year
-----------
Enzo | 35
Paolo | 22
John | 35
----------

Task: compute(max,year)=35, read(name,year)[max]
Example: John and Enzo are the oldest with 35 years old.
""")

    examples.append("""
Table: Persons
-----------
Name | Year
-----------
John | 22
Enzo | 35
Paolo | 22
----------

Task: compute(min,year)=22, read(name,year)[min]
Example: Paolo and John are the yougest with 22 years old.
""")

    examples.append("""
Table: Persons
-----------
Name | Year
-----------
Enzo | 36
Paolo | 46
Mike | 18
----------

Task: compute(count,*)=3, read(count,name)
Example: There are three persons. Namely Enzo, Paolo and Mike.
""")

    examples.append("""
Table: Persons
-----------
Name | Year
-----------
Enzo | 36
Paolo | 46
Mike | 18
----------

Task: compute(avg,year)=33.33, read(avg)
Example: The average age is 33.33.
""")

    examples.append("""
Table: Persons
-----------
Name | Year
-----------
Enzo | 36
Paolo | 46
Mike | 18
----------

Task: filter(> 12,age), compute(count,*)=3, read(name)
Example: There are three persons with age greater than 12. Namely Enzo, Paolo and Mike.
""")

    examples.append("""
Table: Persons
-----------
Name | Country
-----------
Enzo | ITA
Paolo | ITA
Mike | ITA
----------

Task: filter(= ITA, country), compute(count,*)=3, read(Name,Country)
Example: There are three persons from Italy. Namely Enzo, Paolo and Mike.
""")

    examples.append("""
Table: Persons
-----------
Name | Income
-----------
Paolo | 36000
Enzo | 24000
----------

Task: percentage(income, <)=-50.0%
Example: Enzo has 50% of the income lower than Paolo
""")

    examples.append("""
Table: Persons
-----------
Name | Income
-----------
Enzo | 24000
Paolo | 36000
----------

Task: read(name,income)[*] percentage(income, >)=33.33%
Example: Paolo has 33.33% of the income higher than Enzo
""")

#     examples.append("""
# Table: Persons
# -----------
# Name | Country | Income
# -----------
# Enzo | ITA | 10
# Paolo | ITA | 20
# Mike | USA | 30
# Albert | USA | 40
# ----------
#
# Task: compute(sum,Income)[ITA] = 30, compute(sum,Income)[USA] = 70, compare(<,sum)
# Example: The total income from Italian is 30 and is lower than the total income from USA which is 70.
# """)

# ==================== TASKS ====================

# OK
# Mandy has driven 10000 kilometers and Tronky has driven 5000 kilometers.
    taskRead = """
Table: Drivers
-----------
Team | Name | Kilometers
-----------
Ferx | Mandy | 10000
Rammers | Tronky | 5000
----------

Task: read(Name,Kilometers)[*]
Example: 
"""

# OK
# Mandy from Ferx has driven 10000 kilometers and Tronky from Rammers has driven 5000 kilometers.
    taskRead2 = """
Table: Drivers
-----------
Team | Name | Kilometers
-----------
Ferx | Mandy | 10000
Rammers | Tronky | 5000
----------

Task: read(Name,Team,Kilometers)[*]
Example: 
"""

# OK (but we would like to omit kilometers values from the sentence
# Tronky from Rammers team has driven less kilometers (5000) than Mandy from Ferx team (10000).
    taskCompare = """
Table: Drivers
-----------
Team | Name | Kilometers
-----------
Ferx | Mandy | 10000
Rammers | Tronky | 5000
----------

Task: read(team,name)[*], compare(<,kilometers)
Example: 
"""

# OK
# Tronky from Rammers has driven 5000 kilometers, which is less than Mandy from Ferx who has driven 10000 kilometers.
    taskCompare2 = """
Table: Drivers
-----------
Team | Name | Kilometers
-----------
Ferx | Mandy | 10000
Rammers | Tronky | 5000
----------

Task: read(name,team,kilometers)[*], compare(<,kilometers)
Example: 
"""

#ok - note that between group and read there is no coma
#Tronky and Amber have driven 5000 kilometers, which is less than the other drivers in their respective teams.

    taskCompare3 = """
Table: Drivers
-----------
Team | Name | Kilometers 
-----------
Ferx | Mandy | 10000
Ferx | Carleston | 20000
Rammers | Tronky | 5000
Rammers | RetroBar | 62000
Ferx | KiloKalo | 12000
Ferx | Amber | 5000
----------

Task: group(Kilometers) read(name,Kilometers)[Kilometers=5000], compare(<,Kilometers)
Example: 
"""

    taskCompare4 = """
Table: Persons
-----------
Name | Year
-----------
Jack | 45
Enzo | 35
Paolo | 45
----------
Task: read(name,year)[*], compare(>,year)
Example:
"""

## NOTE: in this case the order matters to read the CPU higher than. (But only for CPUs)
    taskCompare5 = """
Table: Devices
-----------
Model | CPU | RAM
-----------
iPhongy | 3.7 | 4
Samzumg X1000 | 2.3 | 6
----------
Task: read(model,cpu)[*], compare(>,cpu)
Example:
"""

# ERROR: The iPhongy has a CPU of 3.7 which is less than the Samsung X1000 that has a CPU of 2.3. The Samsung X1000 has 6GB of RAM while the iPhongy has 4GB of RAM.
    taskCompare6 = """
Table: Devices
-----------
Model | CPU | RAM
-----------
Samzumg X1000 | 2.3 | 6
iPhongy | 3.7 | 4
----------
Task: read(model,cpu,ram)[*], compare(>,cpu)
Example:
"""

#ok
#The driver with the highest number of kilometers is RetroBar.
    taskMax = """
Table: Drivers
-----------
Team | Name | Kilometers
-----------
Ferx | Mandy | 10000
Ferx | Carleston | 20000
Rammers | Tronky | 5000
Rammers | RetroBar | 60000
Ferx | KiloKalo | 15000
Ferx | Amber | 5000
----------

Task: read(name)[max]
Example: 
"""

#ok
#RetroBar has driven the most kilometers with 60000.
    taskMax2 = """
Table: Drivers
-----------
Team | Name | Kilometers
-----------
Ferx | Mandy | 10000
Ferx | Carleston | 20000
Rammers | Tronky | 5000
Rammers | RetroBar | 60000
Ferx | KiloKalo | 15000
Ferx | Amber | 5000
----------

Task: compute(max,Kilometers)=60000, read(name,Kilometers)[max]
Example: 
"""

#ok
#Tronky and Amber have driven the least amount of kilometers, with 5000 each.
    taskMax3 = """
Table: Drivers
-----------
Team | Name | Kilometers
-----------
Ferx | Mandy | 10000
Ferx | Carleston | 20000
Rammers | Tronky | 5000
Rammers | RetroBar | 60000
Ferx | KiloKalo | 15000
Ferx | Amber | 5000
----------

Task: compute(min,Kilometers)=5000, read(name)[min]
Example: 
"""

#ok - read is not needed (will read km of all drivers)
#The average kilometers driven by the drivers is 19000.
    taskAvg = """
Table: Drivers
-----------
Team | Name | Kilometers
-----------
Ferx | Mandy | 10000
Ferx | Carleston | 20000
Rammers | Tronky | 5000
Rammers | RetroBar | 62000
Ferx | KiloKalo | 12000
Ferx | Amber | 5000
----------

Task: compute(avg,Kilometers)=19000
Example: 
"""

#ok
#There are 6 drivers in total.
    taskCount = """
Table: Drivers
-----------
Team | Name | Kilometers
-----------
Ferx | Mandy | 10000
Ferx | Carleston | 20000
Rammers | Tronky | 5000
Rammers | RetroBar | 62000
Ferx | KiloKalo | 12000
Ferx | Amber | 5000
----------

Task: compute(count,*)=6
Example: 
"""

    taskGroupCount = """
Table: Drivers
-----------
Team | Name | Kilometers
-----------
Ferx | Mandy | 10000
Ferx | Carleston | 20000
Rammers | Tronky | 5000
Rammers | RetroBar | 60000
Ferx | KiloKalo | 15000
Ferx | Amber | 5000
----------

Task: group(team), compute(max,*)
Example: 
"""

#ok
#There are six drivers in total. Their names are Mandy, Carleston, Tronky, RetroBar, KiloKalo, and Amber.
    taskCount2 = """
Table: Drivers
-----------
Team | Name | Kilometers
-----------
Ferx | Mandy | 10000
Ferx | Carleston | 20000
Rammers | Tronky | 5000
Rammers | RetroBar | 62000
Ferx | KiloKalo | 12000
Ferx | Amber | 5000
----------

Task: compute(count,*)=6, read(name)
Example: 
"""

#ok
#There are six drivers who have driven less than 13000 kilometers. Namely Mandy, Tronky, KiloKalo and Amber from Ferx team, and Tronky and RetroBar from Rammers team.
    taskCount3 = """
Table: Drivers
-----------
Team | Name | Kilometers
-----------
Ferx | Mandy | 10000
Ferx | Carleston | 20000
Rammers | Tronky | 5000
Rammers | RetroBar | 62000
Ferx | KiloKalo | 12000
Ferx | Amber | 5000
----------

Task: filter(< 13000,kilometers), compute(count,*)=6, read(name)
Example: 
"""

#ok
#There are four drivers in the Ferx team. Namely Mandy, Carleston, KiloKalo and Amber.

    taskCount4 = """
Table: Drivers
-----------
Team | Name | Kilometers 
-----------
Ferx | Mandy | 10000
Ferx | Carleston | 20000
Rammers | Tronky | 5000
Rammers | RetroBar | 62000
Ferx | KiloKalo | 12000
Ferx | Amber | 5000
----------

Task: filter(= Ferx,Team), compute(count,*)=4, read(name)
Example: 
"""

#ok
#The average kilometers driven by Ferx team is 12500, while the average kilometers driven by Rammers team is 32500, which is higher.

    taskGroupAvg = """
Table: Drivers
-----------
Team | Name | Kilometers
-----------
Ferx | Mandy | 10000
Ferx | Carleston | 20000
Rammers | Tronky | 5000
Rammers | RetroBar | 60000
Ferx | KiloKalo | 15000
Ferx | Amber | 5000
----------

Task: compute(avg,Kilometers)[team=Ferx]=12500, compute(avg,Kilometers)[team=Rammers]=32500, compare(>,avg)
Example: 
"""

#ok
#For team Ferx, the average kilometers driven is 12500. For team Rammers, the average kilometers driven is 32500.
    taskGroupAvg2 = """
Table: Drivers
-----------
Team | Name | Kilometers
-----------
Ferx | Mandy | 10000
Ferx | Carleston | 20000
Rammers | Tronky | 5000
Rammers | RetroBar | 60000
Ferx | KiloKalo | 15000
Ferx | Amber | 5000
----------

Task: group(team), compute(avg,*)
Example: 
"""

    taskTestRank = """
Table: Persons
-----------
Name | Year
-----------
Enzo | 36
null | 46
null | 18
----------

Task: ranked(2,asc,Year)=36
Example: Enzo is the 2nd oldest person


Table: Drivers
-----------
Team | Name | Kilometers
-----------
Ferx | Mandy | 10000
null | null | 20000
null | null | 5000
null | null| 60000
null | null | 15000
null | null | 5000
----------

Task: ranked(2,asc,Kilometers)=10000
Example: 
"""

    taskTestPercentage = """
Table: Drivers
-----------
Team | Name | Kilometers
-----------
Ferx | Mandy | 98760
BoatK | Kean | 145000
----------

Task: read(team,name,kilometers)[*] percentage(kilometers, >)=46.82%
Example:
"""


#ok
#For team Ferx, the average kilometers driven is 12500. For team Rammers, the average kilometers driven is 32500.
    taskTest = """
\nTable: God Put a Smile upon Your Face\n------------\nAustralia (ARIA) | Italy (FIMI)\n------------\nnull | null\nnull | null\n------------\n\nTask: read(Australia (ARIA),Italy (FIMI))[*]\nExample: 
"""

# ===============================================

    # -> CHOOSE THE TASK
    prompt = "\n==========".join(examples) + "\n==========" + taskTestPercentage

    print(prompt)

    # GTP-3 example
    # response = openai.Completion.create(
    #    model="text-davinci-002",
    #    prompt=prompt,
    #    temperature=0,
    #    max_tokens=256,
    #    top_p=1,
    #    frequency_penalty=0,
    #    presence_penalty=0
    # )
    #text = response["choices"][0]['text'].strip()
    # print(prompt)
    # print(text)

    # print(len(examples))
    # exit(0)

    promptMessages = [
        #{"role": "system", "content": "You are a helpful assistant that generate text from tabular data given an example"},
        #{"role": "system", "content": "You are a helpful assistant that generate text from a given example"},
        {"role": "system", "content": "You are a helpful assistant that generate text from a given example. If you cannot complete task return 'FAILED'."},
        {"role": "user", "content": prompt}
    ]

    # CHAT GPT example
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=promptMessages,
        temperature=0.5,  # make it deterministic
        # max_tokens=256, ## max tokens in the generated output
        # top_p=1, ## alternative to temperature
        # frequency_penalty=0,
        # presence_penalty=0
        n = 2,
    )
    texts = []
    for i in range(0, len(response["choices"])):
        texts.append(response["choices"][i]["message"]["content"])
    #text = response["choices"][0]["message"]["content"]
    usedTokens = response["usage"]["total_tokens"]
    priceTokens = (usedTokens / 1000) * 0.002
    #print(text)
    print(texts)
    print("\nTokens:", usedTokens, "--- Price ($):", priceTokens)
