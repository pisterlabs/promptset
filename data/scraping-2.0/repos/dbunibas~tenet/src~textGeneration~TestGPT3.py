import openai

from src import Constants

if __name__ == '__main__':
    openai.api_key = Constants.API_KEYS_GPT3[0]
    prompt = """
<table>Persons</table>
<important> compare (<) <header>Year</header> </important>
<important> 35 <header>Year</header> </important>
<header>Name</header><header>Year</header>
<row><cell>Enzo</cell><cell>35</cell></row>
<row><cell>Paolo</cell><cell>45</cell></row>
<row><cell>Jonh</cell><cell>35</cell></row>
example: Enzo and John are 35 and both are younger than Paolo.

<table>Persons</table>
<important> compare (>) <header>Year</header> </important>
<important> 45 <header>Year</header> </important>
<header>Name</header><header>Year</header>
<row><cell>Jack</cell><cell>45</cell></row>
<row><cell>Enzo</cell><cell>35</cell></row>
<row><cell>Paolo</cell><cell>45</cell></row>
example: Jack and Paolo are 45 and both are older than Enzo.

<table>Persons</table>
<important> compare (=) <header>Year</header> </important>
<important> 25 <header>Year</header> </important>
<header>Name</header><header>Year</header>
<row><cell>Enzo</cell><cell>25</cell></row>
<row><cell>Paolo</cell><cell>25</cell></row>
example: Enzo and Paolo have the same age.

<table>Persons</table>
<header>Name</header><header>Year</header>
<row><cell>Enzo</cell><cell>35</cell></row>
<row><cell>Paolo</cell><cell>45</cell></row>
example: Enzo is 35 and Paolo is 45 years old.

<table>Persons</table>
<important> max <header>Year</header> </important>
<important> 35 <header>Year</header> </important>
<header>Name</header><header>Year</header>
<row><cell>Enzo</cell><cell>35</cell></row>
<row><cell>Paolo</cell><cell>22</cell></row>
<row><cell>John</cell><cell>35</cell></row>
example: John and Enzo are the oldest with 35 years old.

<table>Persons</table>
<important> min <header>Year</header> </important>
<important> 22 <header>Year</header> </important>
<header>Name</header><header>Year</header>
<row><cell>John</cell><cell>22</cell></row>
<row><cell>Enzo</cell><cell>35</cell></row>
<row><cell>Paolo</cell><cell>22</cell></row>
example: Paolo and John are the yougest with 22 years old.

<table>Persons</table>
<important> count 3 </important>
<header>Name</header><header>Age</header>
<row><cell>Enzo</cell><cell>36</cell></row>
<row><cell>Paolo</cell><cell>46</cell></row>
<row><cell>Mike</cell><cell>18</cell></row>
example: There are three persons. Namely Enzo, Paolo and Mike.

<table>Persons</table>
<important> average <header>Age</header> </important>
<important> 33.33 <header>Age</header> </important>
<header>Name</header><header>Age</header>
<row><cell>Enzo</cell><cell>36</cell></row>
<row><cell>Paolo</cell><cell>46</cell></row>
<row><cell>Mike</cell><cell>18</cell></row>
example: The average age is 33.33.

<table>Persons</table>
<important> count 3 </important>
<important> (>) 12 <header>Age</header></important>
<header>Name</header><header>Age</header>
<row><cell>Enzo</cell><cell>36</cell></row>
<row><cell>Paolo</cell><cell>46</cell></row>
<row><cell>Mike</cell><cell>18</cell></row>
example: There are three persons with age greater than 12. Namely Enzo, Paolo and Mike.

<table>Persons</table>
<important> count 3 </important>
<important> (=) ITA <header>Country</header></important>
<header>Name</header><header>Country</header>
<row><cell>Enzo</cell><cell>ITA</cell></row>
<row><cell>Paolo</cell><cell>ITA</cell></row>
<row><cell>Mike</cell><cell>ITA</cell></row>
example: There are three persons from Italy. Namely Enzo, Paolo and Mike.

<table>Persons</table>
<important> sum ITA <header>Income</header> = 30</important>
<important> sum USA <header>Income</header> = 70</important>
<important> sum ITA <header>Country</header>  compare (<) sum USA <header>Country</header></important>
<header>Name</header><header>Country</header><header>Income</header>
<row><cell>Enzo</cell><cell>ITA</cell><cell>10</cell></row>
<row><cell>Paolo</cell><cell>ITA</cell><cell>20</cell></row>
<row><cell>Mike</cell><cell>USA</cell><cell>30</cell></row>
<row><cell>Albert</cell><cell>USA</cell><cell>40</cell></row>
example: The total income from Italian is 30 and is lower than the total income from USA which is 70.

"""
    prompt += """
<table> Drivers </table>
<important> average Ferx <header>Kilometers</header> = 12500</important>
<important> average Rammers <header>Kilometers</header> = 32500</important>
<important> average Rammers <header>Kilometers</header> compare (>) average Ferx <header>Kilometers</header></important>
<header>Team</header><header>Driver</header><header>Kilometers</header>
<row><cell>Ferx</cell><cell>Mandy</cell><cell>10000</cell></row>
<row><cell>Ferx</cell><cell>Carleston</cell><cell>20000</cell></row>
<row><cell>Rammers</cell><cell>Tronky</cell><cell>5000</cell></row>
<row><cell>Rammers</cell><cell>RetroBar</cell><cell>60000</cell></row>
<row><cell>Ferx</cell><cell>KiloKalo</cell><cell>15000</cell></row>
<row><cell>Ferx</cell><cell>Amber</cell><cell>5000</cell></row>
example:    
"""

    s = """
<row><cell>Ferx</cell><cell>Mandy</cell><cell>13345</cell></row>
<row><cell>Bisk</cell><cell>Carleston</cell><cell>24678</cell></row>
<row><cell>Rammers</cell><cell>Tronky</cell><cell>4333</cell></row>
<row><cell>Ferx</cell><cell>RetroBar</cell><cell>60000</cell></row>
<row><cell>Ferx</cell><cell>KiloKalo</cell><cell>60000</cell></row>
<row><cell>Ferx</cell><cell>Amber</cell><cell>4333</cell></row>
"""

    ## GTP-3 example
    #response = openai.Completion.create(
    #    model="text-davinci-002",
    #    prompt=prompt,
    #    temperature=0,
    #    max_tokens=256,
    #    top_p=1,
    #    frequency_penalty=0,
    #    presence_penalty=0
    #)
    #text = response["choices"][0]['text'].strip()
    #print(prompt)
    #print(text)

    promptMessages = [
      #{"role": "system", "content": "You are a helpful assistant that generate text from tabular data given an example"},
      {"role": "system", "content": "You are a helpful assistant that generate text from a given example"},
      {"role": "user", "content": prompt}
    ]

#    promptTestSQL = "SELECT c.name, p.name FROM Cities c, Politicians p WHERE c.population > '1M' AND p.age < 40 AND p.name = c.currentMayor"
    # promptTestSQL = "SELECT TOP 10 c.cityName, cm.name, cm.birthDate FROM city c, cityMayor cm WHERE c.major = cm.name AND cm.electionYear = 2019"
#    promptMessages = [
#        {"role": "system", "content": "You are an helpful assistant that returns the answer as a set of tuples given a SQL query. Use your training data to answer with real values. Do not explain the query, just report the data."},
#        {"role": "user", "content": promptTestSQL},
#    ]

    ## CHAT GPT example
    response = openai.ChatCompletion.create(
          model="gpt-3.5-turbo",
          messages= promptMessages,
          temperature = 0.0, ## make it deterministic
          #max_tokens=256, ## max tokens in the generated output
          #top_p=1, ## alternative to temperature
          #frequency_penalty=0,
          #presence_penalty=0
    )
    text = response["choices"][0]["message"]["content"]
    usedTokens = response["usage"]["total_tokens"]
    priceTokens = (usedTokens / 1000) * 0.002
    print(text)
    print("Tokens:", usedTokens, "--- Price ($):", priceTokens)