import json
import boto3
import requests
import random
import openai

def lambda_handler(event, context):
    # Generate a random Pokémon ID between 1 and 151 (for the original 151 Pokémon)
    pokemon_id = random.randint(1, 151)
    
    # Fetch data about the Pokémon
    BASE_URL = 'https://pokeapi.co/api/v2/pokemon/'
    response = requests.get(BASE_URL + str(pokemon_id))
    pokemon = response.json()

    # Format the Pokémon data for the email body
    name = pokemon['name'].capitalize()
    height = str(pokemon['height'])
    weight = str(pokemon['weight'])
    types = ', '.join([type_['type']['name'] for type_ in pokemon['types']])
    
    #AI Message
    input = "Based on the following pokemon info please write an email on how to defeat the pokemon and make the email interesting. pokemon name: " + name + " Pokemon Height: " + height + " Pokemone Wight: " + weight + " Pokemone Type: " +types

    openai.api_key = 'ENTER YOUR API KEY'

    response = openai.ChatCompletion.create(
    model="gpt-4",
    messages=[
        {
        "role": "user",
        "content": input
        }
    ]
    )

    #print(response["choices"][0]["message"]["content"])

    AI_Email_Body = response["choices"][0]["message"]["content"]


    # Construct the email
    subject = "Your Random Pokémon for Today!"
    body = f"""
    <html>
    <head></head>
    <body>
      <h2>Name: {name}</h2>
      <p>Height: {height} decimetres</p>
      <p>Weight: {weight} hectograms</p>
      <p>Type(s): {types}</p>
      <p>{AI_Email_Body} </p>
    </body>
    </html>
    """

    # Send the email using AWS SES
    client = boto3.client("ses")
    message = {
        "Subject": {"Data": subject},
        "Body": {"Html": {"Data": body}}
    }
    
    response = client.send_email(
        Source="testAWSeamil6767@gmail.com",
        Destination={"ToAddresses": ["testAWSeamil6767@gmail.com"]},
        Message=message
    )
    
    return response


