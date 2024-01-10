from openai import OpenAI
client = OpenAI()

completion = client.chat.completions.create(
  model="gpt-4",
  messages=[
    {"role": "system", "content": "You will be provided with a list of headers and your goal is to return a list of objects with the headers decapitalized and with no spaces. You should convert spaces to underscores and make all letters lowercase and also remove special characters, i.e., any letter or number not a-z or 0-9. You should also inspect the name of the field and estimate what the unit of the field is. In other words, if the field was 'Air Temeprature' you should provide an estimated unit of C, meaning centigrade. If its something else, guess the unit. If you're unsure what the unit should be then put 'null'"},
    {"role": "user", "content": "['Air Temperature', 'barometric_pressure', 'Ocean Temperature 43', 'Latitude_rtk', 'relative Humidity', 'ice_thickness', 'snow Thickness mm']"}
  ]
)

print(completion.choices[0].message.content)