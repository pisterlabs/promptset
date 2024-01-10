from openai import OpenAI

client = OpenAI()
text= "Een ondknoping.Ik zag trans RTL-director Peter van de Vors met een map onder zijn aar met de tekst .Dus ik denk dat die vrijdag bekend gaat maken dat Matijs Van Niel krijgt bij RTL aan de slaggaat.Met actualiteit en muziek, zeg maar, op de plek van Umberto, maar die leest het wel in de media.Op één moet verdwijnen?Gert wil de nederlander op één, dan blijf nu dus alleen over, de nederlander.Donderdag.Het staat met mij tot tot tot tot tot tot tot tot tot tot tot tot tot tot tot tot tot tot tot tot tot tot tot tot tot tot tot tot tot tot tot tot tot tot tot tot tot tot tot tot tot tot tot tot tot tot tot tot tot tot tot tot tot tot tot tot tot tot tot tot tot tot tot tot tot tot tot tot tot tot tot tot tot tot tot tot tot tot tot tot tot tot tot tot tot tot tot tot tot tot tot tot tot tot tot tot"
prompt = f"Dit stuk komt uit een radio uitzending en is getranscribeerd door AI. Er kunnen fouten in zitten. Kan je eerst het categorie text geven uit `nieuws`, `muziek`, `advertentie` of rest`, en dan in max drie zinnen wat er gezegd is?\n\n{text}\n\n---\n\nSamenvatting:\n\n"

# Limit the text to 3000 tokens
prompt = prompt[:3584]

response = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[{"role": "user", "content": prompt}],
    temperature=0.7,
    max_tokens=512,
    top_p=1
    )

text = f"{text}\n\n---\n\nSamenvatting:\n\n{response.choices[0].message.content}"
print(text)