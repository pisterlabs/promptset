import random
from openai import OpenAI


class Recommendation:
    def __init__(self, converter, io, keyhandler):
        self.converter = converter
        self.io = io
        self.keyhandler = keyhandler
        self.json_data = converter.return_data()

    def get_rec(self):
        while True:
            self.io.write("Komennot:")
            self.io.write("  1   Valitse avain lähteistä")
            self.io.write("      -r hae satunnaisella avaimella")
            self.io.write("  2   Vapaa syöttö")
            self.io.write("  0   Poistu")
            if len(self.io.inputs) == 0:  # pragma: no cover
                self.io.add_input("komento: ")  # pragma: no cover
            command = self.io.read()
            if command == "1":
                self.avain_haku()
            if command == "1 -r":
                self.get_random_key()
            if command == "2":
                self.vapaa_haku()
            if command == "0":
                break

    def vapaa_haku(self):
        while True:
            if len(self.io.inputs) == 0:  # pragma: no cover
                self.io.add_input("Kirjan nimi: ")  # pragma: no cover
            title = self.io.read()
            if title == "":
                self.io.write("\nNimi ei voi olla tyhjä\n")
            else:
                if len(self.io.inputs) == 0:  # pragma: no cover
                    self.io.add_input(
                        "(vapaaehtoinen) Kirjan kirjoittaja: ")  # pragma: no cover
                kirjoittaja = self.io.read()
                author = ", " + kirjoittaja
                if author == ", ":
                    author = ""
                    self.io.write(f"\nHaetaan kirjasuositus kirjasta {title}")
                else:
                    self.io.write(
                        f"\nHaetaan kirjasuositus kirjasta {title}, jonka kirjoittanut {kirjoittaja}")
                prompt_text = f"Pidin kirjasta {title} {author}.\
                    Anna suositus samankaltaisesta julkaisusta kirjasta josta saattaisin pitää.\
                    Anna pelkästään suositellun kirjan nimi ja kirjoittaja."
                response = self.send_prompt(prompt_text)
                self.io.write(f"\n{response}\n")
                return

    def avain_haku(self):
        key = ""
        while True:
            if len(self.io.inputs) == 0:  # pragma: no cover
                self.io.add_input(
                    "\nLähteen avain: ('ENTER' peruaksesi toiminto) ")
            input = self.io.read()
            if input == "":
                self.io.write("\nToiminto peruttu")
                return
            key = input
            existing_keys = self.keyhandler.get_keys()
            if key not in existing_keys:
                self.io.write("\nLähdettä ei löytynyt. Tarkista avain.")
            else:
                for i in range(len(self.json_data)):
                    if input.lower() in str(self.json_data[i]["key"]).lower():
                        book = self.json_data[i]
                title = book['fields']['title']
                auth = book['fields']['author']
                author = ", kirjoittanut: " + auth
                self.io.write(
                    f"\nHaetaan kirjasuositus kirjasta {title}, jonka kirjoittanut {auth}")
                prompt_text = f"Pidin kirjasta {title} {author}.\
                    Anna suositus samankaltaisesta julkaisusta kirjasta josta saattaisin pitää.\
                    Anna pelkästään suositellun kirjan nimi ja kirjoittaja."
                response = self.send_prompt(prompt_text)
                self.io.write(f"\n{response}\n")
                return

    def send_prompt(self, prompt):
        with open('src/assets/api_key.txt', 'r', encoding="utf-8") as file:
            api_key = file.read().strip()
            client = OpenAI(api_key=api_key)
            response = client.chat.completions.create(model="gpt-3.5-turbo",  # Specify the model you want to use
                                                      messages=[
                                                          {"role": "system",
                                                              "content": "Short worded."},
                                                          {"role": "user",
                                                              "content": prompt}
                                                      ],
                                                      max_tokens=50)
            return response.choices[0].message.content if response.choices else "No response"

    def get_random_key(self):  # pragma: no cover
        keys = self.keyhandler.get_keys()  # pragma: no cover
        index = random.randint(0, (len(keys) - 1))  # pragma: no cover
        key = keys.pop(index)  # pragma: no cover
        self.io.inputs = [key]  # pragma: no cover
        self.avain_haku()
