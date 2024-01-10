import os
import openai
from filterSpecialCaracters import filter_special_characters
from IA.generatePicture import generate_picture_univers

openai.api_key = os.getenv("OPENAI_API_KEY")
my_engine = os.getenv("OPENAI_ENGINE")

class Personnage:
    def __init__(self, name):
        self.id = None
        self.name = name
        self.descriptionOfPersonnage = None
        self.univers_id = None
        self.user_id = None

    def to_map(self):
        return {
            'id': self.id,
            'name': self.name,
            #'description': self.descriptionOfPersonnage
            'univers_id': self.univers_id,
            'user_id': self.user_id,
        }

    @classmethod
    def from_map(cls, map):
        personnage = cls(map.get('name', ''))
        personnage.id = map.get('id')
        personnage.descriptionOfPersonnage = map.get('description')
        personnage.univers_id = map.get('univers_id')
        personnage.user_id = map.get('user_id')

        return personnage

    def generate_descriptionOfPersonnage(self, univers):
        # Générer avec OpenAI
        # Utiliser OpenAI pour générer une description d'un univers
        response = openai.Completion.create(
            engine= my_engine, # Choisir le moteur de génération de texte
            prompt = f"Provide me with a description of the character {self.name} from the universe {univers}. Share their history, personality, and specific traits with me.",
            #prompt=f"Give me an English description of the {self.name} character from the {univers} universe.", 
            max_tokens=200,  # Limitez le nombre de tokens pour contrôler la longueur de la réponse
            n=1,  # Nombre de réponses à générer
            stop=None  # Vous pouvez spécifier des mots pour arrêter la génération
        )
        reponse = response.choices[0].text.strip()

        filtered_text = filter_special_characters(reponse)
        
        self.descriptionOfPersonnage = filtered_text

        isDescription = 2

        generate_picture_univers(self, self.name, self.descriptionOfPersonnage, isDescription, univers)
        
        #return self.descriptionOfPersonnage
    
        #self.description = f"Description du personnage {self.name} générée par OpenAI"

    def generate_new_descriptionOfPersonnage(self,new_name, univers):
        # Générer avec OpenAI
        # Utiliser OpenAI pour générer une description d'un univers
        response = openai.Completion.create(
            engine= my_engine, # Choisir le moteur de génération de texte
            prompt=f"Provide me with a description of the character {new_name} from the universe {univers}. Share their history, personality, and specific traits with me.", 
            max_tokens=200,  # Limitez le nombre de tokens pour contrôler la longueur de la réponse
            n=1,  # Nombre de réponses à générer
            stop=None  # Vous pouvez spécifier des mots pour arrêter la génération
        )
        reponse = response.choices[0].text.strip()

        filtered_text = filter_special_characters(reponse)
        
        self.new_descriptionOfPersonnage = filtered_text
        
        return self.new_descriptionOfPersonnage
    
        #self.new_descriptionOfPersonnage = f"Description du personnage {new_name} générée par OpenAI"