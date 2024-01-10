import json
import os
from datetime import datetime, timedelta
from utils.utils import chunk_text
from langchain.chat_models import AzureChatOpenAI
from langchain.schema import HumanMessage, SystemMessage
from azure.identity import DefaultAzureCredential

GPT4_DEPLOYMENT_NAME = os.getenv("GPT4_DEPLOYMENT_NAME", "gpt4")
GPT35_DEPLOYMENT_NAME = os.getenv("GPT35_DEPLOYMENT_NAME", "gpt-35-turbo-16k")
GPT4_32_DEPLOYMENT_NAME = os.getenv("GPT4_32_DEPLOYMENT_NAME", "gpt-4-32k")
AZURE_OPENAI_API_BASE = os.getenv("AZURE_OPENAI_API_BASE")

SUMMARY_PROMPT = os.getenv("SUMMARY_PROMPT", "Summarize the following text:")
#COVER_PROMPT = os.getenv("COVER_PROMPT", "INSTRUCTION : Crée une quatrième de couverture à partir du résumé du livre qu'il t'es donné. Tu dois présenter la trame principale de l'intrigue, sans dévoiler la fin ou la résolution. Tu dois également présenter les personnages principaux et les lieux principaux. Tu dois également donner envie au lecteur de lire le livre, sans interpeller le lecteur. Tu utiliseras le persona du lecteur qu'il t'es donné pour adapter ton résultat en fonction de ses centres d'intérêts, sans dire au lecteur que tu connais son persona. Tu peux par exemple cloturer en posant une question ou en donnant un apercu des premières aventures du personnage principal. le résultat ne dois pas dépasser 200 mots. Tu dois adapter ton style au persona qu'il t'es décrit. Tu donneras uniquement la quatrième de couverture sans introduction, analyse ou commentaire.")
COVER_PROMPT = os.getenv("COVER_PROMPT", "INSTRUCTION : Crée une quatrième de couverture à partir du résumé du livre qu'il t'es donné. Tu dois présenter la trame principale de l'intrigue, sans dévoiler la fin ou la résolution. Tu dois donner envie au lecteur de lire le livre, sans interpeller le lecteur. Tu peux par exemple cloturer en posant une question ou en donnant un apercu des premières aventures du personnage principal. le résultat ne dois pas dépasser 200 mots. Tu dois adapter ton style au PERSONA qui t'es décrit. Tu donneras uniquement la quatrième de couverture sans introduction, analyse ou commentaire.")

PIA_PROMPT = os.getenv("PIA_PROMPT", "PERSONA : Pia lit beaucoup de genres et de styles différents, des classiques à la fiction contemporaine, de la poésie aux essais. Elle aime découvrir de nouveaux auteurs et de nouvelles tendances, mais aussi revisiter ses anciens favoris. Elle est toujours curieuse et ouverte d'esprit en ce qui concerne les livres.")
FRANCOIS_PROMPT = os.getenv("FRANCOIS_PROMPT", "PERSONA : François lit principalement des romans littéraires, en particulier des auteurs français et européens. Il apprécie la prose bien écrite, les personnages complexes et les thèmes riches. Il n'est pas très intéressé par les romans de genre ou les best-sellers populaires. Il aime discuter de livres avec ses amis et sa famille.")
JEAN_PROMPT = os.getenv("JEAN_PROMPT", "PERSONA : Jean aime les résumés qui sont écrits avec un langage urbain et familier.")
OLYMPE_PROMPT = os.getenv("OLYMPE_PROMPT", "PERSONA : Olympe lit principalement des livres d'art et d'histoire, en particulier ceux qui sont bien documentés, informatifs et illustrés. Elle aime apprendre de nouvelles choses et approfondir ses connaissances sur divers sujets. Elle n'est pas très intéressée par les livres de fiction ou les livres contemporains. Elle préfère acheter des livres auprès de sources réputées et d'experts.")
PERSONAS_PROMPTS = {
    "Pia": PIA_PROMPT,
    "François": FRANCOIS_PROMPT,
    "Jean": JEAN_PROMPT,
    "Olympe": OLYMPE_PROMPT
}


THEME_PROMPT_LIST = ['Littérature', ' Philosophie', ' Sciences', ' Histoire', ' Politique', ' Economie', ' Art', ' Religion', ' Société', ' Psychologie', ' Biographie', ' Crime', ' Fantaisie', ' Science-fiction', ' Horreur', ' Romance', ' Jeunesse', ' Technologie', ' Drame', ' Tourisme', ' Sport', ' Cuisine', ' Voyage', ' Environnement', ' Education', ' Santé', ' Humour', ' Musique', ' Cinéma', ' Télévision', ' Médias', ' Mode', ' Vie quotidienne', ' Famille', ' Amour', ' Amitié', ' Sexualité', ' Animaux', ' Nature', ' Aventure', ' Jeux', ' Science', ' Mathématiques', ' Physique', ' Chimie', ' Biologie', ' Géologie', ' Astronomie', ' Médecine', ' Psychiatrie', ' Sociologie', ' Géographie', ' Droit', ' Littérature', ' Langues']
THEME_PROMPT = os.getenv("THEME_PROMPT", f"""Tu dois classer l'extrait de livre parmis les catégories qui te sont proposées. Le résultat sera donné sous forme d'un tableau ["catégorie1", "catégorie2"...] avec les catégories dont la probabilité est supérieur à 0.7. Les catégories sont les suivantes : {', '.join(THEME_PROMPT_LIST)}. L'extrait est le suivant : """)

OPENAI_API_TYPE = "azure_ad"
OPENAI_API_VERSION = "2023-05-15"

class CognitiveSkills:
    _instance = None
    _gpt35turbo = None
    _gpt4 = None
    _gpt432 = None
    _azure_credential = None
    _openai_credential = None
    _openai_credential_expiration = None

    def __new__(cls, *args, **kwargs):
        if not isinstance(cls._instance, cls):
            cls._instance = super(CognitiveSkills, cls).__new__(cls, *args, **kwargs)
        return cls._instance

    def get_llm_instance(cls, gpt_version):
        if cls._openai_credential_expiration is None or cls._openai_credential_expiration < datetime.now():
            cls._openai_credential_expiration = datetime.now() + timedelta(hours=8)
        
        cls._azure_credential = DefaultAzureCredential()
        cls._openai_credential = cls._azure_credential.get_token("https://cognitiveservices.azure.com/.default").token
        
        match gpt_version:
            case "gpt-35-turbo-16k":
                cls._gpt35turbo = AzureChatOpenAI(
                    openai_api_base=AZURE_OPENAI_API_BASE,
                    openai_api_key=cls._openai_credential,
                    openai_api_type=OPENAI_API_TYPE,
                    openai_api_version=OPENAI_API_VERSION,
                    deployment_name=GPT35_DEPLOYMENT_NAME,
                )
                return cls._gpt35turbo
            case "gpt4":
                cls._gpt4 = AzureChatOpenAI(
                    openai_api_base=AZURE_OPENAI_API_BASE,
                    openai_api_key=cls._openai_credential,
                    openai_api_type=OPENAI_API_TYPE,
                    openai_api_version=OPENAI_API_VERSION,
                    deployment_name=GPT4_DEPLOYMENT_NAME,
                )
                return cls._gpt4
            case "gpt-4-32k":
                cls._gpt432 = AzureChatOpenAI(
                    openai_api_base=AZURE_OPENAI_API_BASE,
                    openai_api_key=cls._openai_credential,
                    openai_api_type=OPENAI_API_TYPE,
                    openai_api_version=OPENAI_API_VERSION,
                    deployment_name=GPT4_32_DEPLOYMENT_NAME,
                )
                return cls._gpt432
        return cls._gpt35turbo
            
        
    def summarize(cls, text):
        chunks = chunk_text(text, 3000)
        summary = ""
        llm_instance = cls.get_llm_instance("gpt-35-turbo-16k")

        for index, chunk in enumerate(chunks):
            print(f"{index}/{len(chunks)}")
            
            try:
                response = llm_instance([
                    HumanMessage(content=SUMMARY_PROMPT),
                    HumanMessage(content=chunk)
                ])
                summary += f"{response.content} "

            except Exception as e:
                print(f"error : {e}")
                continue
            
        return summary
    
    def create_cover(cls, text, prompt):
        chunks = chunk_text(text, 30000)
        llm_instance = cls.get_llm_instance("gpt-4-32k")

        try:
            response = llm_instance([
                HumanMessage(content=prompt),
                HumanMessage(content=chunks[0])
            ])
            return response.content

        except Exception as e:
            print(f"error: {e}")
            return f"error : {e}", 500
                
        
    def get_prompts(cls):
        return PERSONAS_PROMPTS
    
    def create_themes(cls, text):
        chunks = chunk_text(text, 25000)
        llm_instance = cls.get_llm_instance("gpt-4-32k")

        try:
            retry = 3
            while(retry > 0):
                response = llm_instance([
                    HumanMessage(content=THEME_PROMPT),
                    HumanMessage(content=chunks[0])
                ])

                print(f"Detected themes : {response.content}")
                
                try:
                    themes = json.loads(response.content.strip())
                    return themes
                except Exception as e:
                    print(f"error: {e}")
                    retry -= 1
                    continue
            
            return []
        
        except Exception as e:
            print(f"error: {e}")
            return f"error : {e}", 500

    def create_covers(cls, text):
        res = []

        chunks = chunk_text(text, 28000)
        llm_instance = cls.get_llm_instance("gpt-4-32k")


        # pour tous les prompts de PERSONNAS_PROMPTS, on fait un create_cover avec comme key le nom du personna et comme value le prompt
        for persona, prompt in PERSONAS_PROMPTS.items():
            try:
                response = llm_instance([
                    SystemMessage(content=f"{COVER_PROMPT} {prompt}"),
                    HumanMessage(content=f'LIVRE: {chunks[0]}')
                ])
                res.append({
                    "persona_name": persona,
                    "content": response.content,
                })
                print(f"Cover for {persona} : {response.content}")

            except Exception as e:
                print(f"error: {e}")
                return f"error : {e}", 500

        return res



