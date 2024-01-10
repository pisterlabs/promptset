from fastapi import HTTPException
import spacy
from src.config import get_settings, get_secret_settings
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from src.schemas.named_entity_output import NamedEntityOutput
from src.schemas.part_of_speech_output import PartOfSpeechOutput
from src.schemas.analysis_output import AnalysisOutput
from src.prompts.analysis_prompt import ANALYSIS_PROMPT
from src.prompts.sentiment_prompt import SENTIMENT_PROMPT
from src.schemas.pydantic_parser import get_analysis_project_parser
from src.schemas.pydantic_parser import get_sentiment_project_parser
from src.schemas.score_output import ScoreOutput


SETTINGS = get_settings()
SECRET_SETTINGS = get_secret_settings()

class TextAnalysisService:
    def __init__(self) -> None:
        self.nlp = spacy.load(SETTINGS.models_versions[0])
        self.llm = ChatOpenAI(model=SETTINGS.models_versions[2], openai_api_key=SECRET_SETTINGS.openai_key)
        self.llm_embeddings = OpenAIEmbeddings(model=SETTINGS.models_versions[3], openai_api_key=SECRET_SETTINGS.openai_key)
        self.analysis_parser = get_analysis_project_parser()
        self.sentiment_parser = get_sentiment_project_parser()
        self.analysis_prompt_template = PromptTemplate(
            template=ANALYSIS_PROMPT,
            input_variables=["text"],
            partial_variables={"format_instructions":self.analysis_parser.get_format_instructions()}
        )
        self.sentiment_prompt_template = PromptTemplate(
            template=SENTIMENT_PROMPT,
            input_variables=["text"],
            partial_variables={"format_instructions":self.sentiment_parser.get_format_instructions()}

        )

    def extract_part_of_speech(self, text:str) -> list[PartOfSpeechOutput]:
        doc = self.nlp(text)
        pos = [PartOfSpeechOutput(word=token.text, part_of_speech_tag=token.pos_) for token in doc]
        return pos
    def extract_named_entities(self, text:str) -> list[NamedEntityOutput]:
        doc = self.nlp(text)
        ne = [NamedEntityOutput(entity=entity.text, named_entity_tag=entity.label_) for entity in doc.ents]
        return ne
    def extract_embeddings(self, text:str) -> list:
        doc = self.nlp(text)
        embeddings = doc.vector.tolist()
        return embeddings
    
    def extract_analysis_by_gpt(self, text:str) -> AnalysisOutput:
        _input = self.analysis_prompt_template.format(text=text)
        output = self.llm.predict(_input)
        try:
            parsed = self.analysis_parser.parse(output)
        except:
            raise HTTPException(status_code=413, detail="Input text is too long for GPT-4 for process")    
        return parsed
    
    def extract_sentiment_by_gpt(self, text:str) -> ScoreOutput:
        _input = self.sentiment_prompt_template.format(text=text)
        output = self.llm.predict(_input)
        return self.sentiment_parser.parse(output)
    
    def extract_embeddings_by_gpt(self, text:str) -> list[float]:
        return self.llm_embeddings.embed_query(text)[:20]

        
    
