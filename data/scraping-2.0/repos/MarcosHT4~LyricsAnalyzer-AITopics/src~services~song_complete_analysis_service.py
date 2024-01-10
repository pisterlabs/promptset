from langchain.chat_models import ChatOpenAI
from fastapi import HTTPException
from langchain.prompts import PromptTemplate
from src.config import get_settings, get_secret_settings
from src.schemas.pydantic_parser import get_song_meaning_parser
from src.prompts.song_full_analysis_prompt import ANALYSIS_PROMPT
from src.schemas.song_structure import SongStructure
from src.schemas.song_meaning_output import SongMeaningOutput
from src.schemas.song_section_output import SongSectionOutput
SETTINGS = get_settings()
SECRET_SETTINGS = get_secret_settings()
class SongCompleteAnalysisService:
    def __init__(self) -> None:
        self.llm = ChatOpenAI(model=SETTINGS.models_versions[3], openai_api_key=SECRET_SETTINGS.openai_key, temperature=0.5)
        self.analysis_parser = get_song_meaning_parser()
        self.analysis_prompt_template = PromptTemplate(
            template=ANALYSIS_PROMPT,
            input_variables=["name","artist","sections", "sentiment", "emotion", "album_cover"],
            partial_variables={"format_instructions":self.analysis_parser.get_format_instructions()}
        )

    def predict(self, name:str,artist:str,song_structure:SongStructure,sentiment:SongSectionOutput, emotion:SongSectionOutput, album_cover:str ) -> SongMeaningOutput:
        while True:
            _input = self.analysis_prompt_template.format(name=name, artist=artist,sections=song_structure.sections, sentiment=sentiment, emotion=emotion, album_cover=album_cover)    
            output = self.llm.predict(_input)
            try:
                parsed = self.analysis_parser.parse(output)
            except Exception as e:
                print(e)
                continue
            break

        return parsed