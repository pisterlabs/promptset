from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from src.config import get_settings, get_secret_settings
from src.schemas.pydantic_parser import get_song_meaning_parser
from src.prompts.song_meaning_prompt import MEANING_PROMPT
from src.schemas.song_structure import SongStructure
from src.schemas.song_meaning_output import SongMeaningOutput 
SETTINGS = get_settings()
SECRET_SETTINGS = get_secret_settings()
class SongMeaningAnalysisService:
    def __init__(self) -> None:
        self.llm = ChatOpenAI(model=SETTINGS.models_versions[3], openai_api_key=SECRET_SETTINGS.openai_key, temperature=0.5)
        self.meaning_parser = get_song_meaning_parser()
        self.analysis_prompt_template = PromptTemplate(
            template=MEANING_PROMPT,
            input_variables=["sections"],
            partial_variables={"format_instructions":self.meaning_parser.get_format_instructions()}
        )

    def predict(self, song_structure:SongStructure) -> SongMeaningOutput:
        _input = self.analysis_prompt_template.format(sections=song_structure.sections)    
        output = self.llm.predict(_input)
        try:
            parsed = self.meaning_parser.parse(output)
        except Exception as e:
            parsed = SongMeaningOutput(sections=[])
        return parsed    
        