from langchain.output_parsers import PydanticOutputParser
from src.schemas.song_meaning_output import SongMeaningOutput
from src.schemas.song_full_analysis_output import SongFullAnalysisOutput
def get_song_meaning_parser():
    return PydanticOutputParser(pydantic_object=SongMeaningOutput)

def get_song_full_analysis_parser():
    return PydanticOutputParser(pydantic_object=SongFullAnalysisOutput)
