from langchain.output_parsers import PydanticOutputParser
from src.schemas.analysis_output import AnalysisOutput
from src.schemas.score_output import ScoreOutput
def get_analysis_project_parser():
    return PydanticOutputParser(pydantic_object=AnalysisOutput)
def get_sentiment_project_parser():
    return PydanticOutputParser(pydantic_object=ScoreOutput)