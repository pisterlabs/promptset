import joblib
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List

from skimmer.abridger import Abridger
from skimmer.span_scorer import ScoredSpan
from skimmer.openai_embedding import OpenAIEmbedding
from skimmer.openai_summarizer import OpenAISummarizer
from skimmer.parser import RightBranchingParser
from skimmer.summary_matching_scorer import SummaryMatchingScorer



memory = joblib.Memory('cache', mmap_mode='c', verbose=0)
embed = OpenAIEmbedding(memory=memory)
summarize = OpenAISummarizer(prompt_name='few-points-1', memory=memory)
parser = RightBranchingParser('en')
scorer = SummaryMatchingScorer(parser, embed, summarize)


app = FastAPI()

class ScoreRequest(BaseModel):
    text: str

@app.post("/score", response_model=List[ScoredSpan])
async def score(request: ScoreRequest) -> List[ScoredSpan]:
    return scorer(request.text)

class AbridgeRequest(BaseModel):
    text: str
    keep: float

@app.post("/abridge", response_model=str)
async def abridge(request: AbridgeRequest) -> str:

    abridger = Abridger(scorer, request.keep)

    return abridger.abridge(request.text)
