# 참고: https://github.com/langchain-ai/langchain/issues/7685
import pandas as pd
from langchain.tools import BaseTool
from typing import Union, Dict, Tuple
class LatestmovieTool(BaseTool):
    name = "get_latest_movies"
    description = "Get the current playing movies. No parameter needed from input"

    def _to_args_and_kwargs(self, tool_input: Union[str, Dict]) -> Tuple[Tuple, Dict]:
        return (), {}

    def _run(self):
        response = get_latest_movies()
        return response

    def _arun(self):
        raise NotImplementedError("get_latest_movies does not support async")

def get_latest_movies() -> dict:
    """
    Returns: 다음 영화에서 크롤링한 최신 영화 정보를 리턴한다.
    현재는 그냥 데이터프레임을 넘기지만 나중에 agent화 할 예정.
    """
    return pd.read_csv('./data/daum_movie/movie_list.csv')[['제목','평점','예매율','개봉일','줄거리']].head().to_dict()