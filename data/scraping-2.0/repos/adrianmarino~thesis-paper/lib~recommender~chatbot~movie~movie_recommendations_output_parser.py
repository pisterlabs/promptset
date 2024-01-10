from langchain.schema import BaseOutputParser
from bunch import Bunch
from typing import List
import util as ut
import re
import logging
from pydantic import BaseModel, PrivateAttr


class MovieRecommendationsOutputParser(BaseOutputParser[List[str]]):
    __list_size: int = PrivateAttr(True)

    def __init__(self, list_size):
        super().__init__()
        self.__list_size = list_size

    def parse(self, text: str) -> List[str]:
            results = []
            for idx in range(self.__list_size):
                try:
                    line = ut.between(text, f'{idx+1}.', f'{idx+2}.')

                except Exception as e:
                    logging.error(f'Error to parse response. {e}')
                    return results

                data = re.split(r'\(|\)\:', line)

                if len(data) <= 1:
                    continue

                results.append({
                    'title'      : data[0].strip().replace('"', '').capitalize(),
                    'description': data[2].strip().capitalize(),
                    'release'    : int(data[1].strip())
                })

            return { 'recommendations': results }