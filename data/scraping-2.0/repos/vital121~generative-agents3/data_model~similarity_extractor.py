#!/usr/bin/python
# -*- coding: utf-8 -*-

from data_model_accessor import DataModelAccessor
import json
from openai_libs import get_score, get_tokenlen

class SimilarityExtractor:
    def __init__(self, accessor: DataModelAccessor, maxtoken_num: int):
        self.maxtoken_num = maxtoken_num
        self.accessor = accessor

    def get_filelist(self, query: str):
        scores = self._calc_scores(query, accessor.get_filelist())
        result = []
        token_sum = 0
        for entry in scores:
            if token_sum + entry["tokens"] > self.maxtoken_num:
                break
            result.append(entry["file"])
            token_sum += entry["tokens"]
        return result

    def _calc_scores(self, query: str, filelist: list):
        scores = []
        for entry in filelist:
            #print("file:", entry)
            json_data = self.accessor.get_data_model(entry).get_json_data()
            json_str = json.dumps(json_data)
            score = get_score(query, json_str)
            tokens = get_tokenlen(json_str)
            scores.append({
                "file": entry,
                "tokens": tokens,
                "score": score
            })
        scores.sort(key=lambda x: x["score"], reverse=True)
        return scores


    def extract(self, head_name: str, filelists: list):
        models = self.accessor.get_json_models(filelists)
        data = {
            head_name: models
        }
        return data


if __name__ == "__main__":
    import sys
    import json
    if len(sys.argv) != 3:
        print("Usage: <query> <dir>")
        sys.exit(1)
    query = sys.argv[1]
    dir = sys.argv[2]

    accessor = DataModelAccessor(dir)
    extractor = SimilarityExtractor(accessor, 2048)

    filelist = extractor.get_filelist(query)
    data = extractor.extract("inputs", filelist)
    data_str = json.dumps(data, indent=4, ensure_ascii=False)
    print(data_str)
