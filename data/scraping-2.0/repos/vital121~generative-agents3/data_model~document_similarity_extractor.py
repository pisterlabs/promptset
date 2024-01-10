#!/usr/bin/python
# -*- coding: utf-8 -*-

import sys
import json
from data_model_accessor import DataModelAccessor
from document_data_model import DocumentDataModel
from openai_libs import get_score, get_tokenlen

class DocumentSimilarityExtractor:
    def __init__(self, accessor: DataModelAccessor, maxtoken_num: int):
        self.maxtoken_num = maxtoken_num
        self.accessor = accessor
        self.load()

    def load(self):
        filelist = self.accessor.get_filelist()
        self.models = []
        for entry in filelist:
            filepath = self.accessor.get_data_model_filepath(entry)
            model = DocumentDataModel.load_json_file(filepath)
            self.models.append(model)

    def _calc_scores(self, query: str):
        self.scores = []
        for model in self.models:
            contents = model.get_contents()
            if contents is None:
                continue
            for entry in model.get_contents():
                #print("info:", entry)
                data = json.dumps(entry["Answer"])
                score = get_score(query, data)
                tokens = get_tokenlen(data)
                self.scores.append({
                    "term": model.get_title(),
                    "info": entry["Answer"],
                    "tokens": tokens,
                    "score": score
                })
        self.scores.sort(key=lambda x: x["score"], reverse=True)

    def extract(self, query: str):
        self._calc_scores(query)
        terms = {}
        token_sum = 0
        for entry in self.scores:
            if token_sum + entry["tokens"] > self.maxtoken_num:
                break
            if terms.get(entry["term"]) is None:
                terms[entry["term"]] = []
            terms[entry["term"]].append(entry["info"])
            token_sum += entry["tokens"]

        data = {
            "DocumentIDs": terms
        }
        return data

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: <query> <dir> <max_tokens>")
        sys.exit(1)
    query = sys.argv[1]
    dir = sys.argv[2]
    max_tokens = int(sys.argv[3])

    accessor = DataModelAccessor(dir)
    extractor = DocumentSimilarityExtractor(accessor, max_tokens)
    data = extractor.extract(query)
    data_str = json.dumps(data, indent=4, ensure_ascii=False)
    print(data_str)
