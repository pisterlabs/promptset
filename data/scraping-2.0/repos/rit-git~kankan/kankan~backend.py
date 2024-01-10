import sys
sys.path.append('.')

from fastapi import FastAPI
from pydantic import BaseModel
from array import array
import uvicorn
import os
import heapq
import csv

import openai
from openai.embeddings_utils import get_embedding

from vecscan.vecscan import PyVectorScanner

from genre import genre_all, genre_food, STR_GENRE_ALL, STR_GENRE_NO_FOOD

class KankanRequest(BaseModel):
    tdfk: str
    query: str
    genre: str

class KankanAPI(FastAPI):
    def __init__(self):
        super().__init__()
        openai.api_key=os.environ["OPENAI_API_KEY"]

        self.scanners = {}
        self.raw_data = {}
        self.genre_data = {}
        self.vec_file_path_base = '/home/01052711/kankan/dataset/sentvecs.ada.kuchikomi_report.vec'
        self.raw_file_path_base = '/home/01052711/kankan/dataset/jalan_kanko.csv'
        self.genre_file_path_base = '/home/01052711/kankan/dataset/jalan_kanko.genre'
        self.n_best_hotel = 10
        self.n_best_kuchikomi = 10
        self.post("/api/search")(self.search)

    def search(self, req: KankanRequest):
        if req.tdfk not in self.scanners:
            self.scanners[req.tdfk] = PyVectorScanner(f'{self.vec_file_path_base}.{req.tdfk}', '')
            self.raw_data[req.tdfk] = []
            for line_idx, line in enumerate(open(f'{self.raw_file_path_base}.{req.tdfk}.csv')):
                if line_idx > 0:
                    self.raw_data[req.tdfk].append(line.strip())
            for row in csv.DictReader(open(f'{self.genre_file_path_base}.{req.tdfk}.csv')):
                self.genre_data[row['odk_id']] = row['genre']

        genre_accept = set()
        if req.genre == STR_GENRE_ALL:
            genre_accept = genre_all
        elif req.genre == STR_GENRE_NO_FOOD:
            genre_accept = genre_all - genre_food
        else:
            genre_accept.add(req.genre)
        
        scan_results = self.scanners[req.tdfk].n_best_vecs(
            query_vec=array('f', get_embedding(req.query, engine='text-embedding-ada-002')),
            n_best=100000,
            openmp=True,
            debug=True,
            threshold=0.50
        )[0]

        spot_result = {}
        full_odk_set = set()
        scan_results_len = scan_results.size()
        for i in range(scan_results_len):
            idx, score = scan_results.get(i)
            fields = self.raw_data[req.tdfk][idx].strip().split(',')
            kuchikomi_id, odk_id, other = fields[0], fields[1], ','.join(fields[2:])
            if self.genre_data[odk_id] not in genre_accept:
                continue
            if odk_id not in spot_result:
                spot_result[odk_id] = {'score': 0.0, 'kuchikomi': [], 'cnt': 0, 'genre': self.genre_data[odk_id]}
            if spot_result[odk_id]['cnt'] < self.n_best_kuchikomi:
                spot_result[odk_id]['kuchikomi'].append((score, other))
                spot_result[odk_id]['score'] += score
                if spot_result[odk_id]['cnt'] > 0:
                    spot_result[odk_id]['score'] /= 2
                spot_result[odk_id]['cnt'] += 1
            if spot_result[odk_id]['cnt'] == self.n_best_kuchikomi:
                if True: # len(full_odk_set) < self.n_best_hotel:
                    full_odk_set.add(odk_id)

        if len(full_odk_set) >= self.n_best_hotel:
            result_list = [spot_result[k] for k in full_odk_set]
            minor_list = [spot_result[k] for k in (set(spot_result.keys()) - full_odk_set)]
        else:
            result_list = spot_result.values()
            minor_list = []
        result_list = heapq.nlargest(self.n_best_hotel, result_list, key=lambda x: x['score'])
        minor_list = heapq.nlargest(self.n_best_hotel, minor_list, key=lambda x: x['score'])

        ret = {'hotel': [], 'hotel_minor': []}
        for __hotel_key, __hotel_result in zip(['hotel', 'hotel_minor'], [result_list, minor_list]):
            for spot_rank, kuchikomi_list in enumerate(__hotel_result):
                ret[__hotel_key].append(
                    {
                        'rank': spot_rank + 1,
                        'score': kuchikomi_list['score'],
                        'genre': kuchikomi_list['genre'],
                        'kuchikomi': []
                    }
                )
                for kuchikomi in kuchikomi_list['kuchikomi']:
                    fields = kuchikomi[1].split(',')
                    ret[__hotel_key][-1]['kuchikomi'].append(
                        {
                            'score': kuchikomi[0],
                            'rate': fields[0],
                            'title': fields[1],
                            'text': ','.join(fields[2:-5]),
                            'date': fields[-5] + '/' + fields[-4],
                            'name': fields[-3],
                            'address': fields[-2],
                            'ybn': fields[-1]
                        }
                    )

        return ret
    
def main():
    app = KankanAPI()
    uvicorn.run(
        app,
        port=21344,
        root_path='/app/kankan'
    )

if __name__ == '__main__':
    main()