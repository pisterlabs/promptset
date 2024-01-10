from langchain.docstore.document import Document
from libreco.algorithms import LightGCN
from libreco.data import DataInfo


class RecSys:
    def __init__(self, model_path: str):
        data_info = DataInfo.load(model_path, "recsys_model")
        self.model = LightGCN.load(model_path, "recsys_model", data_info)

    def rec_from_relevant(self, nickname: str, docs: list[Document]) -> list[Document]:
        item_ids = list(map(lambda x: x.metadata["uid"], docs))

        scores = self.model.predict(
            user=nickname, item=item_ids, cold_start="popular"
        ).astype(object)

        filtered = []
        for uid, score in sorted(
            zip(item_ids, scores), key=lambda x: x[1], reverse=True
        ):
            if not score > 0:
                break
            for doc in docs:
                if doc.metadata["uid"] == uid:
                    filtered.append(doc)

        return filtered

    def recommend(self, nickname: str, n_rec: int):
        response = self.model.recommend_user(user=nickname, n_rec=n_rec)[nickname]
        return list(response.astype(object))
