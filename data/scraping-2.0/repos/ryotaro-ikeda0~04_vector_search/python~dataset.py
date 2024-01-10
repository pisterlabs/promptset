
import pandas as pd
from abc import ABC, abstractmethod
from langchain.docstore.document import Document
from tabulate import tabulate


class BaseDataset(ABC):

    def __init__(self):
        pass

    def get_docs(self) -> list[Document]:
        '''ドキュメントを取得
        Returns:
            list[Document]:
        '''
        return self.docs

    def get_query(self) -> list:
        return self.df_query[['query_id', 'query']].to_numpy()

    def __len__(self) -> int:
        return len(self.docs)

    def filtering(self, df_query: pd.DataFrame, df_product: pd.DataFrame, df_label: pd.DataFrame, n: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        '''データをn件に絞る
        Args:
            df_query (pd.DataFrame):
            df_product (pd.DataFrame):
            df_label (pd.DataFrame):
            n (int): 絞る件数

        Returns:
            tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        '''
        if n is None:
            return df_query, df_product, df_label

        df_product = df_product.iloc[:n, :]
        df_label = df_label[df_label['product_id'].isin(df_product['product_id'].to_numpy())]
        df_query = df_query[df_query['query_id'].isin(df_label['query_id'].to_numpy())]

        return df_query, df_product, df_label

    def create_docs(self, df_product) -> list[Document]:
        return [
            Document(
                page_content=product['product_description'],
                metadata={
                    "document_id": product['product_id'],
                    "title": product['product_name'],
                }
            )
            for _, product in df_product.iterrows()
        ]

    @abstractmethod
    def get_score(self) -> list:
        pass


class WandDataset(BaseDataset):

    def __init__(self, n=None):
        path = "../data/WANDS/dataset/"
        df_query = pd.read_csv(f"{path}/query.csv", sep='\t')
        df_product = pd.read_csv(f"{path}/product.csv", sep='\t')
        df_label = pd.read_csv(f"{path}/label.csv", sep='\t')

        self.df_query, self.df_product, self.df_label = self.filtering(df_query, df_product, df_label, n)
        self.docs = self.create_docs(self.df_product)

    # query_idとproduct_id_listからスコアを取得
    def get_score(self, query_id: str, product_id_list: list) -> list:
        df = pd.DataFrame(data=[{
            'query_id': query_id,
            'product_id': product_id
        } for product_id in product_id_list])

        df = pd.merge(df, self.df_label, on=['query_id', 'product_id'], how='left')

        # df['label'] = self.df_label[(self.df_label['query_id'] == query_id) & self.df_label['product_id'].isin(product_id_list)]['label']
        df['gain'] = df['label'].replace({'Exact': 1, 'Partial': 0.5, 'Irrelevant': 0})
        df['gain'] = df['gain'].fillna(0)

        return df['gain'].tolist()


class HomeDepotDataset(BaseDataset):

    def __init__(self):
        path = "../data/home_depot/dataset/"
        self.df_query = pd.read_csv(f"{path}/query.csv", encoding="ISO-8859-1")
        self.df_product = pd.read_csv(f"{path}/product.csv")
        self.df_label = pd.read_csv(f"{path}/label.csv")
        self.docs = self.create_docs(self.df_product)

    # query_idとproduct_id_listからスコアを取得
    def get_score(self, query_id: str, product_id_list: list) -> list:
        df = pd.DataFrame(data=[{
            'query_id': query_id,
            'product_id': product_id
        } for product_id in product_id_list])

        df = pd.merge(df, self.df_label, on=['query_id', 'product_id'], how='left')
        df['relevance'] = df['gain'].fillna(0)

        return df['relevance'].tolist()