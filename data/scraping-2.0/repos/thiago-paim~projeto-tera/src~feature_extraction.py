import ast
import logging
import re
import openai
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import snscrape.modules.twitter as sntwitter

from src.config import OPENAI_API_KEY

openai.api_key = OPENAI_API_KEY

logger = logging.getLogger(__name__)


def get_chatgpt_sentiment_analysis(text: str) -> dict:
    base_msg = """
        Você irá realizar uma análise de sentimentos em um tweet que irei mandar.
        Sua resposta deve vir como um dicionario Python de três chaves: 'sentiment', 'value' e 'comment', com o seguinte significado:
        - sentiment: Positivo, Neutro ou Negativo
        - value: Um valor de -1 (Negativo) a 1 (Positivo)
        - comment: Uma frase explicando o motivo da sua classificação
        Por exemplo: {
            'sentiment': 'Positivo',
            'value': 0.9,
            'comment': 'O tweet expressa felicidade devido a uma boa notícia'
        }
        
        O tweet é:
    """
    msg = base_msg + text
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": msg}],
        temperature=0,
    )
    response_content = response["choices"][0]["message"]["content"]
    return ast.literal_eval(response_content)


# def get_chatgpt_sentiment_analysis_column(
#     df: pd.DataFrame,
#     text_column: str = "content",
#     new_columns: list = ["gpt_class", "gpt_value", "gpt_comment"],
#     limit: int = 10,
#     tweets_per_request: int = 5,
# ) -> pd.DataFrame:
#     """Adiciona uma nova coluna ao DataFrame com a análise de sentimento do ChatGPT"""

#     base_msg = """
#         Você irá realizar análise de sentimentos em uma série de tweets que mandarei.
#         Para cada tweet voce deve gerar um dicionario Python de três chaves: 'sentiment', 'value' e 'comment', com o seguinte significado:
#         - sentiment: Positivo, Neutro ou Negativo
#         - value: Um valor de -1 (Negativo) a 1 (Positivo)
#         - comment: Uma frase explicando o motivo da sua classificação
#         Por exemplo: {
#             'sentiment': 'Positivo',
#             'value': 0.9,
#             'comment': 'O tweet expressa felicidade devido a uma boa notícia'
#         }
#         A sua resposta deve vir no formato de uma lista com os dicionários gerados.
#         Seguem abaixo os tweets:

#     """

#     response = openai.ChatCompletion.create(
#         model="gpt-3.5-turbo",
#         messages=[{"role": "user", "content": base_msg}],
#         temperature=0,
#     )
#     response_content = response["choices"][0]["message"]["content"]
#     logger.info(response_content)


def get_twitter_username_by_url(url: str) -> str:
    """Extrai o username do Twitter a partir da URL"""
    if url and isinstance(url, str):
        url = url.rstrip("/")
        username = url.split("/")[-1]
        username = username.split("?")[0]
        username = re.findall(r"\b[0-9A-zÀ-úü]+\b", username.lower())
        return "".join(username)
    return None


def get_twitter_usernames(
    df: pd.DataFrame, url_col: str = "DS_URL", username_col: str = "TW_USER"
) -> pd.DataFrame:
    """Adiciona uma nova coluna ao DataFrame com os usernames do Twitter"""
    df[username_col] = df[url_col].apply(get_twitter_username_by_url)
    return df


def get_twitter_user_data(usernames: list) -> dict:
    """Retorna um dicionário com os dados dos usuários do Twitter"""
    user_data = {}

    for i, username in enumerate(usernames):
        if not username:
            continue
        try:
            last_tweet = next(sntwitter.TwitterProfileScraper(username).get_items())
            user_data[username] = {
                "followersCount": last_tweet.user.followersCount,
                "friendsCount": last_tweet.user.friendsCount,
                "statusesCount": last_tweet.user.statusesCount,
                "favouritesCount": last_tweet.user.favouritesCount,
                "listedCount": last_tweet.user.listedCount,
                "mediaCount": last_tweet.user.mediaCount,
            }
            logger.info(f"{i+1}/{len(usernames)} {username}: {user_data[username]}")
        except Exception as e:
            logger.error(f"{i+1}/{len(usernames)} {username}: Erro {e}")
            user_data[username] = {
                "followersCount": 0,
                "friendsCount": 0,
                "statusesCount": 0,
                "favouritesCount": 0,
                "listedCount": 0,
                "mediaCount": 0,
            }

    return user_data


def get_tweets_count(
    usernames: list, since: str = "2022-09-01", until: str = "2022-11-01"
) -> dict:
    """Retorna um dicionário com os dados dos usuários do Twitter"""
    user_tweets = {}

    for i, username in enumerate(usernames):
        if not username:
            continue
        try:
            query = f"from:{username} since:{since} until:{until}"
            user_scrapping_results = sntwitter.TwitterSearchScraper(query).get_items()
            tweets = []
            for tweet in user_scrapping_results:
                tweets.append(tweet)

            user_tweets[username] = {
                "posts": tweets,
                "count": len(tweets),
            }
            logger.info(f"{i+1}/{len(usernames)} {username}: {len(tweets)} tweets")
        except Exception as e:
            logger.error(f"{i+1}/{len(usernames)} {username}: Erro {e}")
            user_tweets[username] = {
                "posts": [],
                "count": 0,
            }

    return user_tweets


def tokenize(text):
    return [word for word in text.split()]


def bag_of_words(tweets: pd.Series) -> pd.Series:
    bow_vectorizer = CountVectorizer()
    X = bow_vectorizer.fit_transform(tweets)

    # Converte de volta para DataFrame
    X = pd.DataFrame(X.todense())

    # Atribuindo nomes das colunas aos termos
    vocabulary_map = {v: k for k, v in bow_vectorizer.vocabulary_.items()}
    X.columns = X.columns.map(vocabulary_map)
    return X
