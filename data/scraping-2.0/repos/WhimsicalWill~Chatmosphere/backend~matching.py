import numpy as np
import faiss
from openai.embeddings_utils import get_embedding

from query import AsymmetricQueryHelper


class TopicMatcher:
    """
    A class that searches a vector database for the topics that are most
    semantically similar to the given query.

    Attributes:
        k (int): Number of similar topics to find.
        engine (str): The name of the embedding engine to use.
        topicInfo (list): List of topic info (topicID, userID, topicName).
        embeddings (list): List of embeddings for each topic.
        index (faiss.Index): Index for searching embeddings.
    """

    def __init__(self, llm, k=2, engine='text-embedding-ada-002'):
        """
        The constructor for TopicMatcher class.

        Parameters:
           k (int): Number of similar topics to find. Default is 2.
           engine (str): The name of the embedding engine to use. Default is 'text-embedding-ada-002'.
        """
        self.llm = llm
        self.k = k
        self.engine = engine
        self.topicInfo = []
        self.embeddings = []
        self.index = None
        self.queryHelper = AsymmetricQueryHelper(llm)

    def addTopics(self, topicTuples):
        """
        Adds a list of topics to the matcher.

        Parameters:
           topicTuples (list): A list of tuples where each tuple contains a user ID and a topic title.
        """
        for info in topicTuples:
            topicID, _, title = info
            if title == "Brainstorm": # skip Brainstorm chats
                continue
            self.topicInfo.append(info)
            self.embeddings.append(get_embedding(title, engine=self.engine))
            print(f"Added topic {topicID}")
        self.buildIndex()

    def addTopic(self, topicID, userID, title):
        """
        Adds a single topic to the matcher.

        Parameters:
           userID (str): The user ID associated with the topic.
           title (str): The title of the topic.
        """
        if title == "Brainstorm": # skip Brainstorm chats
            return
        self.topicInfo.append((topicID, userID, title))
        self.embeddings.append(get_embedding(title, engine=self.engine))
        self.buildIndex()

    def buildIndex(self):
        """
        Builds the FAISS index from the current list of embeddings.
        """
        embeddings = np.array(self.embeddings).astype('float32')
        d = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(d)
        self.index.add(embeddings)

    def searchIndexWithQuery(self, embedding, userID, k, selectedTopicIDs=None):
        """
        Retrieves the most similar topics to the provided query.

        Parameters:
           embedding (np.array): The embedding used to search the vector store.
           userID (str): The ID of the user making the query.
           k (int): The number of similar topics to return.

        Returns:
           list: A list of dictionaries, each containing the topic name, topic ID, and user ID for a similar topic.
        """
        D, I = self.index.search(embedding, 6*k)
        res = []
        for idx, score in zip(I[0], D[0]):
            topicID, userCreatorID, title = self.topicInfo[idx]
            print('Search results: ', topicID, userCreatorID, title)
            if selectedTopicIDs and topicID in selectedTopicIDs:
                continue
            if userCreatorID == userID:
                continue

            print(f"Topic {topicID} has score {score}. \nTopic: {title}\n")
            res.append({
                "topicName": title,
                "topicID": topicID,
                "userID": userCreatorID
            })
            if len(res) == k:
                break
        return res

    def getSimilarTopics(self, query, userID):
        """
        Retrieves the most similar topics to the provided query.

        Parameters:
           query (str): The query to find similar topics for.
           userID (str): The ID of the user making the query.

        Returns:
           list: A list of dictionaries, each containing the topic name, topic ID, and user ID for a similar topic.
        """
        queryEmbedding = get_embedding(query, engine=self.engine)
        queryEmbedding = np.array([queryEmbedding]).astype('float32')
        originalResults = self.searchIndexWithQuery(queryEmbedding, userID, self.k)
        return originalResults

        # TODO: profile the timing of altnerate queries, and implement them efficiently

        # alternateQueries = self.queryHelper.getAlternateQuery(query, numAlternates=5)
        # alternateEmbeddings = [get_embedding(alternateQuery, engine=self.engine) for alternateQuery in alternateQueries]
        # alternateEmbeddings = np.array(alternateEmbeddings).astype('float32')

        # numDesiredOriginal = self.k // 2
        # numDesiredAlternate = self.k - numDesiredOriginal

        # originalResults = self.searchIndexWithQuery(queryEmbedding, userID, numDesiredOriginal)
        # selectedTopics = set([topic['topicID'] for topic in originalResults])

        # print(f"Alternate queries: {alternateQueries}\n")
        # alternateResults = self.searchIndexWithQuery(alternateEmbeddings, userID, numDesiredAlternate, selectedTopics)

        # return originalResults + alternateResults