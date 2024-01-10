import openai
import pickle
import numpy as np
from typing import List
from openai.embeddings_utils import cosine_similarity
from tenacity import retry, wait_random_exponential, stop_after_attempt
import os

openai.api_key = os.getenv("OPENAI_API_KEY")

"""
Code search and relevance models:
	code-search-ada-code-001
	code-search-ada-text-001
	code-search-babbage-code-001
	code-search-babbage-text-001
"""


@retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(6))
def get_embedding(text: str, engine="text-similarity-davinci-001") -> List[float]:
    # replace newlines, which can negatively affect performance.
    text = text.replace("\n", " ")
    return openai.Embedding.create(input=[text], engine=engine)["data"][0]["embedding"]


def get_embedding_vector(snippet, engine="text-similarity-davinci-001"):
    snippet_embedding = get_embedding(snippet, engine)
    return np.array(snippet_embedding)


def snippet_similarity(snippet_1, snippet_2):
    snippet_1_embedding = get_embedding(snippet_1, engine='code-search-ada-code-001')
    snippet_2_embedding = get_embedding(snippet_2, engine='code-search-ada-code-001')
    cos_similarity = cosine_similarity(snippet_1_embedding, snippet_2_embedding)
    return cos_similarity


def main():
    code_snippet_1 = """
    # java
testRead_01bit_MoreThanOneBandIsUnsupported() {
  try {
    testReadFullLevel0(org.geotools.arcsde.raster.gce.TYPE_1BIT, 2);
    org.junit.Assert.fail("Expected<sp>IAE");
  } catch (java.lang.IllegalArgumentException e) {
    "<AssertPlaceHolder>";
  }
}
"""

    code_snippet_2 = """
    # java
    testCreateNetworkApi() {
      resource.configure("BrocadeVcsResource", parameters);
      when(api.createNetwork(com.cloud.network.resource.BrocadeVcsResourceTest.VLAN_ID, com.cloud.network.resource.BrocadeVcsResourceTest.NETWORK_ID)).thenReturn(true);
      final com.cloud.agent.api.CreateNetworkCommand cmd = new com.cloud.agent.api.CreateNetworkCommand(com.cloud.network.resource.BrocadeVcsResourceTest.VLAN_ID, com.cloud.network.resource.BrocadeVcsResourceTest.NETWORK_ID, "owner");
      final com.cloud.agent.api.CreateNetworkAnswer answer = ((com.cloud.agent.api.CreateNetworkAnswer)(resource.executeRequest(cmd)));
      "<AssertPlaceHolder>";
    }
    """
    embedding_vectors = []
    code_snippets = [code_snippet_1, code_snippet_2]

    for code_snippet in code_snippets:
        embedding = get_embedding_vector(code_snippet)
        print(f"embedding vector:{embedding}")
        embedding_vectors.append(embedding)

    with open('embedding_vectors.pkl', 'wb') as outfile:
        pickle.dump(embedding_vectors, outfile, pickle.HIGHEST_PROTOCOL)

    embedding_vectors_from_file = []
    with open('embedding_vectors.pkl', 'rb') as inputFile:
        embedding_vectors_from_file = pickle.load(inputFile)

    for a_vector in embedding_vectors_from_file:
        print(a_vector)

    # score = snippet_similarity(code_snippet_1, code_snippet_2)
    # print(f"similarity score:{score}")


if __name__ == "__main__":
    main()
