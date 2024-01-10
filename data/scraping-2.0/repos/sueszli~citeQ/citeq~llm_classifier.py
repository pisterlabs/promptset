from enum import Enum

from logger import LOG_SINGLETON as LOG, trace

from thefuzz import fuzz
from langchain.llms import Ollama


PROMPT = """
The following is a set of citation purpose categories, each category name is followed by a description of the category and an example of a sentence that belongs to this category.

Name: Criticizing
Description: Criticism can be positive or negative. A citing sentence is classified as “criticizing” when it mentions the weakness/strengths of the cited approach, negatively/positively criticizes the cited approach, negatively/positively evaluates the cited source.
Example: Chiang (2005) introduced a constituent feature to reward phrases that match a syntactic tree but did not yield significant improvement.

Name: Comparison
Description: A citing sentence is classified as "comparison” when it compares or contrasts the work in the cited paper to the author’s work. It overlaps with the first category when the citing sentence says one approach is not as good as the other approach. In this case we use the first category.
Example: Our approach permits an alternative to minimum error-rate training (MERT; Och, 2003);

Name: Use
Description: A citing sentence is classified as "use” when the citing paper uses the method, idea or tool of the cited paper.
Example: We perform the MERT training (Och, 2003) to tune the optimal feature weights on the development set.

Name: Substantiating
Description: A citing sentence is classified as “substantiating” when the results, claims of the citing work substantiate, verify the cited paper and support each other.
Example: It was found to produce automated scores, which strongly correlate with human judgements about translation fluency (Papineni et al. , 2002).

Name: Basis
Description: A citing sentence is classified as “basis” when the author uses the cited work as starting point or motivation and extends on the cited work.
Example: Our model is derived from the hidden-markov model for word alignment (Vogel et al., 1996; Och and Ney, 2000).

Name: Neutral (Other)
Description: A citing sentence is classified as “neutral” when it is a neutral description of the cited work or if it doesn't come under any of the above categories.
Example: The solutions of these problems depend heavily on the quality of the word alignment (Och and Ney, 2000).

Classify the following in text citation into one of these categories. Only use a single category for each citation. Respond with only a single word, the name of the category.
"""


class SentimentClass(Enum):
    CRITICIZING = 0
    COMPARISON = 1
    USE = 2
    SUBSTANTIATING = 3
    BASIS = 4
    NEUTRAL_OR_UNKNOWN = 5


class LlmClassifier:
    LLM = Ollama(model="llama2")

    @staticmethod
    def get_sentiment_class(citation: str) -> SentimentClass:
        response: str = LlmClassifier.LLM(PROMPT + citation)
        assert response

        num_words = len(response.split())
        if num_words != 1:
            LOG.warning(f"expected 1 word in response, got {num_words}")

        fst_word = "".join([c for c in response.split()[0] if c.isalnum()]).strip().lower()
        match = [
            (SentimentClass.CRITICIZING, fuzz.ratio(fst_word, "criticizing")),
            (SentimentClass.COMPARISON, fuzz.ratio(fst_word, "comparison")),
            (SentimentClass.USE, fuzz.ratio(fst_word, "use")),
            (SentimentClass.SUBSTANTIATING, fuzz.ratio(fst_word, "substantiating")),
            (SentimentClass.BASIS, fuzz.ratio(fst_word, "basis")),
            (SentimentClass.NEUTRAL_OR_UNKNOWN, fuzz.ratio(fst_word, "neutral")),
        ]
        enum_match = max(match, key=lambda x: x[1])[0]
        LOG.info(f"llm result: '{response}' → '{enum_match}'")
        return enum_match
