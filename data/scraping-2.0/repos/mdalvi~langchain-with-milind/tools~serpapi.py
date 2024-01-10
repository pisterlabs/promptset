from langchain.serpapi import SerpAPIWrapper


class CustomSerpAPIWrapper(SerpAPIWrapper):
    @staticmethod
    def _process_response(res: dict) -> str:
        """Process response from SerpAPI."""
        if "error" in res.keys():
            raise ValueError(f"Got error from SerpAPI: {res['error']}")
        if "answer_box" in res.keys() and type(res["answer_box"]) == list:
            res["answer_box"] = res["answer_box"][0]
        if "answer_box" in res.keys() and "answer" in res["answer_box"].keys():
            to_return = res["answer_box"]["answer"]
        elif "answer_box" in res.keys() and "snippet" in res["answer_box"].keys():
            to_return = res["answer_box"]["snippet"]
        elif (
            "answer_box" in res.keys()
            and "snippet_highlighted_words" in res["answer_box"].keys()
        ):
            to_return = res["answer_box"]["snippet_highlighted_words"][0]
        elif (
            "sports_results" in res.keys()
            and "game_spotlight" in res["sports_results"].keys()
        ):
            to_return = res["sports_results"]["game_spotlight"]
        elif (
            "shopping_results" in res.keys()
            and "title" in res["shopping_results"][0].keys()
        ):
            to_return = res["shopping_results"][:3]
        elif (
            "knowledge_graph" in res.keys()
            and "description" in res["knowledge_graph"].keys()
        ):
            to_return = res["knowledge_graph"]["description"]
        elif "link" in res["organic_results"][0].keys():  # link first
            to_return = res["organic_results"][0]["link"]
        elif "snippet" in res["organic_results"][0].keys():
            to_return = res["organic_results"][0]["snippet"]
        elif (
            "images_results" in res.keys()
            and "thumbnail" in res["images_results"][0].keys()
        ):
            thumbnails = [item["thumbnail"] for item in res["images_results"][:10]]
            to_return = thumbnails
        else:
            to_return = "No good search result found"
        return to_return


def search(text: str) -> str:
    """
    Google web search for given text and returns results.
    :param text: search string
    :return: result
    """
    serp_api = CustomSerpAPIWrapper()
    return serp_api.run(f"{text}")
