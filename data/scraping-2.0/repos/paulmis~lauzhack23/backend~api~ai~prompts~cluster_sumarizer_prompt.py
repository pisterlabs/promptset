import anthropic_bedrock

cluster_summarizer_prompt = """
    You are built in order to help with reviews!
    You will get a list of issue names of a cluster.
    Your task is to give a descriptive name to this cluster!
    The answer must be a valid json and only json. Include the ```json``` tag at the beginning of the answer.
    
    Example answer for the following cluster of issues:
    Bad sound quality, Bad audio, could not hear anything
    
    ```json
    {
        "cluster": "Audio issues"
    }
    ```
"""


def create_cluster_summarizer_prompt(
    cluster_items: list[str], system_prompt: str = cluster_summarizer_prompt
):
    return f"{anthropic_bedrock.HUMAN_PROMPT} {system_prompt}\n\n These are the issue names, please give a descriptive cluster name: {','.join(cluster_items[:10])}\n\n {anthropic_bedrock.AI_PROMPT}"
