from langchain.chains import LLMChain
from langchain import PromptTemplate, FewShotPromptTemplate


class TopicSegway:
    """
    A class that uses a language model to generate engaging responses to a given query,
    in the context of a series of topic names.

    Attributes:
        llm (OpenAI): Language model to generate responses.
        chain (LLMChain): LLMChain instance to help structure and generate responses.
        few_shot_prompt (FewShotPromptTemplate): Few-shot prompt to guide the language model.
    """

    def __init__(self, llm):
        """
        The constructor for TopicSegway class.
        """
        self.configurePrompt()
        self.chain = LLMChain(llm=llm, prompt=self.few_shot_prompt)

    def configurePrompt(self):
        """
        Configures the few-shot prompt to be used by the language model.

        Sets up the few-shot prompt with examples and structure.
        """
        example_1 = {
            "query": "How will technology shape the future?",
            "topic1": "How is artificial intelligence impacting our daily lives?",
            "topic2": "What do you think about the future of cryptocurrency?",
            "answer": "You might enjoy discussing how AI technology will fit into our future.\n" \
                      "You could explore the lasting impact of cryptocurrency.\n"
        }

        example_2 = {
            "query": "What are the impacts of climate change?",
            "topic1": "How does climate change affect wildlife?",
            "topic2": "What are the economic consequences of climate change?",
            "answer": "You might find it interesting to discuss how climate change is affecting wildlife.\n" \
                      "You might enjoy conversing about how climate change will affect the economy.\n"
        } 

        examples = [example_1, example_2]

        template =  """
        Query: {query}
        Topic 1: {topic1}
        Topic 2: {topic2}
        Answer: {answer}
        """

        # Define the structure of the prompt with input variables and template
        example_prompt = PromptTemplate(
            input_variables=["query", "topic1", "topic2", "answer"], 
            template=template,
        )

        # Define the prefix for the prompt, giving clear instructions on how to construct an engaging response
        prompt_prefix = "Given the user's query, suggest two topics of discussion. For each topic, " \
                        "craft an intriguing line explaining why the topic could be of interest to the user. " \
                        "Make sure that you give the user a logical reason why they may be interested in the topics. " \
                        "Please put a new line between each topic suggestion, since your response will be invalid without this. " \
                        "Here are some examples:\n"
        
        prompt_suffix = """
        Query: {query}
        Topic 1: {topic1}
        Topic 2: {topic2}
        Answer:"""

        # Generate the few-shot prompt with the provided examples and structure
        self.few_shot_prompt = FewShotPromptTemplate(
            examples=examples,
            example_prompt=example_prompt,
            prefix=prompt_prefix,
            suffix=prompt_suffix,
            input_variables=["query", "topic1", "topic2"],
            example_separator="\n",
        )
        print("Set up few shot prompt")

    def getResponse(self, query, topics):
        """
        Generates a response to a given query in the context of a series of topic names.

        Parameters:
           query (str): The query to generate a response for.
           topics (list): A list of topic dictionaries with the keys 'topicName', 'topicID', and 'userID'.

        Returns:
           str: The generated response to the query.
        """
        print(f"Generating response for query {query}")

        assert len(topics) == 2, f"Must provide two topics, not {len(topics)}. Topics: {topics}"

        # Assuming topics is a list of three topicNames
        input = {
            "query": query,
            "topic1": topics[0]['topicName'],
            "topic2": topics[1]['topicName'],
        }

        print("Input:", input)
        response = self.chain.run(input)
        print("Response:", response)
        return response