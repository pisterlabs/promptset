#返回llm对ebm某个feature的graph description

import t2ebm
from langchain.memory import ConversationBufferMemory 
from langchain import LLMChain, PromptTemplate
from langchain.chat_models import ChatOpenAI

def llm2graph_desc(   
        llm,
        ebm,
        feature_idx,
        dataset_description = None,
        y_axis_description = None,
        query = "Now please provide a brief, at most 7 sentence summary of the influence of the feature on the outcome."
):
    prefix = """You are an expert statistician and data scientist.
You interpret global explanations produced by a generalized additive model (GAM). GAMs produce explanations in the form of graphs that contain the effect of a specific input feature.\n
"""
    if dataset_description is None or dataset_description == '':
        prefix +="""You will be given graphs from the model, and the user will ask you questions about the graphs."""
    else:
        prefix +="""The user will first provide a general description of the dataset. Then you will be given graphs from the model, and the user will ask you questions about the graphs.\n"""
    
    prefix +="""\n\nAnswer all questions to the best of your ability, combining both the data contained in the graph"""
    
    if dataset_description is not None and len(dataset_description) > 0:
        prefix +=""", the data set description you were given, and your knowledge about the real world."""
    else:
        prefix +=""" and your knowledge about the real world."""
    
    prefix +="""Graphs will be presented as a JSON object with keys representing the x-axis and values representing the y-axis. For continuous features, the keys are intervals that represent ranges where the function predicts the same value. For categorical features, each key represents a possible value that the feature can take."""
    
    if y_axis_description is not None and len(y_axis_description) > 0:
        prefix +=y_axis_description
    prefix += """\n\nThe user will provide graphs in the following format:
        - The name of the feature depicted in the graph
        - The type of the feature (continuous, categorical, or boolean)
        - Mean values
        - Lower bounds of confidence interval
        - Upper bounds of confidence interval
    """
    if dataset_description is not None and len(dataset_description) > 0:
        prefix += dataset_description + """\nThe description of dataset ends.\n"""
    graph = t2ebm.graphs.extract_graph(ebm, feature_idx)
    graph = t2ebm.graphs.graph_to_text(graph)
    graph = graph.replace("{", "(").replace("}", ")")

    suffix="""\nBegin!
Human: Consider the following graph from the model.\n"""
    suffix+=graph
    suffix+="""\nAI: I have obtained the information of the graph. You can ask me questions next, and I will answer based on the information,my knowledge about the real world, and maybe the data description.
Human: {query}
AI:"""
    template=prefix+suffix
    prompt = PromptTemplate(input_variables=["query"], template=template)
    
    chain = LLMChain(
    llm = llm,
    prompt=prompt,
    verbose=False,
    )
    graph_description = chain.run(query=query)
    return graph_description