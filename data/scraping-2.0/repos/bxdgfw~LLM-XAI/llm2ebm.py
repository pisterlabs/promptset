import t2ebm
from langchain import LLMChain, PromptTemplate
from graph_desc import llm2graph_desc

#得到ebm的每个feature和对应的importance列表
def feature_importances_to_text(ebm):
    feature_importances = ""
    for feature_idx, feature_name in enumerate(ebm.feature_names_in_):  
        feature_importances += (
            f"{feature_name}: {ebm.term_importances()[feature_idx]:.2f}\n"
        )
    return feature_importances


#返回和ebm对话的LLMchain
def llm2ebm(   
        llm,
        ebm,
        memory,
        dataset_description = None,
        y_axis_description = None,
):
    feature_importances = feature_importances_to_text(ebm) 
    graphs = []
    graph_descriptions = []
    for feature_index in range(len(ebm.feature_names_in_)):       #获取ebm中的所有graph
        graphs.append(t2ebm.graphs.extract_graph(ebm, feature_index))
    graphs = [t2ebm.graphs.graph_to_text(graph) for graph in graphs]
    graph_descriptions = [llm2graph_desc(llm,ebm,idx,dataset_description=dataset_description,y_axis_description=y_axis_description) for idx in range(len(ebm.feature_names_in_)) ]
    graph_descriptions = "\n\n".join(
        [
            ebm.feature_names_in_[idx] + ": " + graph_description
            for idx, graph_description in enumerate(graph_descriptions)
        ]
    )
    
    prefix = """You are an expert statistician and data scientist.
            
    Your task is to provide an overall summary of a Generalized Additive Model (GAM) and answer the human's questions about it. The model consists of different graphs that contain the effect of a specific input feature.
    
    You will be given:
        - The global feature importances of the different features in the model.
        - Summaries of the graphs for the different features in the model. There is exactly one graph for each feature in the model.
    """
    if dataset_description is None or dataset_description == '':
        prefix += "\n\nThese inputs will be given to you by the user."
    else:
        prefix += "\n\nThe user will first provide a general description of what the dataset is about. Then you will be given the feature importance scores and the summaries of the individual features."
    
    suffix = ""
    
    if dataset_description is not None and len(dataset_description) > 0:
        suffix += "Human: Here is the general description of the data set\n" + dataset_description
        suffix += "\nAI: Thanks for this general description of the data set. Now please provide the global feature importance.\n"
    
    suffix += "Human: Here are the global feature importaces.\n\n" + feature_importances + "\nAI: Thanks. Now please provide the descriptions of the different graphs."
    suffix += "Human: Here are the descriptions of the different graphs.\n\n" + graph_descriptions
    suffix+="""\nAI: Thanks. You can ask me questions next.
    {history}
    Human: {query}
    AI:"""
    template=prefix+suffix
    prompt = PromptTemplate(input_variables=["history","query"], template=template)
    
    chain = LLMChain(
    llm = llm,
    prompt=prompt,
    verbose=False,
    memory=memory,
    )      
    return chain
