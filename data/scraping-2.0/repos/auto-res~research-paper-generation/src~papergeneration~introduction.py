from langchain.llms import OpenAI
import warnings
warnings.filterwarnings('ignore')



def create_introduction(area_of_expertise_, journal_, experts_, objective_, title_, abstract_, reference_list_, num_words = 300):
    model_name="gpt-3.5-turbo"
    llm = OpenAI(temperature=0, model_name=model_name)

    # 論文を書く上での前提条件の入力
    assumption = """
    #Here  is your definition and the order to follow;
    ・Act as an {area_of_expertise} professor.
    ・You are about to write a scientific paper that will win the award of the most-read article.
    ・You are submitting this manuscript to {journal}, which has a high impact factor.
    ・The readers of the manuscript are {experts} experts.
    ・Feel free to ignore irrelevant information in the reference.
    ・Don’t hesitate to ask any questions to improve the quality of the manuscript.
    ・After reading, just answer, “Got it.”
    """

    assumption = assumption.format(area_of_expertise = area_of_expertise_, 
                                        journal = journal_,
                                        experts = experts_)

    # 先行研究のインプット
    # LangChainで記憶の保持をさせてもいいかも
    input = """
    This is a reference for the research paper you are about to write. 
    Please read it and keep it in mind. After reading, just answer, "Got it." 
    Feel free to ignore irrelevant information given in the reference.
    ---------
    {reference}
    """

    for i in reference_list_:
        input = input.format(reference = i)
        llm(input)


    # introduction生成のためのプロンプト
    # 生成する長さを長くしすぎると同じ子を繰り返してします
    input2 = """
    ・Using the provided reference sentences and {abstract}, write an introduction of the scientific paper entitled {title}.
    ・Write scientific and academic sentences with beautiful and elegant, upper-level English words citing supporting literature.
    1. In the first paragraph, describe general information of {objective} and explain what is known in previous research about the topic of this research for the {objective}.
    2. The second paragraph provides an explanation of unknown facts of {objective} along with the research motivation.
    3. The third paragraph mentions the purpose of this study
    ・The introduction should be approximately {num_words} words in total length.
    """

    input2 = assumption + input2.format(abstract = abstract_,
                                        title = title_,
                                        objective = objective_,
                                        num_words = num_words)
    
    introduction = llm(input2)

    return introduction