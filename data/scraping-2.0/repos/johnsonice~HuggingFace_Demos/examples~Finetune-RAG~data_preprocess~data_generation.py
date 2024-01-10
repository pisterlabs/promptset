#%%
import os,sys,openai
import pandas as pd 
from data_utils import get_completion
sys.path.insert(0,'../../libs')
from utils import load_json,exception_handler
from prompts import gen_q_basic,gen_q_fewshot,gen_q_cot
import tqdm,re
import ast
import time

key = load_json('../../../openai_key.json') 
os.environ['OPENAI_API_KEY'] = key['ChatGPT1']['API_KEY']
openai.api_key  = os.getenv('OPENAI_API_KEY')

def load_process_aiv(file_path,keep_n=None):
    aiv_df = pd.read_csv(file_path)
    aiv_df = aiv_df[(aiv_df['word_count']>80) & (aiv_df['word_count']<400)]
    aiv_df = aiv_df[2000:]
    
    if keep_n:
        aiv_df = aiv_df.sample(n=keep_n)
        
    return aiv_df

def compare_prompts_results(prompts:list,context_info,n_questions):
    for p in prompts:
        print('\n\nPrompt used: \n{} \n'.format(p['des']))
        response = get_completion(prompt=p['Human'].format(context_str=context_info,
                                                        num_questions_per_chunk=n_questions),
                        sys_msg=p['System'],
                        model='gpt-3.5-turbo')
        if p['parsing_func']:
            print(p['parsing_func'](response))
        else:
            print(response)
        
    return None

@exception_handler(error_msg=None,error_return=None)
def gen_q_by_context(prompt_template,context_info,n_questions):
    response = get_completion(prompt=prompt_template['Human'].format(context_str=context_info,
                                                        num_questions_per_chunk=n_questions),
                            sys_msg=prompt_template['System'],
                            model='gpt-3.5-turbo')
    time.sleep(1) ## wait for a sec to alleviate server side errors 
    if prompt_template['parsing_func']:
        questions = prompt_template['parsing_func'](response)
    else:
        raise Exception('Please define your result parsing function in prompt template')

    return questions 

#%%
if __name__ == "__main__":
    
    ## load raw context data 
    data_folder = '/data/LLM_DATA/Fund_docs'
    raw_aiv_file = os.path.join(data_folder,'aiv.20230820.csv')
    raw_program_file = os.path.join(data_folder,'program.20230820.csv')

    out_aiv_file = os.path.join(data_folder,'aiv_QCA_data.xlsx')
    out_program_file = os.path.join(data_folder,'program_QCA_data.xlsx')
    
    keep_n = 3000
    aiv_df = load_process_aiv(raw_aiv_file,keep_n=keep_n)
    program_df = load_process_aiv(raw_program_file,keep_n=keep_n)

    #%% ------------------------
    ## simple test question generation using different prompt
    context_info = 'To accelerate growth while maintaining macroeconomic stability, policies should continue to focus on: (i) accelerating efforts to attract investment and raise the economy’s potential by improving the business environment, reforming the large state-owned enterprise (SOE) sector, developing a market for agricultural land, and tackling corruption, which remains a key challenge; (ii) ensuring fiscal sustainability through fiscal consolidation, supported by pension reform, more efficient public spending, and a more equitable and growth-friendly tax system; (iii) further reducing inflation and rebuilding reserves; and (iv) repairing viable banks and reviving sound bank lending.'
    n_questions = 3
    compare_prompts_results([gen_q_basic,gen_q_fewshot,gen_q_cot],context_info,n_questions)
    
    #%% --------------------------------------
    ## generate questions based on given context 
    prompt_template = gen_q_basic
    n_questions = 3
    
    ### we first do aiv --------------------------
    res = []
    for context_info in tqdm.tqdm(aiv_df['par']):
        questions = gen_q_by_context(prompt_template,context_info,n_questions)
        if questions:
            add_obs = [(q,context_info,'') for q in questions] ## format it the same way as human labled data
            res.extend(add_obs)
        
        if len(res) > 0 and len(res)%100==0:
            ## export as it produces 
            res_aiv_df = pd.DataFrame(res,columns=['question','context','answer'])
            res_aiv_df.to_excel(out_aiv_file)
            
    ## put final output into a dataframe for review 
    res_aiv_df = pd.DataFrame(res,columns=['question','context','answer'])
    res_aiv_df.to_excel(out_aiv_file)

    ### then we do program ----------------------
    res = []
    for context_info in tqdm.tqdm(program_df['par']):
        questions = gen_q_by_context(prompt_template,context_info,n_questions)
        if questions:
            add_obs = [(q,context_info,'') for q in questions] ## format it the same way as human labled data
            res.extend(add_obs)
        
        if len(res) > 0 and len(res)%100==0:
            ## export as it produces 
            res_program_df = pd.DataFrame(res,columns=['question','context','answer'])
            res_program_df.to_excel(out_program_file)
            
    ## put into a dataframe for review 
    res_program_df = pd.DataFrame(res,columns=['question','context','answer'])
    res_program_df.to_excel(out_program_file)



# from llama_index import Document
# from llama_index.node_parser import SimpleNodeParser
# from llama_index.schema import TextNode, NodeRelationship, RelatedNodeInfo
# from llama_index.llms import OpenAI
# from llama_index.finetuning import generate_qa_embedding_pairs

# def generate_q_with_llamaindex():
#     '''
#     This is not very flexible for real world usages as we need to do quality check probably in excel at some point
#     '''
#     ## generate data for aiv
#     ## follow https://gpt-index.readthedocs.io/en/v0.7.7/end_to_end_tutorials/usage_pattern.html
#     ## https://github.com/run-llama/finetune-embedding/blob/main/generate_dataset.ipynb
#     text_list = aiv_df['par'].tolist()
#     #documents = [Document(text=t) for t in text_list]
#     nodes = [TextNode(text=t) for t in text_list]
#     train_nodes = nodes[:40]
#     val_nodes = nodes[40:50]
#     llm = OpenAI(
#     api_key=os.environ['OPENAI_API_KEY'],
#     model='gpt-3.5-turbo',
#     temperature=0.0
#     )   
#     qa_generate_prompt = gen_q_basic['Human']
#     train_dataset = generate_qa_embedding_pairs(train_nodes,
#                                           llm=llm,
#                                           qa_generate_prompt_tmpl=qa_generate_prompt,
#                                           num_questions_per_chunk=3)

#     val_dataset = generate_qa_embedding_pairs(val_nodes,
#                                             llm=llm,
#                                             qa_generate_prompt_tmpl=qa_generate_prompt,
#                                             num_questions_per_chunk=3)
#     #%%
#     out_path = os.path.join(data_folder,"aiv_train_dataset.json")
#     train_dataset.save_json(out_path)
#     out_path = os.path.join(data_folder,"aiv_val_dataset.json")
#     val_dataset.save_json(out_path)
    
#     return None
# %%
