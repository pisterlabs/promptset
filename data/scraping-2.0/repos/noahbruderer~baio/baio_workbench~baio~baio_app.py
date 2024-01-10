import os
import streamlit as st
from st_app.file_manager import FileManager
from langchain.callbacks import get_openai_callback
from st_app.helper_functions import save_uploaded_file
from src.llm import LLM
    
UPLOAD_DIR = "./baio/data/uploaded/"
DOWNLOAD_DIR = './baio/data/output/'

side_bar_txt_path='./baio/st_app/text_content/side_bar_text.txt'
aniseed_instruction_txt_path = './baio/st_app/text_content/aniseed_agent_instructions.txt'
go_file_annotator_instruction_txt_path = './baio/st_app/text_content/go_annotator_instructions.txt'
csv_instruction_txt_path = './baio/st_app/text_content/csv_chatter_instructions.txt'
ncbi_instruction_txt_path = './baio/st_app/text_content/ncbi_agent_instructions.txt'

LICENSE_path = './baio/st_app/text_content/LICENSE.txt'

#initialising paths 
base_dir = os.getcwd()  # Gets the current working directory from where the app is launched
path_aniseed_out = os.path.join(base_dir, 'baio', 'data', 'output', 'aniseed', 'aniseed_out.csv')
path_go_nl_out = os.path.join(base_dir, 'baio', 'data', 'output', 'gene_ontology', 'go_annotation.csv')
go_file_out = os.path.join(base_dir, 'baio', 'data', 'output', 'gene_ontology', 'go_annotation.csv')

def read_txt_file(path):
    with open(path, 'r') as file:
        return file.read()
        
def app():
    st.sidebar.markdown('''# PROVIDE AN OpenAI API KEY Xß:''')

    banner_image = "./baio/data/persistant_files/baio_logo.png"  
    st.image(banner_image, use_column_width=True)  
    openai_api_key = st.sidebar.text_input('OpenAI API KEY')
    model_options = ['gpt-4-1106-preview', 'gpt-4', 'gpt-4-32k', 'gpt-3.5-turbo']
    default_model = 'gpt-4'
    selected_model = st.sidebar.selectbox(
        'Select a model',
        model_options,
        index=model_options.index(default_model)  # Set the default option by index
    )
    
    # Check if the "Reinitialize LLM" button is clicked or if the llm is not in session state
    if st.sidebar.button('Reinitialize LLM') or 'llm' not in st.session_state:
        if openai_api_key:
            LLM.initialize(openai_api_key=openai_api_key, selected_model=selected_model)
            st.sidebar.success(f'LLM reinitialized with selected model!')  # Show success message in the sidebar
        else:
            st.sidebar.error('Please provide an OpenAI API key.')  # Show error message in the sidebar
    st.sidebar.markdown(read_txt_file(side_bar_txt_path), unsafe_allow_html=True)
    if st.sidebar.button('Show License'):    
        st.sidebar.markdown(read_txt_file(LICENSE_path)) 
    if openai_api_key:
        os.environ["OPENAI_API_KEY"] = openai_api_key
        go_file_annotator = 'Local GO agent'
        file_chatter = 'Local file agent'
        ncbi = 'NCBI'
        selected_agent = st.radio("Choose an agent:", ["BaIO agent", go_file_annotator, file_chatter, ncbi])
        from src.mytools.go_tool import go_file_tool
        from src.agents.ncbi_agent import ncbi_agent
        from src.agents.aniseed_agent import aniseed_go_agent
        from src.agents.file_annotator_agent import file_annotator_agent
        from src.mytools.csv_chatter_tool import filechatter_instructions
        from src.agents.csv_chatter_agent import csv_agent_creator
        def create_csv_agent(file_paths):
            valid_file_paths = [fp for fp in file_paths if fp is not None]
            if valid_file_paths:
                return csv_agent_creator(valid_file_paths)
            return None
       
        # ######
        # ######      ANISEED AGENT
        # ######

        if selected_agent == "BaIO agent":
            file_manager_aniseed = FileManager()
            with st.form('form_for_aniseed_agent'):
                st.write("BaIO agent") 
                with st.expander("Instructions"): st.markdown(read_txt_file(aniseed_instruction_txt_path))
                question = st.text_area('Enter text for ANISEED agent:', 'Example: What genes are expressed between stage 1 and 3 in ciona robusta?')
                submitted = st.form_submit_button('Submit')
                reset_memory = st.form_submit_button('Clear chat history')

                # st.write(path_aniseed_out, path_go_nl_out)
                # Add the file paths to the file_paths dictionary
                            
                if submitted:
                    with get_openai_callback() as cb:
                        try:
                            result = aniseed_go_agent(question)
                            st.info(result['output'])
                            st.info(f"Total cost is: {cb.total_cost} USD")
                            st.write(f"Your generated file is below:")
                            #change: output last tool used by agent in order to select previeved file
                            if 'GO' in str(result['chat_history'][1]) or 'Gene Ontology' in str(result['chat_history'][1]) or 'entrez' in str(result['chat_history'][1]) or 'ensembl' in str(result['chat_history'][1]):
                                file_manager_aniseed.preview_file(path_go_nl_out)
                                st.markdown(file_manager_aniseed.file_download_button(path_aniseed_out), unsafe_allow_html=True)                                
                            else:                                
                                file_manager_aniseed.preview_file(path_aniseed_out)
                                st.markdown(file_manager_aniseed.file_download_button(path_aniseed_out), unsafe_allow_html=True)
                                
                        except:
                            st.write('Something went wrong, please try to reforumulate your question')
                if reset_memory:
                    aniseed_go_agent.memory.clear()  
            file_manager = FileManager(UPLOAD_DIR, DOWNLOAD_DIR)
            file_manager.run()                                     


        ####
        ####    FILE GO ANNOTATOR
        ####
        elif selected_agent == go_file_annotator:
            
            go_file_annotator_file_manager= FileManager(UPLOAD_DIR, DOWNLOAD_DIR)
            st.write("Local GO agent")             
            with st.expander("Instructions"): st.markdown(read_txt_file(go_file_annotator_instruction_txt_path))
            uploaded_file = st.file_uploader("Upload your file below:", type=["csv"])
            if uploaded_file:
                st.write("You've uploaded a file!")
                save_uploaded_file(uploaded_file, UPLOAD_DIR)
                go_file_annotator_file_manager= FileManager(UPLOAD_DIR, DOWNLOAD_DIR)
                
            file_path = go_file_annotator_file_manager.select_file_preview_true(key="file_path")    
  
            with st.form('form_for_file_annotator_agent'):
                
                input_file_gene_name_column = st.text_area('Enter gene name column:', 'gene_name')
                submitted = st.form_submit_button('Submit')

                if submitted:
                    with get_openai_callback() as cb:

                        result =go_file_tool(file_path, input_file_gene_name_column)
                        go_file_annotator_file_manager= FileManager(UPLOAD_DIR, DOWNLOAD_DIR)

                        try:
                            go_file_annotator_file_manager.preview_file(result[1])
                            st.markdown(go_file_annotator_file_manager.file_download_button(result[1]), unsafe_allow_html=True)                                
                            st.info(cb.total_cost)
                        except TypeError:
                            st.info(result)     
            
            file_manager = FileManager("./baio/data/output/", "./baio/data/upload/")
            file_manager.run()                       
        ####
        ####    FILE CHATTER
        ####

        #CHAT WITH YOUR CSV AND OTHER FILES
        elif selected_agent == file_chatter:
            csv_chatter_file_manager= FileManager(UPLOAD_DIR, DOWNLOAD_DIR)
            st.write("Ask questions about your selected or uploaded file")
            with st.expander("Instructions"): st.markdown(read_txt_file(csv_instruction_txt_path))
            uploaded_file = st.file_uploader("Upload your file with genes here", type=["csv", "xlsx", "txt"])
            if uploaded_file:
                st.write("You've uploaded a file!")
                save_uploaded_file(uploaded_file, UPLOAD_DIR)
                csv_chatter_file_manager= FileManager(UPLOAD_DIR, DOWNLOAD_DIR)            
            st.write('Select file 1:')
            file_path1 = csv_chatter_file_manager.select_file_preview_false(key="select_file_1")   
            st.write('Select file 2:')
            file_path2 = csv_chatter_file_manager.select_file_preview_false(key="select_file_2")       
            

            with st.form('test form'):
                st.write("Explore a file ")
                question = st.text_area('Enter text:', 'Example: What are the unique genes per stage? please make a new data frame of them and put it in a file')
                reset_memory = st.form_submit_button('Clear chat history')

                submitted = st.form_submit_button('Submit')

                if submitted:
                    files = [file_path1, file_path2]

                    if len(files) != 0 :
                        csv_agent = create_csv_agent(files)
                                            
                    with get_openai_callback() as cb:
                        result = csv_agent.run(filechatter_instructions + question)            
                        try:
                            st.info(result['output'])
                            st.info(cb.total_cost)
                        except TypeError:
                            st.info(result)

                        if reset_memory and csv_agent:
                            csv_agent.memory.clear()   
                                                         
            file_manager = FileManager("./baio/data/output/", "./baio/data/upload/")
            file_manager.run()     
            
        ####
        ####    NCBI
        ####            
            
        elif selected_agent == ncbi:
            file_manager_aniseed = FileManager()
            with st.form('form_for_aniseed_agent'):
                st.write("NCBI agent") 
                with st.expander("Instructions"): st.markdown(read_txt_file(ncbi_instruction_txt_path))
                question = st.text_area('Enter question for NCBI agent:', 'Which organism does the DNA sequence come from:AGGGGCAGCAAACACCGGGACACACCCATTCGTGCACTAATCAGAAACTTTTTTTTCTCAAATAATTCAAACAATCAAAATTGGTTTTTTCGAGCAAGGTGGGAAATTTTTCGAT')
                submitted = st.form_submit_button('Submit')
                reset_memory = st.form_submit_button('Clear chat history')

                # st.write(path_aniseed_out, path_go_nl_out)
                # Add the file paths to the file_paths dictionary
                            
                if submitted:
                    with get_openai_callback() as cb:
                        try:
                            result = ncbi_agent(question)
                            st.info(result)
                            st.info(f"Total cost is: {cb.total_cost} USD")
                            st.write(f"Your generated file is below:")             
                        except:
                            st.write('Something went wrong, please try to reformulate your question')
                if reset_memory:
                    ncbi_agent.memory.clear()  
            file_manager = FileManager(UPLOAD_DIR, DOWNLOAD_DIR)
            file_manager.run()                                     

           
    else:
        st.write('version 0.0.1')
        st.markdown('<p style="font-size:48px;text-align:center;">AGENTS EXECUTE CODE ON YOUR MACHINE, IT IS RECOMMENDED TO USE BAIO IN A CONTAINER</p>', unsafe_allow_html=True)

        st.markdown('<p style="font-size:24px;text-align:center;">To use the <b>ANISEED & GO term</b> and <b>Local file explorer</b> you must provide a valid OpenAI API key</p>', unsafe_allow_html=True)

        st.markdown("""
        
        This is an application connecting the users questions and data to the internet and a coding agent.
        (Agent: an autonomous computer program or system that is designed to perceive its environment, interprets it, plans actions and executes them in order to achieve a defined goal.)
        It connects Large Language Models, such as OpenAI's ChatGPT, to public databases such as NCBI, Ensembl, and ANISEED, as well as the user's local files.
        BaIO allows users to query these databases with natural language and annotate files with relevant information, including GO terms, Ensembl IDs, and RefSeq IDs.
        BaIO is built on the Python LangChain library and various tools developed by myself and the user interface is rendered with Streamlit.
        """)
        st.markdown('# BaIO agent')
        st.markdown(read_txt_file(aniseed_instruction_txt_path))
        st.markdown('# Local GO Agent')
        st.markdown(read_txt_file(go_file_annotator_instruction_txt_path))
        st.markdown('# Local file agent')
        st.markdown(read_txt_file(csv_instruction_txt_path))
        st.markdown('# LICENSE')
        st.markdown(read_txt_file(LICENSE_path))

if __name__ == '__main__':
    app()