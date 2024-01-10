from backend.acropora_model.helper.climate_engine import ClimateEngine
import pandas as pd
import numpy as np
import openai

class ModelWrapper:
    def __init__(self, key, 
                 search_model_name = 'babbage', answer_model = 'text-davinci-002',
                 embed_loc='backend/acropora_model/docs/embed/{}/', folder_template = 'backend/acropora_model/template/ready/samples/'):
        # prepare the engine
        openai.api_key = key
        self.kube = ClimateEngine(openai)
        # prepare model details
        self.doc_search_model = 'text-search-{}-doc-001'.format(search_model_name)
        self.query_search_model = 'text-search-{}-query-001'.format(search_model_name)
        self.answer_model = answer_model
        # prepare the paths
        folder_embed = embed_loc.format(search_model_name)
        self.path_embed_illinois = folder_embed + 'illinois/'
        self.path_embed_jersey = folder_embed + 'new_jersey/'
        self.folder_template = folder_template
        # load the data
        self.load_data()
        self.load_templates()
    
    def load_data(self):
        ## read dataframes
        path = self.path_embed_illinois+'IL State - 2022 LTP.csv' #'IL State - 2022 LTP.pdf'
        df_il_state = pd.read_csv(path) 
        df_il_state['extend_embed'] = df_il_state['extend_embed'].apply(eval).apply(np.array)


        path = self.path_embed_illinois+'IL ComEd - Rate Schedule 11.2022.csv' #'IL ComEd - Rate Schedule 11.2022.pdf'
        df_il_ComEd = pd.read_csv(path)
        df_il_ComEd['extend_embed'] = df_il_ComEd['extend_embed'].apply(eval).apply(np.array)

        path = self.path_embed_illinois+'IL Ameren - TPO NEM 11.2022.csv' #'IL Ameren - TPO NEM 11.2022.pdf'
        df_il_Ameren = pd.read_csv(path)
        df_il_Ameren['extend_embed'] = df_il_Ameren['extend_embed'].apply(eval).apply(np.array)
        #----
        path = self.path_embed_jersey+'NJ SuSi Program.csv' #'NJ SuSi Program.pdf'
        df_nj_state = pd.read_csv(path)
        df_nj_state['extend_embed'] = df_nj_state['extend_embed'].apply(eval).apply(np.array)

        path = self.path_embed_jersey+'NJ JCP&L - Rate Schedule 11.2022.csv' #'NJ JCP&L - Rate Schedule 11.2022.pdf'
        df_nj_JCP = pd.read_csv(path)
        df_nj_JCP['extend_embed'] = df_nj_JCP['extend_embed'].apply(eval).apply(np.array)

        path = self.path_embed_jersey+'NJ ACE - Rate Schedule 11.2022.csv' #'NJ ACE - Rate Schedule 11.2022.pdf'
        df_nj_ACE = pd.read_csv(path)
        df_nj_ACE['extend_embed'] = df_nj_ACE['extend_embed'].apply(eval).apply(np.array)



        illi_dict = {'state': df_il_state, 'comed': df_il_ComEd, 'ameren': df_il_Ameren}
        jersy_dict = {'state': df_nj_state, 'jcp&l': df_nj_JCP, 'ace':df_nj_ACE}
        self.df_look_up = {'il':illi_dict, 'nj':jersy_dict}
    
    def load_templates(self):
        # read all templates
        self.template_main = self.read_file_lines(self.folder_template+'main_context_sample.txt')
        self.template_il = self.read_file_lines(self.folder_template+'IL_context_sample.txt')
        self.template_nj = self.read_file_lines(self.folder_template+'NJ_context_sample.txt')
    
    def read_file_lines(self, file_name):
        lines = ""
        with open(file_name,'r', encoding='utf-8-sig') as f:
            lines = f.readlines()

        return '\n'.join(lines)
    
    def get_answer(self, question, state, utility, N=2, template='main'):
    
        # fix
        if utility == 'N/A': utility = 'state'
        
        state = state.lower()
        utility = utility.lower()

        # validation
        assert state.lower() in ['il', 'nj']
        assert utility.lower() in ['state', 'jcp&l', 'comed', 'ameren', 'ace', ]

        if template == 'main':
            template = ''+self.template_main
        else:
            template = ''+self.template_nj if state == 'nj' else ''+self.template_il


        # get the dataframe 
        search_data = self.df_look_up[state][utility]

        # search the top n items

        meta_df = self.kube.get_top_n_metadata(search_data, question, 
                                         self.query_search_model, top_n = N)
        
        ## create context
        context = '\n'.join(meta_df.content.values)
        context = context.replace('\n', ' ')
        context = context[15:-15]
        ### create reference
        books = ','.join(list(set(meta_df.book_title.values)))
        pages = meta_df.page_num.values
        reference = 'Answer from {} in pages {}'.format(books, pages)
        
        
        model_input = template.replace('{{query_context}}', context).replace('{{query}}', question).replace('\n\n', '\n')

        intro = 'You are energy developer expert, answer the user question from the context\n'
        model_input = intro + model_input

        # get the answer
        answer = self.kube.get_best_answer(self.answer_model, model_input, log=False)

        return answer, reference