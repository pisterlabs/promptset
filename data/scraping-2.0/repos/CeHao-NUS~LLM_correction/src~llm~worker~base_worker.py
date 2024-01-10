from src.utils.py_utils import AttrDict, ParamDict
from src.llm.openai.completor_vision import OpenAICompletorVision
from src.llm.parser.exp.base_exp_parser import BaseExpParser
from src.llm.parser.assist.base_assist_parser import BaseAssistParser

from src.utils.file_utils import *
import re, ast
class BaseWorker:
    def __init__(self):
        self._hp = self._default_hparams() # default hparams
        self._setup()
        self.clear()
    
    def _default_hparams(self):
        # default hparams
        default_dict = ParamDict({
            'llm': OpenAICompletorVision,   
            'exp_parser': BaseExpParser,
            'assist_parser': BaseAssistParser,
            'assist_checker': None,
            'system_prompt': './src/llm/prompts/system.txt',
            'user_prompt': './src/llm/prompts/users/u1.txt',
            
        })
        return default_dict
    
    
    def reflection(self, exp_results):
        # formulate reflection
        
        # 1. parse exp results
        exp_prompt, images = self.exp_parser.parse(exp_results)
        
        # 2. initial analysis
        user_prompt = self.user_prompt.format(exp_prompt=exp_prompt)
        user_prompt  = user_prompt + '''
        Please answer the questions (1) and (2)
        '''
        assist_results1 = self.llm.answer_with_image(user_prompt, images)
        
        # 3. reflect success
        checker_success = False
        while not checker_success:
            print('reflect success')
            llm_results = self.reflect_success()
            if 'success' in llm_results:
                if llm_results['success'] in [True, False]:
                    checker_success = True
                else:
                    print('llm success is unknown, try again')
            else:
                print('llm success failed, try again')
        self.llm_results.update(llm_results)

        # 4. reflect improve
        if not llm_results['success']:        
            
            checker_improve = False
            while not checker_improve:
                print('reflect improve')
                llm_results = self.reflect_improve()
                if 'change of goal' in llm_results:
                    checker_improve = True

            self.llm_results.update(llm_results)
        
        # 5. save all results
        all_results = self.llm.get_all_answers()
        new_path_dir = save_path_time(ASSIT_DIR)
        write_txt_file(ASSIT_DIR, all_results)
        write_txt_file(new_path_dir, all_results)
        
        return self.llm_results
        
    def reflect_success(self):
        user_prompt_succ = read_txt_file('./src/llm/prompts/users/u2.txt')
        prompt2 = f'''
        Please answer the following questions (3). \n
        {user_prompt_succ}
        '''
        assist_results2 = self.llm.answer(prompt2)
        llm_results = self.parse(assist_results2)
        return llm_results
        
    def reflect_improve(self):
        user_prompt_improve = read_txt_file('./src/llm/prompts/users/u3.txt')
        prompt3 = f'''
        Please answer the following questions (4)
        {user_prompt_improve}
        '''
        assist_results3 = self.llm.answer(prompt3)
        llm_results = self.parse(assist_results3)
        return llm_results
        
    # ========================================================================================
    def clear(self):
        self.llm.clear()
        self.llm_results = {}
        
    def save(self):
        all_results = self.llm.get_all_answers()
        print('all ans \n', all_results)
        write_txt_file(ASSIT_DIR, all_results)
        
    def parse(self, assist):
        pattern = r"```(.*?)```"
        result_dict_string = re.search(pattern, assist, re.DOTALL).group(1)
        try:
            llm_results = ast.literal_eval(result_dict_string)
        except:
            print('ast.literal_eval failed')
            self.save()
            llm_results = {}
        return llm_results
        
    def _setup(self):
        # system
        self.system_prompt = read_txt_file(self._hp.system_prompt)
        
        # user
        self.user_prompt = read_txt_file(self._hp.user_prompt)
        
        # llm
        api_key = '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'
        self.llm = self._hp.llm(api_key=api_key)
        self.llm.add_system(self.system_prompt)
        
        # exp_parser
        self.exp_parser = self._hp.exp_parser()
        # assist_parser
        
        self.assist_parser = self._hp.assist_parser()
        # assist_checker


if __name__ == "__main__":
    worker = BaseWorker()
    
    from src.llm.worker.store import exp_results_base, exp_results_test
    # exp_results = exp_results_base
    exp_results = exp_results_test
    
    worker.reflection(exp_results)
    
    # python src/llm/worker/base_worker.py