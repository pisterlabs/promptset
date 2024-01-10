import openai
import os
import pickle
import tqdm

# Authenticate with openai
openai.api_key = os.getenv("GPT_3_API_KEY")


class Experiment:
    """For running experiments with GPT-3 API and saving them"""

    def __init__(self, dataset, gpt_3_engine="davinci"):
        self._ds = dataset

        # See engines at https://beta.openai.com/pricing
        self._gpt_3_engine = gpt_3_engine

        # Results has form
        # {idx: [(prompt_1, response_1)...(prompt_n, response_n)]}
        self._results = {}

    def _process_prompt(self, prompt):
        """Process prompt with self._gpt_3_version version of GPT-3"""
        try:
            # TODO: Are these arguments correct?
            # TODO: Do we ever want max_tokens > 1?
            return openai.Completion.create(engine=self._gpt_3_engine,
                                            prompt=prompt,
                                            max_tokens=1,
                                            logprobs=100)
        # TODO: Catch more specific exception here
        except Exception as exc:
            print(exc)
            return None

    def run(self):
        """Get results from GPT-3 API"""
        for (dv_name, dv_dict) in tqdm.tqdm(self._ds.prompts.items()):
            self._results[dv_name] = {}
            for (row_idx, prompt) in dv_dict.items():
                response = self._process_prompt(prompt)
                raw_target = self._ds.dvs[dv_name][row_idx]
                am = self._ds._get_col_prompt_specs()[dv_name].answer_map
                target = am[raw_target]
                self._results[dv_name][row_idx] = (prompt, response, target)

    def save_results(self, fname):
        """Save results obtained from run method"""
        with open(fname, "wb") as f:
            pickle.dump(self._results, f)


if __name__ == "__main__":
    ###############################################################################################################
    # MMMMMMMM               MMMMMMMMEEEEEEEEEEEEEEEEEEEEEE       GGGGGGGGGGGGG               AAA               
    # M:::::::M             M:::::::ME::::::::::::::::::::E    GGG::::::::::::G              A:::A              
    # M::::::::M           M::::::::ME::::::::::::::::::::E  GG:::::::::::::::G             A:::::A             
    # M:::::::::M         M:::::::::MEE::::::EEEEEEEEE::::E G:::::GGGGGGGG::::G            A:::::::A            
    # M::::::::::M       M::::::::::M  E:::::E       EEEEEEG:::::G       GGGGGG           A:::::::::A           
    # M:::::::::::M     M:::::::::::M  E:::::E            G:::::G                        A:::::A:::::A          
    # M:::::::M::::M   M::::M:::::::M  E::::::EEEEEEEEEE  G:::::G                       A:::::A A:::::A         
    # M::::::M M::::M M::::M M::::::M  E:::::::::::::::E  G:::::G    GGGGGGGGGG        A:::::A   A:::::A        
    # M::::::M  M::::M::::M  M::::::M  E:::::::::::::::E  G:::::G    G::::::::G       A:::::A     A:::::A       
    # M::::::M   M:::::::M   M::::::M  E::::::EEEEEEEEEE  G:::::G    GGGGG::::G      A:::::AAAAAAAAA:::::A      
    # M::::::M    M:::::M    M::::::M  E:::::E            G:::::G        G::::G     A:::::::::::::::::::::A     
    # M::::::M     MMMMM     M::::::M  E:::::E       EEEEEEG:::::G       G::::G    A:::::AAAAAAAAAAAAA:::::A    
    # M::::::M               M::::::MEE::::::EEEEEEEE:::::E G:::::GGGGGGGG::::G   A:::::A             A:::::A   
    # M::::::M               M::::::ME::::::::::::::::::::E  GG:::::::::::::::G  A:::::A               A:::::A  
    # M::::::M               M::::::ME::::::::::::::::::::E    GGG::::::GGG:::G A:::::A                 A:::::A 
    # MMMMMMMM               MMMMMMMMEEEEEEEEEEEEEEEEEEEEEE       GGGGGG   GGGGAAAAAAA                   AAAAAAA
    ###############################################################################################################

    # Import experiment code
    from experiment import Experiment

    # Import all the datasets
    from pew_american_trends_78_dataset import PewAmericanTrendsWave78Dataset
    from pew_american_trends_67_dataset import PewAmericanTrendsWave67Dataset
    from baylor_religion_survey_dataset import BaylorReligionSurveyDataset
    from add_health_dataset import AddHealthDataset
    from anes import AnesDataset
    from cces import CCESDataset
    from gss_dataset import GSSDataset
    from prri_dataset import PRRIDataset

    datasets = {PewAmericanTrendsWave67Dataset: "pew67",
                PewAmericanTrendsWave78Dataset: "pew78",
                BaylorReligionSurveyDataset: "baylor",
                AddHealthDataset: "add_health",
                AnesDataset: "anes",
                CCESDataset: "cces",
                GSSDataset: "gss",
                PRRIDataset: "prri"}

    for (name, ds_cls) in datasets.items():

        # Set up the experiment
        e = Experiment(ds_cls(), gpt_3_engine="ada")

        # Run the experiment
        e.run()

        # Save the results
        e.save_results(f"{ds_cls}_mega.pkl")
