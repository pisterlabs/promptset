from syc_act_eng.data.eval_data.anthropic_eval_mcq_dataset import AnthropicEvalsNLPData
from syc_act_eng.data.eval_data.pablo_evals.dataset import FeedbackSycophancyDataset

def get_eval_dataset(dataset_name, n_samples=100):
    
    if dataset_name == "anthropic_nlp":
        return AnthropicEvalsNLPData(n_samples=n_samples)
    
    elif dataset_name == "feedback-math":
        return FeedbackSycophancyDataset("feedback-math")
    
    else:
        raise ValueError(f"Dataset {dataset_name} not found.")