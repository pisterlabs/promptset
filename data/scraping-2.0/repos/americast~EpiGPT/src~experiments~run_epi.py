import os
import pickle
from data.monash import get_datasets
from data.serialize import SerializerSettings
from models.validation_likelihood_tuning import get_autotuned_predictions_data
from models.utils import grid_iter
from models.llmtime import get_llmtime_predictions_data
import numpy as np
import openai
from statsforecast import StatsForecast
from statsforecast.models import AutoARIMA
import pandas as pd
openai.api_key = "abc"

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, choices=['llama-7b', 'gpt4all', 'arima'], default='llama-7b', required=False, help='model to run')
parser.add_argument('--dataset', type=str, choices=["covid_deaths", "hospital","cdc_flu", "cdc_covid", "symp"], default='cdc_flu', required=False, help='dataset to use')
parser.add_argument('--explain', type=bool, default=False, required=False, help='add explanation to the input')
parser.add_argument('--explain_less', type=bool, default=False, required=False, help='add a lot of explanation to the input')
parser.add_argument('--explain_lockdown', type=bool, default=False, required=False, help='how would the time series vary if a lockdown was declared')

args = parser.parse_args()

# Specify the hyperparameter grid for each model
gpt3_hypers = dict(
    temp=0.7,
    alpha=0.9,
    beta=0,
    basic=False,
    settings=SerializerSettings(base=10, prec=3, signed=True, half_bin_correction=True),
)

llama_hypers = dict(
    temp=1.0,
    alpha=0.99,
    beta=0.3,
    basic=False,
    settings=SerializerSettings(base=10, prec=3, time_sep=',', bit_sep='', plus_sign='', minus_sign='-', signed=True), 
)

model_hypers = {
    'text-davinci-003': {'model': 'text-davinci-003', **gpt3_hypers},
    'llama-7b': {'model': 'llama-7b', **llama_hypers},
    'llama-70b': {'model': 'llama-70b', **llama_hypers},
    'gpt4all': {'model': "orca-mini-3b-gguf2-q4_0.gguf", **llama_hypers},
    'arima': {'model': "statsforecast", **llama_hypers}
}

# Specify the function to get predictions for each model
model_predict_fns = {
    'text-davinci-003': get_llmtime_predictions_data,
    'llama-7b': get_llmtime_predictions_data,
    'llama-70b': get_llmtime_predictions_data,
    'gpt4all': get_llmtime_predictions_data
}

def is_gpt(model):
    return any([x in model for x in ['ada', 'babbage', 'curie', 'davinci', 'text-davinci-003', 'gpt-4']])

# Specify the output directory for saving results
if args.explain:
    output_dir = 'outputs/monash_explain'
elif args.explain_less:
    output_dir = 'outputs/monash_explain_less'
elif args.explain_lockdown:
    output_dir = 'outputs/monash_explain_lockdown'
else:
    output_dir = 'outputs/monash'
os.makedirs(output_dir, exist_ok=True)

models_to_run = [
    # 'text-davinci-003',
    # 'llama-7b',
    # 'llama-70b',
    # 'gpt4all'
    # 'arima',
    args.model
]
datasets_to_run =  [
    # "weather",
    # "solar_weekly",
    # "tourism_monthly",
    # "australian_electricity_demand",
    # "pedestrian_counts",
    # "traffic_hourly",
    # "fred_md",
    # "tourism_yearly",
    # "tourism_quarterly",
    # "us_births",
    # "covid_deaths",
    # "hospital",
    # "nn5_weekly",
    # "traffic_weekly",
    # "saugeenday",
    # "cif_2016",
    # "bitcoin",
    # "sunspot",
    # "nn5_daily",
    # "cdc_flu",
    # "cdc_covid",
    # "symp",
    args.dataset
]

max_history_len = 30
max_pred_len = 10
datasets = get_datasets()
for dsname in datasets_to_run:
    print(f"Starting {dsname}")
    data = datasets[dsname]
    train, test = data
    print("mean: "+str(np.mean(np.array([np.mean(x) for x in train]))))
    print("deviation: "+str(np.std(np.array([np.mean(x) for x in train]))))
    if "symp" in dsname:
        train = [x[-max_history_len:] for x in train][-10:]
        test = [x[:max_pred_len] for x in test][-10:]
    else:
        train = [x[-max_history_len:] for x in train][:10]
        test = [x[:max_pred_len] for x in test][:10]
    if os.path.exists(f'{output_dir}/{models_to_run[0]}_{dsname}.pkl'):
        with open(f'{output_dir}/{models_to_run[0]}_{dsname}.pkl','rb') as f:
            out_dict = pickle.load(f)
    else:
        out_dict = {}
    # import pudb; pu.db
    for model in models_to_run:
        if model in out_dict:
            print(f"Skipping {dsname} {model}")
            continue
        else:
            print(f"Starting {dsname} {model}")
            hypers = list(grid_iter(model_hypers[model]))
        if args.explain:
            hypers[0]["explain"] = dsname
        elif args.explain_less:
            hypers[0]["explain"] = dsname + "_less"
        elif args.explain_lockdown:
            hypers[0]["explain"] = dsname + "_lockdown"
        else:
            hypers[0]["explain"] = False
        parallel = True if is_gpt(model) else False
        num_samples = 5
        
        try:
            if model == 'arima':
                preds = {"median": [], "lower": [], "upper": []}
                for t in train:
                    df = pd.DataFrame({"unique_id": [1]*len(t), "y":t, "ds": list(range(1, len(t) +1))})
                    sf = StatsForecast(
                    models = [AutoARIMA(season_length = len(train[0]))],
                        freq = 'M'
                    )

                    sf.fit(df)
                    out = sf.predict(h=len(test[0]), level=[95])
                    preds['median'].append(list(out['AutoARIMA']))
                    preds['lower'].append(list(out['AutoARIMA-lo-95']))
                    preds['upper'].append(list(out['AutoARIMA-hi-95']))
            else:
                preds = get_autotuned_predictions_data(train, test, hypers, num_samples, model_predict_fns[model], verbose=False, parallel=parallel)
            medians = preds['median']
            preds['train'] = train
            preds['test'] = test
            targets = np.array(test)
            maes = np.mean(np.abs(medians - targets), axis=1) # (num_series)        
            preds['maes'] = maes
            preds['mae'] = np.mean(maes)
            out_dict[model] = preds
            print(out_dict)
        except Exception as e:
            print(f"Failed {dsname} {model}")
            print(e)
            continue
        with open(f'{output_dir}/{model}_{dsname}.pkl','wb') as f:
            pickle.dump(out_dict,f)
    print(f"Finished {dsname}")
    