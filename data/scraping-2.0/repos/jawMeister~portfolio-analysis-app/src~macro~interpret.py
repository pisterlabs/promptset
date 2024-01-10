import os
import openai
import json
import pandas as pd
import numpy as np
import concurrent.futures

import config as config

import logging
logging.basicConfig(level=logging.WARNING, format='%(asctime)s (%(levelname)s):  %(module)s.%(funcName)s - %(message)s')

# Set up logger for a specific module to a different level
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class CustomJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.int64):
            return int(obj)
        return super(CustomJSONEncoder, self).default(obj)
    
def ask_openai(question):
    #gpt-4-0613
    #chat_completion = openai.ChatCompletion.create(model=os.getenv("GPT_MODEL"), messages=[{"role": "user", "content": question}])
    chat_completion = openai.ChatCompletion.create(model="gpt-4-0613", messages=[{"role": "user", "content": question}])
    return chat_completion.choices[0].message.content
     
def openai_ask_about_macro_economic_factors(portfolio_summary, cum_monthly_regression_models_df, cum_monthly_multivariate_models_df, monthly_var_models_df):


    if config.check_for_api_key('openai'):
        openai.api_key = config.get_api_key('openai')
        
        tickers = []
        for ticker in portfolio_summary["stock_data"].columns:
            if portfolio_summary["weights"][ticker] > 0:
                tickers.append(ticker)
                
        portfolio_stats = f"Given a portfolio with a Sharpe Ratio of {portfolio_summary['sharpe_ratio']:.2f}, Sortino Ratio of {portfolio_summary['sortino_ratio']:.2f}, " + \
                            f"CVaR of {portfolio_summary['cvar']:.2f} composed of {tickers} weighted as {portfolio_summary['weights']},"

        question = portfolio_stats + \
                    "I am now interested in understanding the impact of macro-economic factors on the portfolio's performance and would like to share the results of " + \
                    "single-factor linear regressions, a multivariate regression, and a VAR model of portfolio returns against various macro-economic factors."
        
        linear_regression_summary = extract_linear_regression_models_to_json(cum_monthly_regression_models_df)
        #regression_question = "In simple terms, could you please provide an investment interpretation for the following single-factor linear regression results with a focus on the 2-5 most important points?\n"
        regression_question = "Summary of single-factor linear regression results:\n"
        regression_question += linear_regression_summary
        
        multivariate_regression_summary = extract_multivariate_model_to_json(cum_monthly_multivariate_models_df)
        #multivariate_regression_question = "In simple terms, could you please provide an investment interpretation for the following multivariate regression results with a focus on the 2-5 key takeaways?\n"
        multivariate_reqgression_question = "Summary of multivariate regression results:\n"
        multivariate_reqgression_question += multivariate_regression_summary

        var_model_summary = extract_var_model_to_json(monthly_var_models_df)
        #var_model_question = "In simple terms, could you please provide an investment interpretation for the following VAR model results with a focus on the 2-5 key takeaways?\n"
        var_model_question = "Summary of VAR model results:\n"
        var_model_question += var_model_summary
        
        finally_question = "\n\nBased on your collective interpretations of the portfolio statistics and the regression model results, " + \
                            "what are your overall investment based observations and recommendations for this portfolio? " + \
                            "How stable or resilient would you consider this portfolio to be?\n" + \
                            "When would appropriate times be to rebalance the portfolio and what 2-3 macro factors should I pay most attention to, why and how should I potentially adjust allocations?\n"

        synthesis_response = ask_openai(question + regression_question + multivariate_reqgression_question + var_model_question + finally_question)
        
        
        #with concurrent.futures.ProcessPoolExecutor() as executor:
        #    questions = [question, regression_question, multivariate_regression_question, var_model_question]
        #    future_responses = {executor.submit(ask_openai, question): question for question in questions}
        portfolio_response = None
        linear_regression_response = None
        multivariate_regression_response = None
        var_model_response = None
        
        #responses = []
        #for future in concurrent.futures.as_completed(future_responses):
        #    responses.append(future.result())
            
        #portfolio_response = responses[0]
        #linear_regression_response = responses[1]
        #multivariate_regression_response = responses[2]
        #var_model_response = responses[3]

        # Once we have all the responses, we can ask for an overall interpretation
        #synthesis_question = "Based on the following interpretations of the portfolio analysis and the regression models, what should the overall takeaways and recommendations be from an investment perspective in this portfolio and please specify any relevant macro factors?\n" + \
        #                    "When would appropriate times be to rebalance the portfolio?\n" + \
        #                    "Portfolio Analysis Interpretation: " + portfolio_response + \
        #                    "\n\nSingle-factor Linear Regression Interpretation: " + linear_regression_response + \
        #                    "\n\nMultivariate Regression Interpretation: " + multivariate_regression_response + \
        #                    "\n\nVAR Model Interpretation: " + var_model_response
        
        #synthesis_response = ask_openai(synthesis_question)

        return synthesis_response, portfolio_response, linear_regression_response, multivariate_regression_response, var_model_response
        
        
    
def extract_var_model_to_json(var_models_df, periods=6):
    logger.debug(f"Extracting VAR model to JSON: var_models_df.shape = {var_models_df.shape}")
    
    json_data = ""
    if not var_models_df.empty:
        row = var_models_df.iloc[0]
        model = row['Model']

        # Extract the high-level summary statistics directly
        high_level_stats = {
            'No. of Equations': model.neqs,
            'Nobs': model.nobs,
            'Log likelihood': model.llf,
            'AIC': model.aic,
            'BIC': model.bic,
            'HQIC': model.hqic,
            'FPE': model.fpe,
            'Det(Omega_mle)': model.detomega,
        }

        # Find the coefficients for the Portfolio up to lag order 6
        coefficients_df = model.params
        tvalues_df = model.tvalues
        pvalues_df = model.pvalues

        portfolio_factors = {}
        for i in range(1, periods+1):
            factor_stats = coefficients_df.loc[f'L{i}.Portfolio'].index
            for factor in factor_stats:
                portfolio_factors[factor] = {
                    "Coefficient": coefficients_df.loc[f'L{i}.Portfolio'][factor],
                    "t-value": tvalues_df.loc[f'L{i}.Portfolio'][factor],
                    "p-value": pvalues_df.loc[f'L{i}.Portfolio'][factor]
                }

        # Prepare the information for API
        model_dict = {
            'Model Type': row['Model Type'],
            'Lag Order': row['Lag Order'],
            'Summary Stats': high_level_stats,
            'Portfolio Factors': portfolio_factors
        }

        # Convert the data to JSON
        json_data = json.dumps(model_dict, cls=CustomJSONEncoder)
    else:
        logger.error(f"var_models_df is empty!")
        
    return json_data

def extract_multivariate_model_to_json(multivariate_models_df):
    row = multivariate_models_df.iloc[0]
    model = row['Model']
    #significant_features = row['Significant Features']
    
    coefficients = model.params 
    p_values = model.pvalues

    summary_data = []
    for factor, coefficient in coefficients.items():
        p_value = p_values[factor]
        summary_data.append({'Factor': factor, 'Coefficient': coefficient, 'P-value': p_value})
    summary_df_multivar = pd.DataFrame(summary_data)
        
    return summary_df_multivar.to_json(orient='records')
    
def extract_linear_regression_models_to_json(regression_models_df):
    # Initialize a list to store summary data
    summary_data = []
    
    # Iterate through each row of the dataframe
    for index, row in regression_models_df.iterrows():
        # Extract the model
        model = row['Model']
        
        # Extract the model summary and other details
        summary_data.append({
            'Model Type': row['Model Type'],
            'Factor': row['Factor'],
            'Coefficient': row['Coefficient'],
            'P-value': row['P-value'],
            'R-squared': row['R-squared'],
            'Correlation': row['Correlation'],
            'Optimal Lag': row['Optimal Lag']
        })
    
    # Convert the summary data to a dataframe
    summary_df = pd.DataFrame(summary_data)
    
    # Convert the dataframe to JSON
    return summary_df.to_json(orient='records')