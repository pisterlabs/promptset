import os
import streamlit as st
import openai

import logging
logging.basicConfig(level=logging.WARNING, format='%(asctime)s (%(levelname)s):  %(module)s.%(funcName)s - %(message)s')

# Set up logger for a specific module to a different level
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

import config as config

    
def openai_analyze_financial_statements_dict(all_summaries_dict, ticker, period, n_periods):
    financial_statement_data =""
    for financial_summary_type in all_summaries_dict.keys():
        financial_statement_data += f"Summary of {period} {financial_summary_type} key metrics for {ticker} across {n_periods} periods:"
        df = all_summaries_dict[financial_summary_type].copy()
        # shorten the message to openai
        df.drop(columns=['link'], inplace=True, errors='ignore')
        df.drop(columns=['finalLink'], inplace=True, errors='ignore')
        financial_statement_data += df.to_json(orient="split")
        
    logger.debug(f"financial_statement_data: {financial_statement_data}, len: {len(financial_statement_data)}")
                        
    if config.check_for_api_key('openai'):
        openai.api_key = config.get_api_key('openai')
        
        analyze = f"You are an investor, analyze the financial statement data for {ticker} to " + \
                    "summarize business results and trends in paragraph form per type of financial statement provided. "
        summary = f"Financial statement data points: {financial_statement_data}.\n"
        additionally = f"Also, write an overall summary on the aggregate results over time sharing positives as well as any concerns with respect to the overall financial health of the company."
                    
        question = analyze + summary + additionally
        logger.info(f"question for OpenAI: {question}, len: {len(question)}")
        
        chat_completion = openai.ChatCompletion.create(model=os.getenv('GPT_MODEL'), messages=[{"role": "user", "content": question}])
        #gpt-4-32k
        #chat_completion = openai.ChatCompletion.create(model="gpt-4-32k", messages=[{"role": "user", "content": question}])
        
        return chat_completion.choices[0].message.content