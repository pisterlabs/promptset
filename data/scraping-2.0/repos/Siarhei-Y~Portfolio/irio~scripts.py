import pandas as pd
import Levenshtein
import openai
import json



class RegionCreator:
    """
    Assign an economic region for a country in a Dataset
    
    use RegionCreator(df, country_column).region()
    """
    def __init__(self, df, country_column, region_col_name='Region'):
        self.df=df
        self.country_column = country_column
        self.region_col_name='region_col_name'
        
        self.regions = pd.read_csv("https://docs.google.com/spreadsheets/d/e/2PACX-1vRmy7Nl3uNIrffyP0MZ2Myzet0x8xrHgPhWWPlD-Bk2K01dAu1Ua1iaGWpbsN6qrA/pub?output=csv")
        self.regions_ = ['EMEA', 'LATAM', 'North America', 'APAC', "CHINA"]

    def region(self):
        self.df[self.country_column] = self.df[self.country_column].str.strip()
        for r in self.regions_:
            self.df.loc[self.df[self.country_column].isin(self.regions[self.regions.Region == r].Country), self.region_col_name] = r
            self.df.loc[self.df[self.country_column].isin(self.regions[self.regions.Region == r].Code), self.region_col_name] = r
        return self.df
    

class FuzzyComparator:
    def __init__(self, df1, df2, col1, col2, score=0.8,tracking=False):
        """
        Class constructor that initializes the FuzzyComparator object with two dataframes,
        the columns to compare, and a matching score.

        Parameters:
        df1 (pandas DataFrame): first dataframe to compare
        df2 (pandas DataFrame): second dataframe to compare
        col1 (str): column name in first dataframe to compare
        col2 (str): column name in second dataframe to compare
        score (float): matching score threshold (default=0.8)
        tracking (bool): set to True to print progress during comparison (default=False)
        """
        self.df1 = df1
        self.df2 = df2
        self.col1 = col1
        self.col2 = col2
        self.score = score
        self.tracking = tracking
    def compare(self):
        """
        Method that compares the two dataframes based on the selected columns using fuzzy matching.

        Returns:
        pandas DataFrame: dataframe with the matched company names and their scores
        """
        n = 0
        matches = []
        for index1, row1 in self.df1.iterrows():
            for index2, row2 in self.df2.iterrows():
                company1 = str(row1[self.col1])
                company2 = str(row2[self.col2])
                dist = Levenshtein.distance(company1.lower(), company2.lower())
                match_score = 1 - (dist / max(len(company1), len(company2)))
                n+=1
                if match_score >= self.score:
                    matches.append({'company1': company1, 'company2': company2, 'match_score': match_score})
                if self.tracking:
                    print(f'{n / (len(self.df1) * len(self.df2)) * 100:.2f}%')
        result_df = pd.DataFrame(matches)
        return result_df
        
class GPT:
    def __init__(self, promt):
        """
        Initializes a new instance of the GPT class with a given prompt.

        Parameters:
        prompt (str): the text prompt to send to the OpenAI API
        """
        self.promt = promt

    def run(self):
    # Set up your OpenAI API credentials
        openai.api_key = "Your API Key"

        # Define the prompt to send to the API
        prompt = self.promt
    
        # Define the parameters for the API request
        params = {
            "model": "text-davinci-002",
            "prompt": prompt,
            "temperature": 0.5,
            "max_tokens": 1024,
            "n": 1,
            "stop": ""
        }
    
        # Send the request to the OpenAI API
        response = openai.Completion.create(**params)
        # Parse the response to extract the generated text
        generated_text = response.choices[0].text.strip()
        return generated_text
            
        
    
