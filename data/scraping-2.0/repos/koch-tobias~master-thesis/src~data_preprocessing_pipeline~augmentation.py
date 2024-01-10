import pandas as pd
import numpy as np
import openai

import math
import random

from loguru import logger
import os
import sys
import yaml
from yaml.loader import SafeLoader
sys.path.append(os.getcwd())

from src.data_preprocessing_pipeline.feature_engineering import Feature_Engineering


with open('src/config.yaml') as file:
    config = yaml.load(file, Loader=SafeLoader)

class DataAugmention:

    # PyTest exist
    @staticmethod
    def random_order(designation: str) -> str:
        '''
        Takes a sentence as input and changes the word order randomly. 
        Arg: 
            Designation of the car part
        Return: 
            New designation
        '''
        # Split the designation into individual words
        words = designation.split()
        new_designation = designation

        # Shuffle the words in the designation
        while new_designation == designation:
            random.shuffle(words)
            
            # Join the shuffled words back into a string
            new_designation = ' '.join(words)

        return new_designation

    # PyTest exist
    @staticmethod
    def swap_chars(designation: str) -> str:
        '''
        Takes a text as input and randomly swap two chars which are next to each other
        Arg: 
            Designation of the car part
        Return: 
            New designation
        '''
        # Convert description to a list of chars
        chars = list(designation)

        # Choose the chars which should be swaped 
        index1 = random.randrange(len(chars) - 1)
        index2 = index1 + random.choice([-1,1])

        # Change their position
        chars[index1], chars[index2] = chars[index2], chars[index1]
        
        return ''.join(chars)

    # PyTest exist
    @staticmethod
    def add_char(designation: str) -> str:
        '''
        Takes a text as input and randomly adds a char to a random position
        Arg: 
            Designation of the car part
        Return: 
            New designation
        '''
        # German alphabet
        alphabet = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
        
        # Randomly choose a char of the alphabet
        char = random.choice(alphabet)

        # Randomly select the position where to add the char
        index = random.randrange(len(designation)+1)

        # Add the selected char
        new_designation = designation[:index] + char + designation[index:]

        return new_designation 

    # PyTest exist
    @staticmethod
    def delete_char(designation: str) -> str:
        '''
        Takes a text as input and randomly deletes a char
        Arg: 
            Designation of the car part
        Return: 
            New designation
        '''
        index = random.randrange(len(designation))
        new_designation = designation[:index] + designation[index+1:]
        
        return new_designation  

    # PyTest exist
    @staticmethod
    def remove_prefix(response: str) -> str:
        ''' 
        The output of GPT has always a prefix like "Answer: " or "Modified Designation: ". This prefix will be deleted that we keep only the generated text.
        Arg: 
            Response of GPT
        Return: 
            New response
        '''
        index = response.find(":")

        if index != -1:
            # Keep only the text after the prefix ending with ":"
            new_response = response[index+1:].lstrip()
        else:
            new_response = response
        
        return new_response

    @staticmethod
    def init_openai():
        ''' 
        Contains the init informations to connect with GPT
        Arg: None
        Return: None
        '''
        openai.api_type = "azure"
        openai.api_base = "https://convaip-sbx-openai.openai.azure.com/"
        openai.api_version = config["gpt_settings"]["api_version"]
        openai.api_key = config["gpt_settings"]["api_key"]

    # PyTest exist
    @staticmethod
    def create_prompt(designation: str) -> str:
        ''' 
        Adds the designation to the costumized prompt    
        Arg: 
            Designation of the car part
        Return: 
            Prompt
        '''
        prompt = f'''
                    Task:
                    Write a modified car component designation based on the given input. Your output must be slightly different to the input while retaining the same meaning.

                    Context:
                    The original designation is written in German and may contain abbreviations and technical terms. 
                    You are not allowed to add or remove words. 
                    Abbreviations that often appear in the data set are 'HI' for 'HINTEN', 'VO' for 'VORN', 'HKL' for 'HECKKLAPPE','GPR' for GEPAECKRAUM', 
                    'VKL' for 'VERKLEIDUNG', 'ISP' for 'INNENSPIEGEL', 'TV' for 'TUER VORNE', 'TH' for 'TUER HINTEN', 'HBL' for 'HOCHGESETZTE BREMSLEUCHTE', 
                    'STF' for 'STOSSFAENGER', 'KST' for 'KOPFSTUETZE', 'ND' for 'NORMALDACH', 'DAHAUBE' for DACHANTENNENHAUBE. 

                    Instructions:
                    1. Carefully examine the input and understand its meaning.
                    2. Modify the designation to create a new version which is very close to the original.
                    3. Make sure that the meaning of the designation remains the same.
                    4. Make sure that the modified designation is not equal to the input data

                    Examples: 
                    - Input Data: ABDECKLEISTE EINSTIEG HI
                    - Modified Designation: ABDECKLEISTE EINSTIEG HINTEN

                    - Original: KANTENSCHUTZ TUER VORN
                    - Modified Designation: KANTENSCHUTZ TV

                    - Original: MD KST VERSTELLBAR AUSSEN MAT
                    - Modified Designation: MD KST AUßENMATTE VERSTELLBAR

                    - Original: SEITENVERKLEIDUNG
                    - Modified Designation: SEITENVERKL

                    - Orignal: ABDECKUNG FENSTERRAHMEN TUER VORNE
                    - Modified Designation: ABDECKUNG FENSTERRAHMEN TUERE VORN

                    """
                    {designation}
                    """
                '''
        return prompt

    @staticmethod
    def gpt35_designation(designation:str) -> str:
        ''' 
        Calls the GPT API and generates the response  
        Arg: 
            Designation of the car part
        Return: 
            New response
        '''
        prompt= DataAugmention.create_prompt(designation=designation)

        # Create a new designation using GPT until it is different to the original designation
        new_response = designation
        while new_response == designation:
            response = openai.ChatCompletion.create(
                engine="chat-gpt-0301",
                messages=[{"role":"system","content":"You are an AI assistant that helps to create a synthetic car component designation."},{"role":"user","content": prompt}],
                temperature = config["gpt_settings"]["temperature"],
                max_tokens = 10,
                top_p = 1,
                n = 1
            )
            response = [choice.message.content for choice in response.choices]
            new_response = DataAugmention.remove_prefix(response=response[0])
        return new_response

    # PyTest exist
    @staticmethod
    def augmented_boundingbox(df_original: pd.DataFrame, df_temp: pd.DataFrame) -> pd.DataFrame:
        ''' 
        Generates a synthetic bounding box based on the car parts in the trainset
        Arg: 
            df_orignal -> origninal dataset, df_temp -> dataframe with one existing car part as starting point
        Return: 
            Dataframe with the new synthetic bounding box  
        '''
        # The valid space is the area +/- 10% where all bounding boxes of car parts with the same class are inside
        corners, valid_length, valid_width, valid_height = Feature_Engineering.find_valid_space(df=df_original)

        list_corners = []
        list_corners.append(corners)

        # Generate random values for length, width, and height until the volume is within the desired range +/- 10% of the original bounding boxes of the same class
        min_volume = df_original['volume'].min()
        max_volume = df_original['volume'].max()
        if min_volume == max_volume:
            min_volume = min_volume*0.9
            max_volume = max_volume*1.1
        
        min_length = df_original['length'].min() * 0.90
        min_width = df_original['width'].min() * 0.90
        min_height = df_original['height'].min() * 0.90
        max_length = df_original['length'].max() * 1.1
        max_width = df_original['width'].max() * 1.1
        max_height = df_original['height'].max() * 1.1
        
        volume = 0
        x_min = 0
        x_max = 0
        y_min = 0
        y_max = 0
        z_min = 0
        z_max = 0
        length = 0
        width = 0
        height = 0
        
        while(volume < min_volume or volume > max_volume):
            length = np.random.uniform(min_length, max_length)
            width = np.random.uniform(min_width, max_width)
            height = np.random.uniform(min_height, max_height)
            volume = length * width * height
        
        # Choose a random center point that the new box fits in the valid area 
        center_point = Feature_Engineering.random_centerpoint_in_valid_space(corners=corners, length=length, width=width, height=height)
        # Calculate the new min, max values in x-, y,- and z-direction
        x_min, x_max, y_min, y_max, z_min, z_max = Feature_Engineering.calculate_transformed_corners(center_point=center_point, length=length, width=width, height=height, theta_x=df_temp.loc[0, "theta_x"], theta_y=df_temp.loc[0, "theta_y"], theta_z=df_temp.loc[0, "theta_z"])

        # Store the new generated bounding box information for the augmented car part
        df_temp.loc[0, "X-Min_transf"] = x_min
        df_temp.loc[0, "X-Max_transf"] = x_max
        df_temp.loc[0, "Y-Min_transf"] = y_min
        df_temp.loc[0, "Y-Max_transf"] = y_max
        df_temp.loc[0, "Z-Min_transf"] = z_min
        df_temp.loc[0, "Z-Max_transf"] = z_max
        df_temp.loc[0, "center_x"] = center_point[0]
        df_temp.loc[0, "center_y"] = center_point[1]
        df_temp.loc[0, "center_z"] = center_point[2]
        df_temp.loc[0, "length"] = length
        df_temp.loc[0, "width"] = width
        df_temp.loc[0, "height"] = height
        df_temp.loc[0, "volume"] = volume
        df_temp.loc[0, "density"] = volume/df_temp.loc[0, "Wert"]

        return df_temp

    # PyTest exist
    @staticmethod
    def data_augmentation(df: pd.DataFrame) -> pd.DataFrame:
        ''' 
        Generates synthetic data to extend the data set. 
        Only the data points that are relevant for a measurement are expanded, since this class is very underrepresented. 
        A component name is expanded if it contains more than 2 words and at least one of the three data augmentation techniques is activated.
        Arg: 
            dataframe with the orignal data
        Return: 
            dataframe with added synthetic data
        '''
        DataAugmention.init_openai()

        logger.info("Start adding artificial designations...")

        # Select the relevant car parts
        df_relevant_parts = df[df["Relevant fuer Messung"] == "Ja"]    

        # Get a list of all uniform names in the dataset and sort them alphabetically
        unique_names = df_relevant_parts["Einheitsname"].unique().tolist()
        unique_names.sort()

        # Iterate over all names 
        for name in unique_names:
            # Create a temporary dataframe with the car parts having the current uniform name 
            df_new = df_relevant_parts[(df_relevant_parts["Einheitsname"] == name)].reset_index(drop=True)
            df_temp = df_new.iloc[[0]]

            # Check if there are enough car parts of this class to split the data in a balanced way into training, validation and test set
            count_designations = df_new.shape[0]
            target_nr_of_unique_carparts = math.ceil(2 / config["dataset_params"]["train_val_split"])

            # If no enough car parts in the dataset, generate synthetic car parts for that class
            if count_designations < target_nr_of_unique_carparts:
                logger.info(f"Adding {target_nr_of_unique_carparts-count_designations} synthetic generated car parts of {name}..")
            while count_designations < target_nr_of_unique_carparts:
                if count_designations < target_nr_of_unique_carparts:
                    df_temp.loc[0,"Benennung (bereinigt)"] = DataAugmention.random_order(designation=df_new.loc[0,"Benennung (bereinigt)"])  
                    count_designations = count_designations + 1
                    if df_temp.loc[0,"volume"] > 0:
                        df_temp = DataAugmention.augmented_boundingbox(df_original=df_new, df_temp=df_temp) 
                    df = pd.concat([df, df_temp], ignore_index=True).reset_index(drop=True) 

                if count_designations < target_nr_of_unique_carparts:
                    df_temp.loc[0,"Benennung (bereinigt)"] = DataAugmention.swap_chars(designation=df_new.loc[0,"Benennung (bereinigt)"])
                    count_designations = count_designations + 1
                    if df_temp.loc[0,"volume"] > 0:
                        df_temp = DataAugmention.augmented_boundingbox(df_original=df_new, df_temp=df_temp) 
                    df = pd.concat([df, df_temp], ignore_index=True).reset_index(drop=True) 

                if count_designations < target_nr_of_unique_carparts:
                    df_temp.loc[0,"Benennung (bereinigt)"] = DataAugmention.delete_char(designation=df_new.loc[0,"Benennung (bereinigt)"])
                    count_designations = count_designations + 1
                    if df_temp.loc[0,"volume"] > 0:
                        df_temp = DataAugmention.augmented_boundingbox(df_original=df_new, df_temp=df_temp) 
                    df = pd.concat([df, df_temp], ignore_index=True).reset_index(drop=True) 

                if count_designations < target_nr_of_unique_carparts:
                    df_temp.loc[0,"Benennung (bereinigt)"] = DataAugmention.gpt35_designation(designation=df_new.loc[0,"Benennung (bereinigt)"])
                    if len(df_temp["Benennung (bereinigt)"][0]) < 40 and df_temp.loc[0,"volume"] > 0:
                        df_temp = DataAugmention.augmented_boundingbox(df_original=df_new, df_temp=df_temp)
                        df = pd.concat([df, df_temp], ignore_index=True).reset_index(drop=True)
                        count_designations = count_designations + 1
                
        logger.success("Successfully added artificial designations to the dataset!")

        return df
