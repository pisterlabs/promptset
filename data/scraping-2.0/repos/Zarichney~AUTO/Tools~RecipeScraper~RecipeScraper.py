# /Tools/RecipeScraper/RecipeScraper.py

import json
import os
import subprocess
import sys
from instructor import OpenAISchema
from pydantic import Field
from Utilities.Log import Debug, Log, type
from Utilities.Config import WORKING_DIRECTORY, gpt4
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from Agency.Agency import Agency

class RecipeScraper(OpenAISchema):
    """
    Scrapes the internet for a collection of a given recipe.
    Returns JSON array of recipes. 
    The results are optimally sanitized and ranked by relevancy.
    """

    recipe: str = Field(
        ...,
        description="The name of the recipe to search the internet for"
    )

    def run(self, agency: 'Agency'):

        script_directory = "./Tools/RecipeScraper/"
        script_output_directory = "Recipes"
        script_output_path = WORKING_DIRECTORY + script_output_directory
        script = "recipe_scraper.py"
        script_path = script_directory + script
        if not os.path.exists(script_path):
            Log(type.ERROR, f"Unexpected script location: {script_path}")

        # Get the path of the current Python interpreter
        python_path = sys.executable

        Log(type.ACTION, f"Executing recipe scraper for: {self.recipe}")
        Debug(f"Agent called subprocess.run with:\n{[python_path, script_path] + [self.recipe]}")
        
        try:
            # Step 1: run python scraper script to scrawl the internet for related recipes
            execution = subprocess.run(
                [python_path, script_path] + [self.recipe],
                text=True,
                capture_output=True,
                check=True,
                timeout=100
            )
            Debug(f"{script} execution result:\n\n{execution.stdout}")
            
        except subprocess.TimeoutExpired:
            result = "Execution timed out. The script may have been waiting with a prompt."
            Log(type.ERROR, result)
            return result

        except subprocess.CalledProcessError as e:
            result = f"Execution error occurred: {e.stderr}"
            Log(type.ERROR, result)
            return result

        recipes = []

        # Output is expected to be a json file under the script_directory 'Recipes' as an array of recipes
        Debug(f"Reading json result")
        with open(f"{script_output_path}/{self.recipe}.json", "r") as f:
            result = json.load(f)
            
        # Make copy of file for archiving reasons
        with open(f"{script_output_path}/{self.recipe}_scrapped.json", "w") as f:
            json.dump(result, f, indent=2)
            
        recipes = result

        if not recipes:
            result = f"No recipes were able to be scraped..."
            Log(type.RESULT, result)
            return result
        
        # Step 2: refine results
        instruction = f"""
        1. Filter: Eliminate irrelevant recipe data.
        2. Sort: Prioritize recipes by relevance to search query.
        3. Deduplicate: Remove exact duplicates.
        4. Variance: Retain close, non-exact matches selectively.
        5. Sanitize: Correct errors, standardize format
        6. Ensure top result contains valid image url

        Context: Top search results from various sites; refine to match original recipe search.

        Procedure:
        - Exact matches prioritized.
        - Use stringent criteria for additional inclusions if exact matches exist.
        - Sort by relevance: exact matches first.
        - Eliminate identical recipes; retain similar for variety.
        - Balance between variety and redundancy: more variety early, less needed later.
        - Example: 15 recipes, 3 exact, 2 closest non-exact kept, ranked.

        Task: Rewrite for consistency, error correction; return sanitized JSON array.
        """.strip()
        
        completion = agency.client.chat.completions.create(
            model=gpt4,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": instruction},
                {"role": "user", "content": f"Recipe Name: {self.recipe}"},
                # maybe ingest file via new assistant thread (side mission) instead?
                {"role": "user", "content": f"Recipes: {json.dumps(recipes, indent=2)}"}
            ]
        )
        
        Debug(f"Sanitization result:\n{completion.choices[0].message.content}")
        
        recipes = json.loads(completion.choices[0].message.content)
        
        # Write to file
        with open(f"{script_output_path}/{self.recipe}.json", "w") as f:
            json.dump(recipes, f, indent=2)

        if not recipes:
            result = f"No valid recipes found..."
            Log(type.RESULT, result)
            return result
        
        Log(type.RESULT, f"Scrapped {len(recipes)} recipes")

        return f"{len(recipes)} recipes dumped to file '{script_output_directory}/{self.recipe}.json'"
        
