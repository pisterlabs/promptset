from ..Extract_Analyze.OutputFolderReader import OutputFolderReader
from ..Extract_Analyze.Themes import ThemeReader
from .Comparer import Comparer
from ..OpenAICaller import openAICaller
from ..Modes import *

import yaml
import csv

class WeightingComparer(Comparer):
    def __init__(self):
        super().__init__()
        self.get_validation_set('summaries')
        self.compared_summaries = dict()

    def  decode_summaries(self, report_summary):
        csv_reader = csv.reader([report_summary])

        elements = next(csv_reader)[:-2]

        return {"weights": [float(weight) if weight !="<NA>" else None for weight in elements[2::3]],
                "explanation": elements[3::3],
                "pages_read": set(elements[1].strip('[]').split("  "))}

    def add_report_ID(self, report_id, report_summary):
        self.validation_set[report_id] = self.decode_summaries(report_summary)

    def compare_weightings(self):
        print("Comparing weightings...")
    
        OutputFolderReader().read_all_summaries(self.compare_two_summaries)

        print("Finished comparing weightings.")
        
        num_reports = len(self.compared_summaries)
        print('==Validation summary==')
        print(f"  {num_reports} reports compared.")
        # print(f"  {[report for report in self.compared_weightings]}")
        print(f"  Average weighting manhattan distance: {sum([self.compared_summaries[report]['weightings'] for report in self.compared_summaries])/num_reports}")
        print(f"  Average pages read jaccard similarity: {sum([self.compared_summaries[report]['pages_read'] for report in self.compared_summaries])/num_reports}")
        print(f"  Average explanation similarity: {sum([self.compared_summaries[report]['explanation'] for report in self.compared_summaries])/num_reports}")
    

    def compare_two_summaries(self, report_id, report_summary):
        if (report_id in self.validation_set.keys()):
            engine_summary = self.decode_summaries(report_summary)
        else:
            return
        
        # Compare the pages read

        validation_pages_read = self.validation_set[report_id]["pages_read"]
        engine_pages_read = engine_summary["pages_read"]

        pages_read_jaccard_similarity = len(validation_pages_read.intersection(engine_pages_read)) / len(validation_pages_read.union(engine_pages_read))
        
        # Compare the weightings
                
        validation_weightings = self.validation_set[report_id]["weights"]
        validation_explanation = self.validation_set[report_id]["explanation"]
        engine_weightings = engine_summary["weights"]
        engine_explanation = engine_summary["explanation"]

        if len(validation_weightings) != len(engine_weightings):
            print(f"  Validation weightings and engine weightings have different lengths. Skipping {report_id}")
            return

        ## Make sure that None are in the same location
        none_in_both = [i for i, (v, e) in enumerate(zip(validation_weightings, engine_weightings)) if v is None and e is None]
        none_in_one = [i for i, (v, e) in enumerate(zip(validation_weightings, engine_weightings)) if (v is None) != (e is None)]

        if none_in_one:
            print(f"  Validation weightings and engine weightings have a None in a different location {none_in_one}. Skipping {report_id}")
            return

        print(f"  Validation weightings and engine weightings have a None in the same location {none_in_both}. Removing from comparison.")

        validation_weightings = [v for i, v in enumerate(validation_weightings) if i not in none_in_both]
        engine_weightings = [e for i, e in enumerate(engine_weightings) if i not in none_in_both]
        validation_explanation = [v for i, v in enumerate(validation_explanation) if i not in none_in_both]
        engine_explanation = [e for i, e in enumerate(engine_explanation) if i not in none_in_both]
        
        manhattan_weightings_similarity = sum([abs(validation_weightings[i] - engine_weightings[i]) for i in range(len(validation_weightings))])/ThemeReader().get_num_themes()

        # Compare the explanations
        explanation_similarity = list()
        for theme, validation_explanation, engine_explanation in zip(ThemeReader(modes = get_report_mode_from_id(report_id))._themes, validation_explanation, engine_explanation):
            explanation_similarity.append(self.compare_weighting_reasoning(theme, validation_explanation, engine_explanation))

        self.compared_summaries[report_id] = {"pages_read": pages_read_jaccard_similarity, 
                                       "weightings": manhattan_weightings_similarity,
                                       "explanation": sum(explanation_similarity)/len(explanation_similarity)}

    def compare_weighting_reasoning(self, theme, validation_explanation, engine_explanation):
        system_message = """
I need you to help me compare two blocks of text.

Both texts to be compared with be given sourrounded by triple quotes.

Below that will be a question specifc to the comparision of these two texts.

Your response will be a percentage of how simliiar these texts are.
Just return a number from 0-100. With 0 being nothing alike and 100 being exactly the same.
"""
        
        user_message = f"""
'''
{validation_explanation}
'''

'''
{engine_explanation}
'''

Question:

Above are two explanations of how much {theme['title']} is related and contributory to a specifc accident.

I want to know how similar these explanations are. Similarity should be judged on; what references are used, what their given weighting is of the safety theme, reasoning of weighting.

{theme['title']} is defined as such:
{theme['description']}

Here are some general definitions:

Safety factor - Any (non-trivial) events or conditions, which increases safety risk. If they occurred in the future, these would
increase the likelihood of an occurrence, and/or the
severity of any adverse consequences associated with the
occurrence.

Safety issue - A safety factor that:
• can reasonably be regarded as having the
potential to adversely affect the safety of future
operations, and
• is characteristic of an organisation, a system, or an
operational environment at a specific point in time.
Safety Issues are derived from safety factors classified
either as Risk Controls or Organisational Influences.

Safety theme - Indication of recurring circumstances or causes, either across transport modes or over time. A safety theme may
cover a single safety issue, or two or more related safety
issues. 
"""
        while(True):
            responses = openAICaller.query(
                system_message,
                user_message,
                large_model=True,
                temp = 0,
                n = 5
            )

            try :
                average_percent = sum([float(response) for response in responses])/5
                if average_percent < 0 or average_percent > 100:
                    raise ValueError
                return average_percent
            except ValueError:
                print(f"  Invalid repsonse from model: {responses}")
                continue