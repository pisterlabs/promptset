from .OutputFolderReader import OutputFolderReader
from .ReportExtracting import ReportExtractor
from ..OpenAICaller import openAICaller
import pandas as pd

class APICostEstimator:
    def __init__(self) -> None:
        self.output_folder_reader = OutputFolderReader()
        
        self.important_text_tokens = []

    def  get_cost_summary_strings(self):
        print("Calculating API cost of Engine run through...")

        print("  Getting token size of all reports")
        self.output_folder_reader.process_reports(self._process_report)

        api_cost_per_token = (0.003/1000)

        df = pd.DataFrame(self.important_text_tokens)
        df['api_cost'] = df['important_text_tokens']  * api_cost_per_token

        # themes
        maximum_output_tokens = 500
        generate_report_themes = len(df) * maximum_output_tokens * api_cost_per_token + df['api_cost'].sum()
        collect_format_themes =  len(df) * maximum_output_tokens * api_cost_per_token + 500*3 * api_cost_per_token
        themes_total = generate_report_themes + collect_format_themes

        # summarize
        summarize_cost = df['api_cost'].sum()
        cost_per_report = df['api_cost'].mean()

        print("API cost calculated")

        number_of_digits= 6

        summarize_str = f"Summarize:\nTotal cost ${round(summarize_cost, number_of_digits)}\nAverage cost per report ${round(cost_per_report, number_of_digits)}."

        theme_str = f"Themes:\nTotal cost ${round(themes_total, number_of_digits)}\nGenerate themes for each report ${round(generate_report_themes, number_of_digits)}\nSummarize all themes into one ${round(collect_format_themes, number_of_digits)}"

        return {"summarize": summarize_str, "themes": theme_str, "all": f"The total cost of a complete run through is ${round(themes_total+summarize_cost,number_of_digits )} for {len(df)} reports. Below are summaries for each section\n\n" + summarize_str + "\n\n" + theme_str}

    def _process_report(self, report_id, report_text):
        important_text = ReportExtractor(report_text, report_id).extract_important_text()[0]

        if important_text is None:
            return
        
        important_text_tokens = openAICaller.get_tokens('gpt-3.5-turbo', [important_text])[0]

        self.df = self.important_text_tokens.append(
            {'report_id': report_id, 'important_text_tokens': important_text_tokens})