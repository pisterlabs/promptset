from engine import OpenAICaller

import yaml
import time

class ResultsAnalyzer:
    def __init__(self, results):
        self.results = results

    def run_analysis(self):
        print("Running analysis")
        self.analyze_safety_issues()
        print("Analyzed safety issues")
        self.analyze_safety_themes()
        print("Analyzed safety themes")


    def analyze_safety_themes(self):

        self.theme_weightings = self.results.loc[:, 'CompleteSafetyIssues':'PDF'].iloc[:, 1:-1]

        # Remove all columsn that start with Complete
        self.theme_weightings = self.theme_weightings.filter(regex='^(?!Complete)')
    
    def analyze_safety_issues(self):
        all_safety_issues = self.results['CompleteSafetyIssues'].to_list()
        all_safety_issues = map(
            lambda x: "No safety issues" if not x else "\n".join(f"- {item}" for item in x),
            all_safety_issues
        )
        report_ids = self.results['ReportID'].to_list()

        safety_issues_str = "\n\n".join(
            map(
                lambda tup: f"{tup[0]}:\n" + tup[1],
                zip(report_ids, all_safety_issues),
            )
        )

        response = OpenAICaller.openAICaller.query(
            system="""
I want you to help me read a list of items and help summarize these into a single list.

The list you will be given will be inside triple quotes.

Your output needs to be in yaml format. Just output the yaml structure with no extra text (This means no ```yaml and ```). What your output entails will be described in the question.""",
            user = f"""
'''
{safety_issues_str}
'''

Question:
I have a list of safety issues found in each accident investigation report.

Can you please read all of these and respond with a list of all the unique safety issues identified. Note that each the same safety issue may be written in a slightly differnet way.

For each unique safety issue can you add what reports it is found in.

The format should look like

- description: "abc"
  reports:
    - "2019_201"
    - etc
""",
large_model=True,
temp=1
        )

        try: 
            self.safety_issues = yaml.safe_load(response)
        except yaml.YAMLError as exc:
            print(response)
            print(exc)
            time.sleep(1)
            self.analyze_safety_issues()

        self.safety_issues_summary = OpenAICaller.openAICaller.query(
            """
I want you to help me summarize a list of items I have.

You are to read the given text between the triple quote and repsond to the question at the bottom.
""",
f"""
'''
{response}
'''

Question:
Please read this list of safety issues and provide a summary of the common trends and issues found. THis should be prose and not use any lists.

I would like your answer to be concise and only have about 300 words.

Here are some useful defintions:

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
""",
large_model=True,
temp=0
        )
        
