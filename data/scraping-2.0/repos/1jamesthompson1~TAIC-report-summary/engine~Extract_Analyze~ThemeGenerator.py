import os
import yaml

from ..OpenAICaller import openAICaller
from . import OutputFolderReader
from .ReportExtracting import ReportExtractor
from . import Themes, ReferenceChecking

class ThemeGenerator:
    def __init__(self, output_folder, report_dir_template, report_theme_template, modes, discard_old):
        self.output_folder = output_folder
        self.open_ai_caller = openAICaller
        self.report_dir_template = report_dir_template
        self.report_theme_template = report_theme_template
        self.all_themes = ""
        self.output_folder_reader = OutputFolderReader.OutputFolderReader()
        self.modes = modes
        self.discard_old = discard_old

    def _get_theme_file_path(self, report_id):
        return os.path.join(self.output_folder,
                            self.report_dir_template.replace(r'{{report_id}}', report_id),
                            self.report_theme_template.replace(r'{{report_id}}', report_id))

    def generate_themes(self):
        print("Generating themes from reports with config:")
        print(f"  Output folder: {self.output_folder}")
        print(f"  Report directory template: {self.report_dir_template}")
        print(f"  Report theme template: {self.report_theme_template}")


        self.output_folder_reader.process_reports(self._get_theme, self.modes)

        print(" Themes generated for each report")

        print(" Creating global themes")
        self.output_folder_reader.read_all_themes(self._read_themes, self.modes)
        print("  All themes read")

        with open(os.path.join(self.output_folder, "all_themes.txt"), "w") as f:
            f.write(self.all_themes)
            
        print("  Summarizing themes...")
        summarized_themes = self.open_ai_caller.query(
            system="""
You are going to help me summarize the given source text.

The source text will be provided inbetween triple quotes. Below that will be the questions and some notes.
"""
            ,
            user=f"""
'''
{self.all_themes}
'''
            
Question:
These are some safety issues and themes for each report.

I would like to know the global safety themes.
For each safety theme you need to provide a clear explanation of what this safety theme really means.
Each safety theme will need to be given with transport modes it is applicable. These modes are a for aviation, r for rail and m for marine. Safety themes can go across multiple modes of transport are prefered.

There should be no more than 15 safety themes.

Your output needs to be in yaml format. Just output the yaml structure with no extra text (This means no ```yaml and ```) . It will look something like this:
  - title: |-
      title of the theme goes here
    description: |
      Multi line description of the theme goes here.
    modes:
      - modes that should be included. One per row


=Here are some definitions=

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
            temp = 0
        )

        print("  Global theme created")

        themes_data = yaml.safe_load(summarized_themes)

        print("  Now grouping themes")

        while True:

            theme_groups = self.open_ai_caller.query(
                system="""
    You are going to help me group some items.

    The items will be given to you in a yaml format with triple qoutes.
    Each item will have a name and description

    You response should be in pure yaml. It will have a title, description and list of items in this group for each group.

    It is important that the list of themes uses the theme titles verbatim.

    The yaml should not be enclosed and folllow this exact format.
    - title: |-
        tile goes here
    description: |
        description of the group goes here
    themes:
        - theme1
        - theme2

    Each item can only be in one group.
    """,
                user=f"""
    '''
    {summarized_themes}
    '''

    question:

    I have some safety themes that have been identifed by reading alot of accident investigation reports.

    Please put these into groups of related themes. Can you please have about 4-6 groups

    Here are some defintion of what the various terms might mean:
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
                temp = 0
            )

            if theme_groups[:7] == "```yaml":
                theme_groups = theme_groups[7:-3]
                

            groups_data = yaml.safe_load(theme_groups)

            # Validate that the themes and groups are valid

            all_themes = [theme['title'] for theme in themes_data]
            groups_themes = [group['themes'] for group in groups_data]

            # Check that all themes are in a group
            for theme in all_themes:
                if not any(theme in group for group in groups_themes):
                    print(f"  Theme {theme} not in any group retrying grouping")
                    continue
            
            break

        # Sort the themes in the themes_data so that they are in the assigned groups order
        flattened_groups = [theme for group_themes in groups_themes for theme in group_themes]

        themes_data = sorted(themes_data, key=lambda theme: flattened_groups.index(theme['title']))

        # Create a new dictionary with 'themes' and 'groups' branches
        combined_data = {'themes': themes_data, 'groups': groups_data}

        
        Themes.ThemeWriter().write_themes(combined_data)

        print(" Themes summaried and written to file")        


    def _get_theme(self, report_id, report_text):

        print(f" Generating themes for report {report_id}")

        # Check to see if it alreaady exists
        if os.path.exists(self._get_theme_file_path(report_id)) and not self.discard_old:
            print(f"  Themes for {report_id} already exists")
            return

        important_text = ReportExtractor(report_text, report_id).extract_important_text()[0]

        if important_text is None:
            return
        
        system_message = """
You will be provided with a document delimited by triple quotes and a question. Your task is to answer the question using only the provided document and to cite the passage(s) of the document used to answer the question. There may be multiple citations needed. If the document does not contain the information needed to answer this question then simply write: "Insufficient information." If an answer to the question is provided, it must include quotes with citation.

You must follow these formats exactly.
For direct quotes there can only ever be one section mentioned:
"quote in here" (section.paragraph.subparagraph)
For indirect quotes there may be one section, multiple or a range: 
sentence in here (section.paragraph.subparagraph)
sentence in here (section.paragraph.subparagraph, section.paragraph.subparagraph, etc)
sentence in here (section.paragraph.subparagraph-section.paragraph.subparagraph)


Example quotes would be:
"it was a wednesday afternoon when the boat struck" (5.4)
It was both the lack of fresh paint and the old radar dish that caused this accident (4.5.2, 5.4.4)

Quotes should be weaved into your answer. 
"""
        user_message = f"""
'''
{important_text}
'''

Question:
Please provide me 3 - 6 safety themes that are most related to this accident.
For each theme provide a paragraph explaining what the theme is and reasoning (about 75 words) as to why it is relevant to this accident. Provide evidence for your reasoning with inline quotes. More than 1 quote may be needed and direct quotes are preferable.

Please output your answer in yaml. There should be no opening or closing code block just straight yaml. The yaml format should have a name and explanation field (which uses a literal scalar block) for each safety theme.

----
Here are some definition

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

        report_themes_str = self.open_ai_caller.query(
            system_message,
            user_message,
            large_model=True,
            temp = 0
        )

        if report_themes_str is None:
            return
        
        if report_themes_str[:7] == "```yaml":    
            report_themes_str = report_themes_str[7:-3]

        try :
            report_themes = yaml.safe_load(report_themes_str)
        except yaml.YAMLError as exc:
            print(exc)
            print("  Error parsing yaml for themes")
            return self._get_theme(report_id, report_text)
        
        print(f"  Themes for {report_id} generated now validating references")

        referenceChecker = ReferenceChecking.ReferenceValidator(report_text)

        validated_themes_counter = 0
        updated_themes_counter = 0

        for theme in report_themes:
            result = referenceChecker.validate_references(theme['explanation'])

            if result is None:
                print("  No references found in theme")
                continue
            elif isinstance(result, str):
                print(f"  Invalid format")
                return self._get_theme(report_id, report_text)

            processed_text, num_references, num_updated_references = result
            updated_themes_counter += num_updated_references
            if isinstance(processed_text, str):
                theme['explanation'] = processed_text


            validated_themes_counter += num_references
            
        print(f"    {validated_themes_counter} references validated across {len(report_themes)} themes with {updated_themes_counter} themes updated")

        print(f"  References for {report_id} validated now writing to file")

        with open(self._get_theme_file_path(report_id), "w") as f:
            yaml.dump(report_themes, f, default_flow_style=False, width=float('inf'), sort_keys=False)
        

    def _read_themes(self, report_id, report_themes):
        theme = yaml.safe_load(report_themes)

        # convert theme object with name and explanation to a string
        theme_str = '\n\n'.join(f"{element['name']}\n{element['explanation']}" for element in theme)

        self.all_themes += (f"Themes for {report_id}: \n{theme_str}\n")

