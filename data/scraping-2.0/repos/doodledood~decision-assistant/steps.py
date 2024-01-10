import itertools
import json
import os
from typing import List, Dict, Tuple, Optional, Callable, Generator

import ahpy
import questionary
from chatflock.backing_stores import InMemoryChatDataBackingStore, LangChainMemoryBasedChatDataBackingStore
from chatflock.base import Chat
from chatflock.conductors import RoundRobinChatConductor, LangChainBasedAIChatConductor
from chatflock.parsing_utils import chat_messages_to_pydantic
from chatflock.participants import LangChainBasedAIChatParticipant, UserChatParticipant
from chatflock.renderers import TerminalChatRenderer, NoChatRenderer
from chatflock.structured_string import Section, StructuredString
from chatflock.use_cases.request_response import get_response
from chatflock.web_research import WebSearch
from halo import Halo
from langchain.chat_models import ChatOpenAI
from langchain.llms.openai import OpenAI
from langchain.memory import ConversationSummaryBufferMemory
from langchain.tools import BaseTool
from pydantic import BaseModel, Field

from presentation import generate_decision_report_as_html, save_html_to_file, open_html_file_in_browser
from ranking.ranking import topsis_score, normalize_label_value
from state import DecisionAssistantState


def fix_string_based_on_list(s: str, l: List[str]) -> Optional[str]:
    for item in l:
        if item.lower() in s.lower():
            return item

    return None


class Criterion(BaseModel):
    name: str = Field(description='The name of the criterion. Example: "Affordability".')
    description: str = Field(
        description='A description of the criterion. Includes the sub-criteria and how to assign scale values to data.')
    scale: List[str] = Field(
        description='The scale of the criterion, from worst to best. Labels only. No numerical value, '
                    'no explainations. Example: "Very Expensive".')


class GoalIdentificationResult(BaseModel):
    goal: str = Field(description='The identified decision-making goal.')


class CriteriaIdentificationResult(BaseModel):
    criteria: List[Criterion] = Field(description='The identified criteria for evaluating the decision.')


class AlternativeListingResult(BaseModel):
    alternatives: List[str] = Field(description='The identified alternatives for the decision.')


class CriteriaResearchQueriesResult(BaseModel):
    criteria_research_queries: Dict[str, List[str]] = Field(
        description='The research queries for each criteria. Key is the criterion name, value is a list of research '
                    'queries for that criterion.')


class AlternativeCriteriaResearchFindingsResult(BaseModel):
    updated_research_findings: str = Field(
        description='The updated and aggregated research findings for the alternative and criterion. Formatted as '
                    'rich markdown with all the citations and links in place.')
    label: str = Field(
        description='The label assigned to the alternative and criterion based on the aggregated research findings '
                    'and user discussion. The label is assigned from the scale of the criterion (name of the label).')


class Alternative(BaseModel):
    name: str = Field(description='The name of the alternative.')
    criteria_data: Optional[Dict[str, Tuple[str, int]]] = Field(
        description='The research data collected for each criterion for this alternative. Key is the name of the '
                    'criterion. Value is a tuple of the research data as text and the assigned value based on the '
                    'scale of the criterion.')


def gather_unique_pairwise_comparisons(
        criteria_names: List[str],
        predict_fn: Optional[Callable[[str, List[str], Dict[Tuple[str, str], str]], str]] = None,
        previous_comparisons: Optional[List[Tuple[Tuple[str, str], float]]] = None,
        on_question_asked: Optional[Callable[[Tuple[str, str], float], None]] = None) \
        -> Generator[Tuple[Tuple[str, str], float], None, None]:
    choices = {
        'Absolutely less important': 1 / 9,
        'A lot less important': 1 / 7,
        'Notably less important': 1 / 5,
        'Slightly less important': 1 / 3,
        'Just as important': 1,
        'Slightly more important': 3,
        'Notably more important': 5,
        'A lot more important': 7,
        'Absolutely more important': 9
    }
    value_to_choice = {v: k for k, v in choices.items()}
    ordered_choice_names = [choice[0] for choice in sorted(choices.items(), key=lambda x: x[1])]

    comparisons = dict(previous_comparisons)
    all_combs = list(itertools.combinations(criteria_names, 2))
    for i, (label1, label2) in enumerate(all_combs):
        if (label1, label2) in comparisons:
            continue

        question_text = f'({i + 1}/{len(all_combs)}) How much more important is "{label1}" when compared to "{label2}"?'

        if predict_fn is not None:
            comparisons_with_str_choice = {k: value_to_choice[v] for k, v in comparisons.items()}
            predicted_answer = predict_fn(question_text, ordered_choice_names, comparisons_with_str_choice)
        else:
            predicted_answer = ordered_choice_names[len(ordered_choice_names) // 2]

        answer = questionary.select(
            question_text,
            choices=ordered_choice_names,
            default=predicted_answer,
        ).ask()

        labels = (label1, label2)
        value = choices[answer]

        comparisons[labels] = value

        yield labels, value


def identify_goal(chat_model: ChatOpenAI, state: DecisionAssistantState,
                  tools: Optional[List[BaseTool]] = None, spinner: Optional[Halo] = None):
    if state.data.get('goal') is not None:
        return

    ai = LangChainBasedAIChatParticipant(
        name='Decision-Making Goal Identifier',
        role='Decision-Making Goal Identifier',
        personal_mission='Identify a clear and specific decision-making goal from the user\'s initial vague statement.',
        other_prompt_sections=[
            Section(
                name='Process',
                list=[
                    'Start by greeting the user and asking for their decision-making goal. Example: "Hello, '
                    'what is your decision-making goal?"',
                    'If the goal is not clear, ask for clarification and refine the goal.',
                    'If the goal is clear, confirm it with the user.',
                ]
            ),
            Section(
                name='User Decision Goal',
                list=[
                    'One and only one decision goal can be identified.',
                    'The goal should be clear and specific.',
                    'The goal should be a decision that can be made by the user.',
                    'No need to go beyond the goal. The next step will be to identify alternatives and criteria for '
                    'the decision.'
                ]
            ),
            Section(
                name='Last Message',
                list=[
                    'After the goal has been identified, the last message should include the goal.'
                    'It should end with the word TERMINATE at the end of the message to signal the end of the chat.'
                ]
            )
        ],
        tools=tools,
        chat_model=chat_model,
        spinner=spinner)
    user = UserChatParticipant(name='User')
    participants = [ai, user]

    chat = Chat(
        backing_store=InMemoryChatDataBackingStore(),
        renderer=TerminalChatRenderer(),
        initial_participants=participants
    )

    chat_conductor = RoundRobinChatConductor()
    _ = chat_conductor.initiate_dialog(chat=chat)
    goal = chat_messages_to_pydantic(
        chat_messages=chat.get_messages(),
        chat_model=chat_model,
        output_schema=GoalIdentificationResult,
        spinner=spinner
    )
    goal = goal.goal

    state.data = {**state.data, **dict(goal=goal)}


def identify_alternatives(chat_model: ChatOpenAI, tools: List[BaseTool],
                          state: DecisionAssistantState, spinner: Optional[Halo] = None):
    if state.data.get('alternatives') is not None:
        return

    ai = LangChainBasedAIChatParticipant(
        name='Decision-Making Alternative Consultant',
        role='Decision-Making Alternative Consultant',
        personal_mission='Assist the user in identifying alternatives for the decision-making process.',
        other_prompt_sections=[
            Section(
                name='Interaction Schema',
                list=[
                    'This is the second part of the decision-making process, after the goal has been identified. No '
                    'need for a greeting.',
                    'Start by asking the user for alternatives they had in mind for the decision.',
                    'Assist the user in generating alternatives if they are unsure or struggle to come up with '
                    'options or need help researching more ideas. You can use the web search tool and your own '
                    'knowledge for this.',
                    'List the final list of alternatives and confirm with the user before moving on to the next step.'
                ]
            ),
            Section(
                name='Requirements',
                list=[
                    'At the end of the process there should be at least 2 alternatives and no more than 20.'
                ]
            ),
            Section(
                name='Alternatives',
                list=[
                    'The alternatives should be clear and specific.',
                    'The alternatives should be options that the user can choose from.',
                    'Naming the alternatives should be done in a way that makes it easy to refer to them later on.',
                    'For example, for a goal such as "Decide which school to go to": The alternative "Go to school X" '
                    'is bad, while "School X" is good.'
                ]
            ),
            Section(
                name='The Last Message',
                list=[
                    'The last response should include the list of confirmed alternatives.',
                    'It should end with the word TERMINATE at the end of the message to signal the end of the chat.'
                ]
            )
        ],
        tools=tools,
        chat_model=chat_model,
        spinner=spinner)
    user = UserChatParticipant(name='User')
    participants = [user, ai]

    chat = Chat(
        backing_store=InMemoryChatDataBackingStore(),
        renderer=TerminalChatRenderer(),
        initial_participants=participants
    )

    chat_conductor = RoundRobinChatConductor()
    _ = chat_conductor.initiate_dialog(chat=chat, initial_message=str(StructuredString(
        sections=[
            Section(name='Goal', text=state.data['goal']),
        ]
    )))
    output = chat_messages_to_pydantic(
        chat_messages=chat.get_messages(),
        chat_model=chat_model,
        output_schema=AlternativeListingResult,
        spinner=spinner
    )
    alternatives = output.alternatives

    state.data = {**state.data, **dict(alternatives=alternatives)}


def identify_criteria(chat_model: ChatOpenAI, tools: List[BaseTool],
                      state: DecisionAssistantState, spinner: Optional[Halo] = None):
    if state.data.get('criteria') is not None:
        return

    shared_prompt_sections = [
        Section(
            name='Process Stage',
            text='This is the third part of the decision-making process, after the goal and alternatives have been '
                 'identified. No need for a greeting.'
        ),
    ]

    criteria_brainstormer = LangChainBasedAIChatParticipant(
        name='Criteria Brainstormer',
        role='Criteria Brainstormer',
        personal_mission='Brainstorm and iterate on the best set of criteria for the decision-making process.',
        other_prompt_sections=shared_prompt_sections + [
            Section(
                name='Criteria Identification Methodology',
                list=[
                    'Start by suggesting an initial set of criteria that is as orthogonal, non-overlapping, '
                    'and comprehensive as possible (including the scale, sub-criteria, and description).',
                    'Iterate on the criteria with the critic until you both are satisfied with them.',
                    'Once you both are satisfied, confirm the criteria with the user and ask for feedback.',
                    'The criteria should include the scale of the criterion description of the criterion, including '
                    'the sub-criteria and how to assign scale values to data.'
                ]
            ),
            Section(name='Criteria Description', list=[
                'The description should include the sub-criteria and how to assign scale values to data. That means '
                'that each criterion should include concrete measures (like indexes, specific indicators, statistics '
                'if there are any) to allow for the researcher to accurately compare alternatives later on',
                'The measures should be as objective and specific as possible.',
                'These measures should be reflective of the sub-criteria and the scale of the criterion.'
            ]),
            Section(name='Scale Definition', list=[
                'The scale should be a list of labels only. No numerical values, no explainations. Example: '
                '"Very Expensive".',
                'The scale should be ordered from worst to best. Example: "Very Expensive" should come before '
                '"Expensive".',
                'Make should the values for the scale are roughly evenly spaced out. Example: "Very '
                'Expensive" should be roughly as far from "Expensive" as "Expensive" is from "Fair".'
            ]),
            Section(name='General Formatting', list=[
                'Make sure all the criteria are formatted nicely in markdown format and are easy to read, including '
                'their description, sub-criteria and explainations.'
            ]),
            Section(
                name='Requirements',
                sub_sections=[
                    Section(
                        name='Criteria',
                        list=[
                            'At the end of the process there MUST be at least 1 criterion and no more than 15 criteria.',
                        ]),
                    Section(
                        name='Scales',
                        list=[
                            'Scales MUST be on at least 2-point scale and no more than 7-point scale.'
                        ]
                    )
                ]
            ),
            Section(
                name='The Last Message',
                list=[
                    'The last response should include the list of confirmed criteria and their respective scales, '
                    'numbered from 1 to N, where N is the best outcome for the criteria.'
                ]
            )
        ],
        tools=tools,
        chat_model=chat_model,
        spinner=spinner)
    criteria_critic = LangChainBasedAIChatParticipant(
        name='Criteria Critic',
        role='Criteria Critic',
        personal_mission='Critique the criteria and provide feedback on what to improve.',
        other_prompt_sections=shared_prompt_sections + [
            Section(
                name='Criteria Critiquing',
                list=[
                    'When critiquing the criteria, make sure they are orthogonal, non-overlapping, and comprehensive.',
                    'When critiquing the scales, make sure they are ordered from worst to best, evenly spaced out, '
                    'and have labels that make sense.',
                ],
                sub_sections=[
                    Section(
                        name='Questions to ask yourself',
                        list=[
                            'Are all the criteria named such that the worst option on their respective potential '
                            'scales is the worst outcome for the decision, and vise-versa for the last label/best '
                            'outcome?',
                            'Are there any criteria that are redundant or duplicated?',
                            'Are there any criteria that are missing to create a comprehensive set of criteria?',
                            'Is the criteria set maximally orthogonal and non-overlapping?',
                            'Are there any criteria that are too subjective or vague?',
                            'Same thing for the sub-criteria within the main criteria.',
                            'Is there at least 1 criterion identified?',
                            'Are there no more than 15 criteria identified?',
                            'Are all the descriptions for the criteria clear and easy to understand?',
                            'Do all the descriptions include concrete measures (like indexes, metrics, statistics, '
                            'etc. - if possible) that can be effectively used to'
                            'research and compare alternatives later on?',
                            'Are all the labels on a scale ordered from worst to best?',
                            'Can a scale be simplified such that it is easier to assign a value to a piece of data '
                            'based on it?',
                            'Is a scale too simple such that it is not useful for the decision-making process?',
                            'Are all the scales on a 2-point to 7-point scale?'
                        ]
                    )
                ]
            )
        ],
        tools=tools,
        chat_model=chat_model,
        spinner=spinner)
    user = UserChatParticipant(name='User')
    participants = [user, criteria_brainstormer, criteria_critic]

    try:
        memory = ConversationSummaryBufferMemory(
            llm=chat_model,
            max_token_limit=OpenAI.modelname_to_contextsize(chat_model.model_name)
        )
        backing_store = LangChainMemoryBasedChatDataBackingStore(memory=memory)
    except ValueError:
        backing_store = InMemoryChatDataBackingStore()

    chat = Chat(
        backing_store=backing_store,
        renderer=TerminalChatRenderer(),
        initial_participants=participants
    )

    chat_conductor = LangChainBasedAIChatConductor(
        chat_model=chat_model,
        goal='Identify clear well-defined criteria and their respective scales for the decision.',
        interaction_schema=(
            '1. The Criteria Brainstormer suggests an initial set of criteria (including description and scales) '
            'based on the user input.\n'
            '2. The Criteria Critic critiques the criteria suggested and suggests improvements.\n'
            '3. The Criteria Brainstormer iterates on the criteria until they think they are good enough and ask the '
            'user for feedback.\n'
            '4. If the user is not satisfied with the criteria, go back to step 1, refining the criteria based on the '
            'user feedback.\n'
            '5. If the user is satisfied with the criteria, the criteria identification process is complete. The '
            'Criteria Brainstormer should present the final list of criteria and their respective scales to the '
            'user.\n'
            '6. The chat should end.'),
    )
    _ = chat_conductor.initiate_dialog(chat=chat, initial_message=str(StructuredString(
        sections=[
            Section(name='Goal', text=state.data['goal']),
            Section(name='Alternatives', list=state.data['alternatives']),
        ]
    )))
    output = chat_messages_to_pydantic(
        chat_messages=chat.get_messages(),
        chat_model=chat_model,
        output_schema=CriteriaIdentificationResult,
        spinner=spinner
    )
    criteria = output.model_dump()['criteria']

    state.data = {**state.data, **dict(criteria=criteria)}


def prioritize_criteria(chat_model: ChatOpenAI, tools: List[BaseTool],
                        state: DecisionAssistantState, spinner: Optional[Halo] = None):
    if state.data.get('criteria_weights') is not None:
        return

    criteria_comparisons = state.data.get('criteria_comparisons', {})
    criteria_comparisons = {tuple(json.loads(labels)): value for labels, value in criteria_comparisons.items()}
    criteria_comparisons = list(criteria_comparisons.items())

    criteria_names = [criterion['name'] for criterion in state.data['criteria']]

    def predict_answer(question: str, choices: List[str], previous_answers: Dict[Tuple[str, str], str]):
        ai = LangChainBasedAIChatParticipant(
            name='Decision-Making Pairwise Criteria Comparisons Predictor',
            role='Decision-Making Pairwise Criteria Comparisons Predictor',
            personal_mission='Predict the most likely option the user will choose based on previous pairwise '
                             'comparisons between criteria.',
            other_prompt_sections=[
                Section(
                    name='Steps',
                    list=[
                        'Retrieve the user\'s previous pairwise comparisons between criteria.',
                        'Analyze the list of options.',
                        'Make a prediction about the user\'s most likely choice based on the analyzed data.',
                        'Return the predicted option.'
                    ],
                    list_item_prefix=None
                ),
                Section(
                    name='Note',
                    list=[
                        'Only one option should be predicted.',
                        'The prediction should be the best possible guess based on the user\'s previous answers.',
                        'If you really do not know or it is impossible to guess, return the middle option.'
                    ]
                ),
                Section(
                    name='Output',
                    text='Only the label of the best-guess option'
                ),
                Section(
                    name='Output Format',
                    text='"...\nPREDICTION: CHOICE" Where CHOICE is a verbatim label from the choices given only.'
                )
            ],
            tools=tools,
            chat_model=chat_model,
            spinner=spinner)
        user = UserChatParticipant(name='User')
        participants = [user, ai]

        predicted_answer, _ = get_response(query=str(StructuredString(
            sections=[
                Section(
                    name='Previous Pairwise Comparisons',
                    list=[
                        f'How much more important is "{criterion_1}" when compared to "{criterion_2}"? -> {value}' for
                        (criterion_1, criterion_2), value in previous_answers.items()
                    ]
                ),
                Section(
                    name='Comparison to Predict',
                    text=question
                ),
                Section(
                    name='Choices',
                    list=choices
                )
            ]
        )), answerer=ai, renderer=NoChatRenderer())

        parts = predicted_answer.split('PREDICTION:', 2)
        if len(parts) != 2:
            return choices[len(choices) // 2]

        predicted_answer = parts[1].strip()
        predicted_answer = fix_string_based_on_list(predicted_answer, choices)

        if predicted_answer is None:
            return choices[len(choices) // 2]

        return predicted_answer

    for labels, value in gather_unique_pairwise_comparisons(
            criteria_names,
            predict_fn=predict_answer,
            previous_comparisons=criteria_comparisons):
        criteria_comparisons.append((labels, value))

        state.data = {**state.data, **dict(
            criteria_comparisons={json.dumps(labels): value for labels, value in criteria_comparisons})}
        yield state

    state.data['criteria_weights'] = ahpy.Compare('Criteria', dict(criteria_comparisons)).target_weights


def generate_research_questions(chat_model: ChatOpenAI, tools: List[BaseTool],
                                state: DecisionAssistantState, spinner: Optional[Halo] = None):
    if state.data.get('criteria_research_queries') is not None:
        return

    ai = LangChainBasedAIChatParticipant(
        name='Decision-Making Process Researcher',
        role='Decision-Making Process Researcher',
        personal_mission='Generate a template for automated research queries for each criterion, whose answers can be '
                         'used as context when evaluating alternatives.',
        other_prompt_sections=[
            Section(
                name='Process',
                list=[
                    'This is the fifth part of the decision-making process, after the goal, alternatives and criteria, '
                    'have been identified. No need for a greeting.',
                    'For each criterion, generate relevant, orthogonal, and comprehensive set query templates.',
                ],
                list_item_prefix=None
            ),
            Section(
                name='Query Templates',
                list=[
                    'The query templates should capture the essence of the criterion based on the scale and how to '
                    'assign values.',
                    'The queries should be strategic and aim to minimize the number of questions while maximizing the '
                    'information gathered.',
                    'The list of queries should include counterfactual queries and make use of all knowledge of '
                    'information foraging and information literacy.',
                    'Each query template MUST include "{alternative}" in the template to allow for replacement with '
                    'various alternatives later.',
                    'If a criterion is purely subjective and nothing an be researched on it, it\'s ok to have 0 '
                    'queries about it.'
                ]
            ),
            Section(
                name='The Last Message',
                list=[
                    'The last response should include the list of research query templates for each criterion.',
                    'It should end with the word TERMINATE at the end of the message to signal the end of the chat.'
                ]
            )
        ],
        tools=tools,
        chat_model=chat_model,
        spinner=spinner)
    user = UserChatParticipant(name='User')
    participants = [user, ai]

    chat = Chat(
        backing_store=InMemoryChatDataBackingStore(),
        renderer=TerminalChatRenderer(),
        initial_participants=participants,
        max_total_messages=2
    )

    chat_conductor = RoundRobinChatConductor()
    _ = chat_conductor.initiate_dialog(chat=chat, initial_message=str(StructuredString(
        sections=[
            Section(name='Goal', text=state.data['goal']),
            Section(name='Alternatives', list=state.data['alternatives']),
            Section(name='Criteria',
                    sub_sections=[
                        Section(name=criterion['name'], text=criterion['description'], list=criterion['scale'],
                                list_item_prefix=None) for criterion in
                        state.data['criteria']
                    ]),
        ]
    )))
    output = chat_messages_to_pydantic(
        chat_messages=chat.get_messages(),
        chat_model=chat_model,
        output_schema=CriteriaResearchQueriesResult,
        spinner=spinner
    )
    criteria_names = [criterion['name'] for criterion in state.data['criteria']]
    output.criteria_research_queries = {fix_string_based_on_list(name, criteria_names): queries for name, queries in
                                        output.criteria_research_queries.items()}

    criteria_research_queries = output.model_dump()['criteria_research_queries']

    state.data = {**state.data, **dict(criteria_research_queries=criteria_research_queries)}


def perform_research(chat_model: ChatOpenAI, web_search: WebSearch, n_search_results: int,
                     tools: List[BaseTool], state: DecisionAssistantState,
                     spinner: Optional[Halo] = None,
                     fully_autonomous: bool = True):
    research_data = state.data.get('research_data')
    if research_data is None:
        research_data = {}

    for alternative in state.data['alternatives']:
        alternative_research_data = research_data.get(alternative)

        if alternative_research_data is None:
            alternative_research_data = {}

        for i, criterion in enumerate(state.data['criteria']):
            criterion_name = criterion['name']
            criterion_research_questions = state.data['criteria_research_queries'][criterion_name]
            alternative_criterion_research_data = alternative_research_data.get(criterion_name)

            if alternative_criterion_research_data is None:
                alternative_criterion_research_data = {'raw': {}, 'aggregated': {}}

            # Already researched and aggregated, skip
            if alternative_criterion_research_data['aggregated'] != {}:
                continue

            # Research data online for each query
            for query in criterion_research_questions:
                query = query.format(alternative=alternative)

                # Already researched query, skip
                if query in alternative_criterion_research_data['raw']:
                    continue

                found_answer, answer = web_search.get_answer(query=query, n_results=n_search_results,
                                                             spinner=spinner)

                if not found_answer:
                    alternative_criterion_research_data['raw'][query] = 'No answer found online.'

                    if spinner:
                        spinner.warn(f'No answer found for query "{query}".')
                else:
                    alternative_criterion_research_data['raw'][query] = answer

                alternative_research_data[criterion_name] = alternative_criterion_research_data
                research_data[alternative] = alternative_research_data
                state.data['research_data'] = research_data

                yield state

    # Do this separately, so all the automated research runs entirely before the user is asked to discuss the findings
    for alternative in state.data['alternatives']:
        alternative_research_data = research_data.get(alternative)

        if alternative_research_data is None:
            alternative_research_data = {}

        for i, criterion in enumerate(state.data['criteria']):
            criterion_name = criterion['name']

            alternative_criterion_research_data = alternative_research_data[criterion_name]

            # Already researched and aggregated, skip
            if alternative_criterion_research_data['aggregated'] != {}:
                continue

            ai = LangChainBasedAIChatParticipant(
                name='Decision-Making Process Researcher',
                role='Decision-Making Process Researcher',
                personal_mission='Refine research findings through user interaction and assign an accurate label '
                                 'based on data, user input, and criteria.',
                other_prompt_sections=[
                    Section(
                        name='Process',
                        list=[
                            'This is the sixth part of the decision-making process, after the goal, alternatives, '
                            'criteria, and research queries have been identified. No need for a '
                            'greeting.',
                            'Present the researched data to the user and assign a preliminary label & ask for feedback',
                            'Revise the research findings based on user input, until the user is satisfied with the '
                            'findings and label.',
                        ],
                    ),
                    Section(
                        name='Research Presentation',
                        list=[
                            'Maintain original findings if no new user input.',
                            'Mention the sources of the research findings as inline links.'
                        ]
                    ),
                    Section(
                        name='Label Assignment',
                        list=[
                            'Assign one label per criterion per alternative based on scale and value assignment '
                            'rules. A label should be a string only, e.g., "Very Expensive".',
                            'If unclear, make an educated guess based on data and user input.'
                        ]
                    ),
                    Section(
                        name='The First Message',
                        list=[
                            'Your first message should look something like this: "Here is what I found about {'
                            'alternative} for {criterion}:\n\n{research_findings}\n\nBecause {'
                            'reason_for_label_assignment}, I think the label for {alternative} for {criterion} should '
                            'be {label}. What do you think? Do you have anything else to add, clarify or change that '
                            'might affect this label?"'
                        ]
                    ),
                    Section(
                        name='The Last Message',
                        list=[
                            'The last response should include the refined research findings for a criterion\'s '
                            'alternative in rich markdown format with all the citations and links inline.',
                            'Does not include conversational fluff. Think about it like a research report.',
                            'Does not include the starting sentence: "Here is what I found about...". It should dive '
                            'straight into the refined findings.',
                            'It should end with the word TERMINATE at the end of the message to signal the end of the '
                            'chat.'
                        ]
                    )
                ],
                tools=tools,
                chat_model=chat_model,
                spinner=spinner)
            user = UserChatParticipant(name='User')
            participants = [user, ai]

            chat = Chat(
                backing_store=InMemoryChatDataBackingStore(),
                renderer=TerminalChatRenderer(),
                initial_participants=participants,
                max_total_messages=2 if fully_autonomous else None
            )

            chat_conductor = RoundRobinChatConductor()
            _ = chat_conductor.initiate_dialog(chat=chat, initial_message=str(StructuredString(
                sections=[
                    Section(name='Goal', text=state.data['goal']),
                    Section(name='Alternatives', list=state.data['alternatives']),
                    Section(name='Criterion',
                            sub_sections=[
                                Section(name=criterion_name, text=criterion['description'], list=criterion['scale'],
                                        list_item_prefix=None)
                            ]),
                    Section(name='Research Findings',
                            sub_sections=[
                                Section(name=query, text=answer) for query, answer in
                                alternative_criterion_research_data[
                                    'raw'].items()
                            ])
                ]
            )))
            criterion_full_research_data = chat_messages_to_pydantic(
                chat_messages=chat.get_messages(),
                chat_model=chat_model,
                output_schema=AlternativeCriteriaResearchFindingsResult,
                spinner=spinner
            )

            research_data[alternative][criterion_name]['aggregated'] = {
                'findings': criterion_full_research_data.updated_research_findings,
                'label': criterion_full_research_data.label
            }
            state.data['research_data'] = research_data

            yield state

    state.data = {**state.data, **dict(research_data=research_data)}


def analyze_data(state: DecisionAssistantState):
    if state.data.get('scored_alternatives') is not None:
        return

    items = [state.data['research_data'][alternative] for alternative in state.data['alternatives']]

    criteria_weights = state.data['criteria_weights']
    criteria_names = [criterion['name'] for criterion in state.data['criteria']]

    scores = topsis_score(items=items,
                          weights=criteria_weights,
                          value_mapper=lambda item, criterion: \
                              normalize_label_value(label=item[criterion]['aggregated']['label'],
                                                    label_list=state.data['criteria'][
                                                        criteria_names.index(criterion)]['scale'],
                                                    lower_bound=0.0,
                                                    upper_bound=1.0),
                          best_and_worst_solutions=(
                              {criterion['name']: {'aggregated': {'label': criterion['scale'][-1]}} for
                               criterion in state.data['criteria']},
                              {criterion['name']: {'aggregated': {'label': criterion['scale'][0]}} for
                               criterion in state.data['criteria']}
                          ))
    scored_alternatives = {alternative: score for alternative, score in zip(state.data['alternatives'], scores)}

    state.data = {**state.data, **dict(scored_alternatives=scored_alternatives)}


def compile_data_for_presentation(state: DecisionAssistantState, report_file: str):
    if os.path.exists(report_file):
        return

    enriched_alternatives = []
    for alternative in state.data['alternatives']:
        alternative_research_data = state.data['research_data'][alternative]
        alternative_score = state.data['scored_alternatives'][alternative]

        enriched_alternatives.append({
            'name': alternative,
            'score': alternative_score,
            'criteria_data': alternative_research_data
        })

    html = generate_decision_report_as_html(
        criteria=state.data['criteria'],
        criteria_weights=state.data['criteria_weights'],
        alternatives=enriched_alternatives,
        goal=state.data['goal'])
    save_html_to_file(html, report_file)


def present_report(state: DecisionAssistantState, report_file: str):
    open_html_file_in_browser(report_file)
