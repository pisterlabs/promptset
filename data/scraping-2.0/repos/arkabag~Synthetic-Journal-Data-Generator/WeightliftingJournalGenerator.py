from typing import Any, Dict, List, Optional
from importlib import metadata

from langchain.chains.base import Chain
from langchain.chains.llm import LLMChain
from langchain.prompts import PromptTemplate
from langchain.schema.language_model import BaseLanguageModel

from langchain_experimental.synthetic_data.prompts import SENTENCE_PROMPT

from langchain.pydantic_v1 import BaseModel

from langchain.chains.llm import LLMChain
from langchain.prompts.few_shot import FewShotPromptTemplate
from langchain.schema.language_model import BaseLanguageModel
from pydantic import BaseModel, Field
from langchain.chains.base import Chain
from langchain.prompts.few_shot import FewShotPromptTemplate
from langchain.pydantic_v1 import BaseModel, root_validator
from typing import Any, Dict, List, Optional
from importlib import metadata

class WorkoutSet(BaseModel):
    weight: Optional[int] = Field(None, description="Weight used in the exercise")
    reps: Optional[int] = Field(None, description="Number of repetitions")
    time: Optional[int] = Field(None, description="Duration in seconds")
    distance: Optional[float] = Field(None, description="Distance in miles")

class Exercise(BaseModel):
    label: str
    type: str
    unit: str
    sets: List[WorkoutSet]

class WorkoutJournalEntry(BaseModel):
    current_weight: int
    exercises: List[Exercise]
    journal_text: str

examples = [
    {
        "current_weight": 185,
        "exercises": [
            {
                "label": "Bench Press",
                "type": "workout_weight_reps",
                "unit": "lbs",
                "sets": [
                    {"weight": 135, "reps": 10},
                    {"weight": 135, "reps": 9},
                    {"weight": 135, "reps": 8}
                ]
            },
            {
                "label": "Weighted Planks",
                "type": "workout_weight_time",
                "unit": "lbs",
                "sets": [
                    {"weight": 25, "time": 60},
                    {"weight": 25, "time": 60},
                    {"weight": 25, "time": 30}
                ]
            },
            {
                "label": "Fast Runs",
                "type": "workout_distance_time",
                "unit": "miles",
                "sets": [
                    {"distance": 0.5, "time": 180},
                    {"distance": 0.5, "time": 200}
                ]
            }
        ],
        "journal_text": "Just finished a high-intensity workout at the gym. It was tough, but invigorating. The weights felt heavier than usual, but I pushed through. Sweating and panting, I found a surge of energy within me I didn't know I had. The feeling of accomplishment afterward is always worth the effort. I'll sleep well tonight, knowing I gave it my all."
    },
    {
        "current_weight": 253,
        "exercises": [
            {
                "label": "Deadlifts",
                "type": "workout_weight_reps",
                "unit": "lbs",
                "sets": [
                    {"weight": 225, "reps": 8},
                    {"weight": 225, "reps": 7},
                    {"weight": 225, "reps": 6}
                ]
            },
            {
                "label": "Squats",
                "type": "workout_weight_reps",
                "unit": "lbs",
                "sets": [
                    {"weight": 185, "reps": 10},
                    {"weight": 185, "reps": 10},
                    {"weight": 185, "reps": 9}
                ]
            },
            {
                "label": "Treadmill Running",
                "type": "workout_distance_time",
                "unit": "miles",
                "sets": [
                    {"distance": 1, "time": 10},
                    {"distance": 1, "time": 11}
                ]
            }
        ],
        "journal_text": "Today's session felt challenging yet rewarding. Managed to increase the weight on my deadlifts and felt stronger. The squats were intense, but I kept my form solid throughout. It's days like this that remind me why I started this journey. I'm looking forward to seeing how far I can push my limits."
    },
    {
        "current_weight": 113,
        "exercises": [
            {
                "label": "Pull-ups",
                "type": "workout_reps",
                "sets": [
                    {"reps": 12},
                    {"reps": 10},
                    {"reps": 8}
                ]
            },
            {
                "label": "Leg Press",
                "type": "workout_weight_reps",
                "unit": "lbs",
                "sets": [
                    {"weight": 300, "reps": 10},
                    {"weight": 300, "reps": 10},
                    {"weight": 300, "reps": 9}
                ]
            },
            {
                "label": "Elliptical Training",
                "type": "workout_distance_time",
                "unit": "miles",
                "sets": [
                    {"distance": 0.75, "time": 15}
                ]
            }
        ],
        "journal_text": "Felt great to hit the gym today. The pull-ups were a bit challenging, but I managed to complete them. Leg press was steady, and I enjoyed wrapping up with some cardio on the elliptical. Feeling stronger and more confident with each passing day."
    },
    {
        "current_weight": 374,
        "exercises": [
            {
                "label": "Overhead Press",
                "type": "workout_weight_reps",
                "unit": "lbs",
                "sets": [
                    {"weight": 95, "reps": 5},
                    {"weight": 95, "reps": 5},
                    {"weight": 95, "reps": 4}
                ]
            },
            {
                "label": "Lat Pulldowns",
                "type": "workout_weight_reps",
                "unit": "lbs",
                "sets": [
                    {"weight": 120, "reps": 10},
                    {"weight": 120, "reps": 9},
                    {"weight": 120, "reps": 8}
                ]
            },
            {
                "label": "Cycling",
                "type": "workout_distance_time",
                "unit": "miles",
                "sets": [
                    {"distance": 2, "time": 30}
                ]
            }
        ],
        "journal_text": "Felt a bit off today at the gym. Struggled with the overhead press and didn't manage to complete my last set. It's frustrating to not see the progress I want. Hoping it's just a bad day and not the start of a slump. I’ll try to shake it off and come back stronger next time."
    },
    {
        "current_weight": 147,
        "exercises": [
            {
                "label": "Bicep Curls",
                "type": "workout_weight_reps",
                "unit": "lbs",
                "sets": [
                    {"weight": 40, "reps": 8},
                    {"weight": 40, "reps": 7},
                    {"weight": 35, "reps": 8}
                ]
            },
            {
                "label": "Tricep Extensions",
                "type": "workout_weight_reps",
                "unit": "lbs",
                "sets": [
                    {"weight": 50, "reps": 8},
                    {"weight": 50, "reps": 7},
                    {"weight": 45, "reps": 8}
                ]
            },
            {
                "label": "Stair Climber",
                "type": "workout_distance_time",
                "unit": "steps",
                "sets": [
                    {"distance": 100, "time": 5}
                ]
            }
        ],
        "journal_text": "Today's workout was a bit demotivating. Had to reduce the weight for my biceps and triceps. It seems I've hit a plateau and not improving much. Need to rethink my workout plan. I need to maybe consult with a trainer for some advice. Just hoping I can find a way to break through this barrier soon."
    },
    {
        "current_weight": 174,
        "exercises": [
            {
                "label": "Rowing Machine",
                "type": "workout_distance_time",
                "unit": "miles",
                "sets": [
                    {"distance": 1, "time": 15}
                ]
            },
            {
                "label": "Kettlebell Swings",
                "type": "workout_weight_reps",
                "unit": "lbs",
                "sets": [
                    {"weight": 35, "reps": 12},
                    {"weight": 35, "reps": 12},
                    {"weight": 35, "reps": 12}
                ]
            },
            {
                "label": "Stretching",
                "type": "workout_time",
                "unit": "minutes",
                "duration": 10
            }
        ],
        "journal_text": "Had an okay session, nothing special. The rowing machine was fine, but there was some dude grunting loudly at the squat rack next to me which was quite distracting. Just not the best atmosphere today. Guess I need to find a way to stay focused amidst distractions."
    },
    {
        "current_weight": 198,
        "exercises": [
            {
                "label": "Squats",
                "type": "workout_weight_reps",
                "unit": "lbs",
                "sets": [
                    {"weight": 200, "reps": 5},
                    {"weight": 200, "reps": 5},
                    {"weight": 200, "reps": 5}
                ]
            },
            {
                "label": "Lunges",
                "type": "workout_weight_reps",
                "unit": "lbs",
                "sets": [
                    {"weight": 40, "reps": 10},
                    {"weight": 40, "reps": 10},
                    {"weight": 40, "reps": 10}
                ]
            },
            {
                "label": "Jump Rope",
                "type": "workout_time",
                "unit": "minutes",
                "duration": 10
            }
        ],
        "journal_text": "Not the best day at the gym. I felt a sharp pain in my knee during squats and had to stop. Might need to rest for a few days and see a physiotherapist. Kind of worried about this setback. It's frustrating but I know health comes first. I'll take it easy and hope to recover soon."
    }
]

from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate, FewShotPromptTemplate

import os
# Set your API key as an environment variable
os.environ['OPENAI_API_KEY'] = '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'

# Initialize the language model
llm = ChatOpenAI(model_name="gpt-4-1106-preview", temperature=0.1)

from langchain.prompts import PromptTemplate, FewShotPromptTemplate

# Define a custom template suitable for workout journal entries
workout_journal_template = """Given the following workout details, create a detailed and engaging journal entry. The entry should reflect on the workout experience, include any feelings or notable events, and incorporate all the provided details.
Details:
{example}
Journal Entry:
"""

# Create a PromptTemplate using this custom template
WORKOUT_JOURNAL_PROMPT = PromptTemplate(
    template=workout_journal_template, input_variables=["example"]
)

# Define the FewShotPromptTemplate
prompt_template = FewShotPromptTemplate(
    prefix="Generate a synthetic weightlifting journal entry based on the details below:",
    examples=examples,
    suffix="End of journal entry.",
    input_variables=["example"],
    example_prompt=WORKOUT_JOURNAL_PROMPT,
)

# Modified to print components of FewShotPromptTemplate

from langchain_experimental.tabular_synthetic_data.openai import create_openai_data_generator

# Print the examples to inspect their structure
print("Examples:")
for example in examples:
    print(example)

# Print the components of the FewShotPromptTemplate
print("\nFewShotPromptTemplate Components:")
print("Prefix:", prompt_template.prefix)
print("Suffix:", prompt_template.suffix)
print("Example Prompts:")
for ex in prompt_template.examples:
    print(ex)

# Create the synthetic data generator
synthetic_data_generator = create_openai_data_generator(
    output_schema=WorkoutJournalEntry,  # Your defined Pydantic model
    llm=llm,                            # The ChatOpenAI instance you initialized
    prompt=prompt_template              # The FewShotPromptTemplate you created
)

# Print to indicate that the generator is ready
print("\nSynthetic data generator is ready to be used.")

# Define the number of synthetic entries you want to generate
number_of_entries_to_generate = 5

# Debugging: Print the input being passed to the generator
print("\nGenerating synthetic data with the following input:")
print(f"Subject: 'weightlifting_journal', Runs: {number_of_entries_to_generate}")

# Generate synthetic journal entries
synthetic_entries = synthetic_data_generator.generate(
    subject="weightlifting_journal",
    extra="",
    runs=number_of_entries_to_generate
)

# Check the generated entries
print("\nGenerated Entries:")
for entry in synthetic_entries:
    print(entry)
