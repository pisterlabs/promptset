import asyncio
import json
import pickle
import time
from typing import Dict, List, Optional
import numpy as np
import openai
from pydantic import BaseModel
from tqdm import tqdm
from sklearn.ensemble import RandomForestClassifier
from packages.medagogic_sim.logger.logger import get_logger, logging
from packages.medagogic_sim.gpt.medagogic_gpt import MODEL_GPT4, gpt_streamed_lines, UserMessage, SystemMessage, GPTMessage, gpt, MODEL_GPT35

logger = get_logger(level=logging.INFO)

import os

folder = os.path.dirname(os.path.realpath(__file__))

class ExampleData(BaseModel):
    example: str
    embedding: List[float]

class ExampleDatabase(BaseModel):
    examples: Dict[str, List[ExampleData]]


def generate_embeddings() -> None:
    with open(f"{folder}/action_examples.json", "r") as f:
        actions: Dict[str, List[str]] = json.load(f)

    example_db = ExampleDatabase(examples={})

    for action_name, raw_examples in tqdm(actions.items()):
        example_db.examples[action_name] = []
        for example in raw_examples:
            embedding = openai.Embedding.create(
                input=example,
                model="text-embedding-ada-002"
            )['data'][0]['embedding']
            example_db.examples[action_name].append(ExampleData(
                example=example,
                embedding=embedding
            ))

    with open(f"{folder}/action_examples_with_embeddings.json", "w") as f:
        json.dump(example_db.model_dump(), f, indent=4)

# generate_embeddings()


def load_embeddings() -> ExampleDatabase:
    with open(f"{folder}/action_examples_with_embeddings.json", "r") as f:
        example_db = ExampleDatabase(**json.load(f))
    
    return example_db


from packages.medagogic_sim.action_db.actions_for_brains import ActionDatabase, TaskCall

def ExampleDBFromActionDB(action_db: ActionDatabase) -> ExampleDatabase:
    example_db = ExampleDatabase(examples={})
    for action in tqdm(action_db.actions, desc="Creating Action Example Database"):
        example_db.examples[action.name] = []
        for example in action.examples:
            example_str = ""
            if isinstance(example, str):
                example_str = example
            else:
                example_str = example.input
            embedding = openai.Embedding.create(
                input=example_str,
                model="text-embedding-ada-002"
            )['data'][0]['embedding']
            example_db.examples[action.name].append(ExampleData(
                example=example_str,
                embedding=embedding
            ))
        
    return example_db


class PredictionElement(BaseModel):
    label: str
    probability: float

class PredictionResults(BaseModel):
    input_text: str
    input_embedding: List[float]
    sorted_results: List[PredictionElement]

    @property
    def results_str(self) -> str:
        return "\n".join([f"{r.label} ({r.probability:.2f})" for r in self.sorted_results])

    @property
    def actions_list(self) -> str:
        return "\n".join([f"{r.label}" for r in self.sorted_results])



class ActionClassifier:
    def __init__(self, db: ExampleDatabase, classifier: RandomForestClassifier = None):
        self.db = db
        if classifier is None:
            self.classifier = self.train_classifier()
        else:
            self.classifier = classifier


    def train_classifier(self) -> RandomForestClassifier:
        # Extract embeddings and labels
        embeddings = []
        labels = []
        for label, example_data_list in self.db.examples.items():
            for example_data in example_data_list:
                embeddings.append(example_data.embedding)
                labels.append(label)

        # Convert to numpy arrays
        X = np.array(embeddings)
        y = np.array(labels)

        # Train a random forest classifier on the entire data
        clf = RandomForestClassifier(n_estimators=100)
        clf.fit(X, y)

        return clf


    def predict(self, input_text: str, top_n=-1) -> PredictionResults:
        input_embedding = openai.Embedding.create(
            input=input_text,
            model="text-embedding-ada-002"
        )['data'][0]['embedding']

        probabilities = self.classifier.predict_proba([input_embedding])
        class_labels = self.classifier.classes_

        mapped_probabilities = {}
        for label, prob in zip(class_labels, probabilities[0]):
            mapped_probabilities[label] = prob

        sorted_probabilities = sorted(mapped_probabilities.items(), key=lambda x: x[1], reverse=True)

        results = [PredictionElement(label=label, probability=prob) for label, prob in sorted_probabilities[:top_n]]

        return PredictionResults(
            input_text=input_text,
            input_embedding=input_embedding,
            sorted_results=results
        )
    
    @staticmethod
    def from_action_db(action_db: ActionDatabase):
        example_db = ExampleDBFromActionDB(action_db)
        return ActionClassifier(example_db)

    def save(self):
        with open(f"{folder}/action_classifier.json", "w") as f:
            json.dump(self.db.model_dump(), f, indent=4)
        with open(f"{folder}/action_classifier.pkl", "wb") as f:
            pickle.dump(self.classifier, f)

    @staticmethod
    def load():
        with open(f"{folder}/action_classifier.json", "r") as f:
            db = ExampleDatabase(**json.load(f))
        with open(f"{folder}/action_classifier.pkl", "rb") as f:
            classifier = pickle.load(f)
        return ActionClassifier(db=db, classifier=classifier)


    async def get_function_calls(self, input_text: str) -> Optional[List[str]]:
        start_time = time.time()

        prediction = self.predict(input_text, top_n=5)

        logger.debug(input_text)
        logger.debug(prediction.results_str)

        system_prompt = f"""
Map the user's input to an action from the following list, or select "None" if no action is appropriate.

{prediction.actions_list}

Your response must be either a selection of actions from the list (one per line, same order as user input), or "None". Include no other text.
        """.strip()

        user_prompt = f"""
{input_text}
        """.strip()

        gpt_start_time = time.time()

        response = await gpt(messages=[SystemMessage(system_prompt), UserMessage(user_prompt)],
            model=MODEL_GPT35,
            max_tokens=100,
            cache_skip=False
            )
        response = response.strip()

        end_time = time.time()

        logger.debug(response)

        logger.debug(f"Total time taken: {end_time - start_time:.2f} seconds")
        logger.debug(f"GPT Time taken: {end_time - gpt_start_time:.2f} seconds")

        if response is None or response == "" or response.lower() == "none":
            return None
        else:
            return [l.strip() for l in response.split("\n") if l.strip() != ""]


if __name__ == "__main__":
    test_inputs = '''
    "Initiate CPR, 15 compressions to 2 breaths. Continue till told."
    "Administer 0.01mg/kg epinephrine, IV push."
    "Get the crash cart ready."
    "Oxygen mask, 5 liters per minute."
    "Prepare for intubation."
    "Start the IO line, right tibia."
    "Draw blood for CBC, BMP, and ABG."
    "Two-person bag-mask-ventilate."
    "Set up the defibrillator, pediatric pads."
    "Deliver one shock at 2 J/kg."
    "Check for carotid pulse."
    "Administer 20 mL/kg isotonic fluids, rapid bolus."
    "Switch to one-person bag-mask-ventilation."
    "Set up for lumbar puncture."
    "Call for immediate surgical consult."
    "Prepare the child for transport to ICU."
    "Administer 0.1mg/kg Midazolam for sedation."
    "Attach limb leads for ECG."
    "Prepare to administer second epinephrine dose."
    "Time to reassess vitals, everyone pause."
    "Assess the airway for obstructions and alertness. Report back."
    "Check the child's breathing rate and depth. Let me know immediately."
    "Evaluate circulation by checking pulse and skin color. Update me."
    "Assess for disability. Check pupil response and level of consciousness."
    "Expose the chest and limbs for a full assessment. Cover afterward."
    "Quickly review ABCDE and report any red flags."
    "Administer oxygen via mask at a flow rate of 5 L/min."
    "Start oxygen via nasal cannula, set the flow to 3 L/min."
    "Use a non-rebreather mask for oxygen delivery at 8 L/min."
    "Switch to bag valve mask for oxygen. Set flow at 10 L/min."
    "Start blow-by oxygen administration. Keep flow rate at 2 L/min."
    "Intubate with a 4.5mm tube of cuffed endotracheal. Confirm placement."
    "Get a blood glucose reading and report the levels."
    "Take a temperature reading, rectal."
    "Auscultate the heart and note any irregularities."
    "Listen to lung sounds. Inform me of wheezing or crackles."
    "Auscultate the abdomen and report on bowel sounds."
    "Check capillary refill time on nail beds. Update me."
    "Perform a chin lift to secure the airway."
    "Initiate a jaw thrust for airway management."
    "Execute a head tilt to align the airway."
    "Administer Amoxicillin at 20 mg/kg via oral suspension."
    "Titrate oxygen to reach a target saturation of 95%."
    "Monitor for changes due to sepsis and report."
    "Check pulse at the wrist and neck. Relay the rate."
    "Prepare a bolus of Normal Saline at 200 mL."
    "Wait for blood test results before proceeding."
    "Administer the prepared bolus."
    "Speak to the parent and explain the situation and steps being taken."
    "Obtain IV access, make sure it's secure."
    "Insert an IO needle at left tibia using a 16 gauge."
    "Connect to EKG and monitor heart rate."
    "Hook up the BP monitor and get a reading."
    "Connect pulse oximeter and report O2 saturation."
    "Connect the ventilator, set to SIMV and FiO2 at 50%."
    "Connect a continuous glucometer and monitor sugar levels."
    '''.strip().split("\n")

    test_inputs = [x.strip() for x in test_inputs if x.strip() != ""]
    test_inputs = [x.replace('"', '') for x in test_inputs]

    async def main():
        try:
            classifier = ActionClassifier.load()
        except Exception as e:
            print(f"Could not load classifier ({e}), creating new one.")
            # db = load_embeddings()
            # classifier = ActionClassifier(db)
            action_db = ActionDatabase()
            classifier = ActionClassifier.from_action_db(action_db)
            classifier.save()


        # good_input = "Intubate with a 4.5mm tube of cuffed endotracheal. Confirm placement."
        # bad_input = "Amputate the left leg"
        # hard_input = "Get IV access and give 200mg epinephrine stat"

        logger.info(await classifier.get_function_calls("Assess the airway"))
        

    asyncio.run(main())