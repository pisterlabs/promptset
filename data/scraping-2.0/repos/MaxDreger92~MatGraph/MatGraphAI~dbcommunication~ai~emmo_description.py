import json
from json import JSONDecodeError

import openai
from django.conf import settings
from dotenv import load_dotenv
from neomodel import UniqueIdProperty
from owlready2 import *
from owlready2 import get_ontology, Thing

from dbcommunication.ai.setupMessages import ONTOLOGY_ASSISTANT_MESSAGES
from matgraph.models.abstractclasses import UIDDjangoNode
from matgraph.models.ontology import EMMOMatter, EMMOQuantity, EMMOProcess
from graphutils.models import AlternativeLabel


def convert_alternative_labels(onto):
    onto_path = os.path.join("/home/mdreger/Documents/MatGraphAI/Ontology/", onto)
    onto_path_alt = os.path.join("/home/mdreger/Documents/MatGraphAI/Ontology/alt_list", onto)
    ontology = get_ontology(onto_path_alt).load()
    print("ALT", onto_path_alt)

    # Define the new alternative_label property
    # Define the new alternative_label property
    with ontology:
        class alternative_label(AnnotationProperty):
            domain = [Thing]
            range = [str]

        # Iterate over all classes in the ontology
        for cls in ontology.classes():
            # If the class has the 'alternative_labels' property
            if cls.alternative_labels:
                # Retrieve the alternative_labels value, parse it, and remove the property
                alt_labels = list(
                    cls.alternative_labels[0].replace("[", "").replace("]", "").replace("'", "").split(","))
                # cls.alternative_labels = []
                for l in alt_labels:
                    label = l.strip()
                    label = re.sub(r'\W+', '', label)
                    cls.alternative_label.append(label)  # Make sure to use the newly defined property

        ontology.save(onto_path, format="rdfxml")


class OntologyManager:
    def __init__(self, api_key, ontology_folder="/home/mdreger/Documents/MatGraphAI/Ontology/"):
        self.api_key = api_key
        self.ontology_folder = ontology_folder
        self.file_to_model = {
            "matter.owl": EMMOMatter,
            "quantities.owl": EMMOQuantity,
            "manufacturing.owl": EMMOProcess}

    def chat_with_gpt3_5(self, setup_message=[], prompt=''):
        openai.api_key = self.api_key

        conversation_history = setup_message
        conversation_history.append({"role": "user", "content": prompt})

        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=conversation_history,
            max_tokens=250,
            n=1,
            stop=None,
            temperature=0,
        )
        return response["choices"][0]["message"]["content"]

    def update_ontology(self, ontology_file):
        ontology_path1 = os.path.join(self.ontology_folder, ontology_file)
        ontology_path = os.path.join(self.ontology_folder, ontology_file)

        onto = get_ontology(ontology_path).load()
        with onto:
            class alternative_label(AnnotationProperty):
                domain = [Thing]
                range = [str]
            class onto_name(AnnotationProperty):
                domain = [Thing]
                range = [str]
        print(ontology_path)
        for cls in onto.classes():
            if not cls.onto_name:
                print(f"Need to update class: {cls.name}")
                try:
                    output = self.chat_with_gpt3_5(ONTOLOGY_ASSISTANT_MESSAGES, cls.name)
                    output = json.loads(output)
                    cls.comment = output["description"]
                    cls.alternative_labels = str(output["alternative_labels"])
                    cls.onto_name = cls.name
                except JSONDecodeError:
                    print(f"Invalid JSON response for class: {cls.name}")

        onto.save(ontology_path1, format="rdfxml")

    def import_to_neo4j(self, ontology_file):

        ontology_path = os.path.join(self.ontology_folder, ontology_file)
        onto = get_ontology(ontology_path).load()

        with onto:
            pass

        for cls in onto.classes():

            class_name = str(cls.name)
            class_uri = str(cls.iri)
            class_comment = str(cls.comment[0]) if cls.comment else None
            try:
                cls_instance = self.file_to_model[ontology_file].nodes.get(uri=class_uri)
                cls_instance.name = class_name
                cls_instance.description = class_comment
                cls_instance.save()
            except:
                cls_instance = self.file_to_model[ontology_file](uri=class_uri, name=class_name,
                                                                 description=class_comment)
                cls_instance.save()

            if cls.alternative_label:
                [str(label) for label in
                 cls.alternative_label] if cls.alternative_label else None

                for label in cls.alternative_label:
                    alt_label = str(label)
                    try:
                        alternative_label_node = AlternativeLabel.nodes.get(label=alt_label)
                        if not cls_instance.alternative_label.is_connected(alternative_label_node):
                            secondary_label_node = AlternativeLabel(label=alt_label)
                            secondary_label_node.save()
                            cls_instance.alternative_label.connect(secondary_label_node)
                    except:
                        alternative_label_node = AlternativeLabel(label=alt_label)
                        alternative_label_node.save()
                        cls_instance.alternative_label.connect(alternative_label_node)

            # Add subclass relationships
            for subclass in cls.subclasses():
                subclass_name = str(subclass.name)
                subclass_uri = str(subclass.iri)
                subclass_comment = str(subclass.comment[0]) if subclass.comment else None
                # print(f"Creating subclass_instance with kwargs: {subclass_name=}, {subclass_comment=}, {subclass_uri=}")
                try:
                    subclass_instance = self.file_to_model[ontology_file].nodes.get(uri=subclass_uri)
                    subclass_instance.name = subclass_name
                    subclass_instance.description = subclass_comment
                    subclass_instance.save()
                    subclass_instance.emmo_subclass.connect(cls_instance)

                except:
                    subclass_instance = self.file_to_model[ontology_file](uri=subclass_uri, name=subclass_name,
                                                                          description=subclass_comment)
                    subclass_instance.save()
                    subclass_instance.emmo_subclass.connect(cls_instance)

    def update_all_ontologies(self):
        ontologies = [f for f in os.listdir(self.ontology_folder) if f.endswith(".owl")]
        print(self.ontology_folder, ontologies)
        for ontology_file in ontologies:
            self.update_ontology(ontology_file)

    def import_all_ontologies(self):
        ontologies = [f for f in os.listdir(self.ontology_folder) if f.endswith(".owl")]
        print(self.ontology_folder, ontologies)
        for ontology_file in ontologies:
            self.import_to_neo4j(ontology_file)


def main():
    # Get the project root directory
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # Change the current working directory to the project root directory
    os.chdir(project_root)

    load_dotenv()
    from neomodel import config

    config.DATABASE_URL = os.getenv('NEOMODEL_NEO4J_BOLT_URL')
    print(config.DATABASE_URL)

    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "mat2devplatform.settings")
    api_key = settings.OPENAI_API_KEY
    ontology_folder = "/home/mdreger/Documents/MatGraphAI/Ontology/"

    ontology_manager = OntologyManager(api_key, ontology_folder)
    ontology_manager.import_all_ontologies()





    # from emmopy import get_emmo








if __name__ == "__main__":
    main()
