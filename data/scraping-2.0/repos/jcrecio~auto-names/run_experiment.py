import sys
import json
import os

from open_api_model import OpenAiModel

from objects import (
    ClassObject,
    ExtractionMethodContent,
    Method,
    OriginalMethodContent,
    Project,
    Error,
    serialize,
)
from utils import (
    compute_method_similarity,
    get_dataset_item,
    log,
    read_file,
    write_file,
)

from_folder = sys.argv[1]
model = OpenAiModel(sys.argv[2])
output_folder = sys.argv[3]
projects = dict()

########## Load dataset from folder argument ##########

log("Loading dataset from {from_folder}...\n".format(from_folder=from_folder))
for file_name in os.listdir(from_folder):
    code = read_file(os.path.join(from_folder, file_name))
    item = get_dataset_item(file_name, code)

    if item.project not in projects:
        projects[item.project] = Project()
    if item.class_name not in projects[item.project].classes:
        projects[item.project].classes[item.class_name] = ClassObject()
    if item.method_name not in projects[item.project].classes[item.class_name].methods:
        projects[item.project].classes[item.class_name].methods[
            item.method_name
        ] = Method()
    if item.variation == "Original":
        projects[item.project].classes[item.class_name].methods[
            item.method_name
        ].set_original(
            OriginalMethodContent(
                item.method_name, item.code.replace("\n", ""), file_name
            )
        )
    else:
        projects[item.project].classes[item.class_name].methods[
            item.method_name
        ].add_extraction(
            ExtractionMethodContent(
                item.method_name, item.code.replace("\n", ""), file_name, item.variation
            )
        )
log("Dataset fully loaded.\n")


########## Run the experiment to get the method names prediction ##########

log("Starting methods prediction...")

for project_key, project in projects.items():
    print("\n\n")
    log("- Project: " + project_key)

    for class_key, class_name in project.classes.items():
        log("--- Class: " + class_key)

        for method_key, method in class_name.methods.items():
            log("----- Target method: " + method.original.method_name)
            if os.path.exists(
                output_folder
                + "/"
                + project_key
                + "."
                + class_key
                + "."
                + method_key
                + ".json"
            ):
                log("------ Method prediction already exists, skipping method...")
                continue

            accumulative_extractions_predictions = dict()
            extraction_number = 1

            halt_method = False
            for extraction in method.extractions:
                if halt_method == True:
                    break
                extraction_with_previous_predictions = extraction.code

                for acc in accumulative_extractions_predictions:
                    extraction_with_previous_predictions = (
                        extraction_with_previous_predictions.replace(
                            acc, accumulative_extractions_predictions[acc]
                        ).replace(
                            acc + "_method", accumulative_extractions_predictions[acc]
                        )
                    )

                next_extraction_index = extraction_with_previous_predictions.find(
                    method_key + "_extraction_"
                )
                next_extraction = extraction_with_previous_predictions[
                    next_extraction_index : next_extraction_index
                    + 1
                    + len(method_key + "_extraction_")
                ]

                extraction_prediction_response = model.predict_extraction(
                    method.original.code,
                    extraction_with_previous_predictions,
                    next_extraction,
                )

                if isinstance(extraction_prediction_response, Error):
                    extraction_prediction_response.method_name = method_key
                    halt_method = True
                    break

                extraction.prediction = extraction_prediction_response
                extraction.updated_code = extraction_with_previous_predictions.replace(
                    next_extraction, extraction.prediction
                ).replace(next_extraction + "_method", extraction.prediction)

                accumulative_extractions_predictions[
                    next_extraction
                ] = extraction.prediction

                log("------- Extraction: " + extraction.extraction_name)
                log("------- Prediction: " + extraction.prediction)

                original_prediction_after_next_extraction = model.predict(
                    extraction.updated_code
                )
                method.original.predictions.append(
                    original_prediction_after_next_extraction
                )
                similarity = compute_method_similarity(
                    method.original.method_name,
                    original_prediction_after_next_extraction,
                )
                method.original.similarities.append(similarity)
                log(
                    "----- Predicted method for target method after extraction {extraction_number}: {prediction}".format(
                        extraction_number=extraction_number,
                        prediction=original_prediction_after_next_extraction,
                    )
                )
                log(
                    "----- Similarity between original method name and predicted method name after extraction {extraction_number}: {similarity} ".format(
                        extraction_number=extraction_number, similarity=str(similarity)
                    )
                )
                extraction_number += 1

            json_str = json.dumps(method, default=serialize, indent=4)
            write_file(
                output_folder
                + "/"
                + project_key
                + "."
                + class_key
                + "."
                + method_key
                + ".json",
                json_str,
            )
