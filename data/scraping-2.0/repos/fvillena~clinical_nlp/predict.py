#!/opt/conda/bin/python
# -*- coding: utf-8 -*-

from transformers import pipeline, HfArgumentParser
from datasets import load_dataset
from clinical_nlp.arguments import (
    DataArguments,
    ModelArguments,
    PipelineArguments,
    PredictionArguments,
)
from clinical_nlp.models import IclClassifier, IclNer
from llmner import ZeroShotNer, FewShotNer
import re

if __name__ == "__main__":
    parser = HfArgumentParser(
        (DataArguments, ModelArguments, PipelineArguments, PredictionArguments)
    )
    (
        data_args,
        model_args,
        pipeline_arguments,
        prediction_arguments,
    ) = parser.parse_args_into_dataclasses()
    data = load_dataset(data_args.dataset_name, revision=data_args.dataset_revision)
    test_data = data["test"]
    if data_args.dataset_sample:
        test_data = test_data.select(
            range(int(len(test_data) * data_args.dataset_sample))
        )
    if model_args.task == "text-classification":
        if "http" in model_args.model_name_or_path:
            classes = list(set(test_data["label"]))
            if not model_args.api_key:
                model = IclClassifier(model_args.model_name_or_path)
            else:
                model = IclClassifier(
                    model_args.model_name_or_path,
                    api_key=model_args.api_key,
                    model_name=model_args.openai_model_name,
                    stop=["###"]
                )
            if "spanish_diagnostics" in data_args.dataset_name:
                model.contextualize(
                    system_message="Eres un asistente serio que sólo da respuestas precisas y concisas que recibirá diagnósticos en Español y deberás sólo responder con el nombre de la especialidad en Español a la cual debe enviarse el diagnóstico. Las especialidades disponibles son: <classes>.",
                    user_template='¿A qué especialidad debo enviar el diagnóstico "<x>"?.',
                    classes=classes,
                    retry_message="No encuentro en tu mensaje ninguna de las especialidades de la lista de especialidades disponibles. Por favor, intenta nuevamente.",
                )
                test_data = test_data.map(
                    lambda x: {"prediction": model.predict(x["text"])}, num_proc=12
                )
            elif "ges" in data_args.dataset_name:
                system_message = 'En Chile, las garantías explícitas de salud establecen prioridad para un conjunto de problemas de salud. Debes responder en español sólo la palabra "Verdadero" si la enfermedad que te entregue pertenece a uno de los 80 problemas de salud y sólo la palabra "Falso" si la enfermedad no pertenece al conjunto de problemas. Los problemas de salud son: "Accidente Cerebrovascular Isquémico en personas de 15 años y más", "Alivio del dolor y cuidados paliativos por cáncer avanzado ", "Analgesia del Parto", "Artritis Reumatoídea", "Artritis idiopática juvenil", "Asma Bronquial moderada y grave en personas menores de 15 años", "Asma bronquial en personas de 15 años y más", "Cardiopatías congénitas operables en menores de 15 años", "Colecistectomía preventiva del cáncer de vesícula en personas de 35 a 49 años", "Consumo Perjudicial o Dependencia de riesgo bajo a moderado de alcohol y drogas en personas menores de 20 años", "Cáncer Cervicouterino", "Cáncer Colorectal en personas de 15 años y más", "Cáncer Vesical en personas de 15 años y más", "Cáncer de Ovario Epitelial", "Cáncer de mama en personas de 15 años y más", "Cáncer de próstata en personas de 15 años y más", "Cáncer de testículo en personas de 15 años y más", "Cáncer en personas menores de 15 años", "Cáncer gástrico", "Depresión en personas de 15 años y más", "Desprendimiento de retina regmatógeno no traumático", "Diabetes Mellitus Tipo 1", "Diabetes Mellitus Tipo 2", "Displasia broncopulmonar del prematuro", "Displasia luxante de caderas", "Disrafias espinales", "Endoprótesis total de cadera en personas de 65 años y más con artrosis de cadera con limitación funcional severa", "Enfermedad Pulmonar Obstructiva Crónica de Tratamiento Ambulatorio", "Enfermedad Renal Crónica Etapa 4 y 5", "Enfermedad de Parkinson", "Epilepsia no refractaria en personas de 15 años y más", "Epilepsia no refractaria en personas desde 1 año y menores de 15 años", "Esclerosis múltiple remitente recurrente ", "Esquizofrenia", "Estrabismo en personas menores de 9 años", "Fibrosis Quística", "Fisura labiopalatina", "Gran Quemado", "Hemofilia", "Hemorragia Subaracnoidea secundaria a Ruptura de Aneurismas Cerebrales", "Hepatitis C", "Hepatitis crónica por Virus Hepatitis B", "Hipertensión arterial primaria o esencial en personas de 15 años y más", "Hipoacusia Bilateral en personas de 65 años y más que requieren uso de audífono", "Hipoacusia neurosensorial bilateral del prematuro", "Hipotiroidismo en personas de 15 años y más", "Infarto agudo del miocardio", "Infección respiratoria aguda (IRA) de manejo ambulatorio en personas menores de 5 años", "Leucemia en personas de 15 años y más", "Linfomas en personas de 15 años y más", "Lupus Eritematoso Sistémico", "Neumonía adquirida en la comunidad de manejo ambulatorio en personas de 65 años y más", "Osteosarcoma en personas de 15 años y más", "Politraumatizado Grave", "Prevención de Parto Prematuro", "Prevención secundaria enfermedad renal crónica terminal", "Retinopatía del prematuro", "Retinopatía diabética", "Salud Oral Integral del adulto de 60 años", "Salud oral integral de la embarazada", "Salud oral integral para niños y niñas de 6 años", "Síndrome de Dificultad Respiratoria en el recién nacido", "Síndrome de la inmunodeficiencia adquirida VIH/SIDA", "Trastorno Bipolar en personas de 15 años y más", "Trastornos de generación del impulso y conducción en personas de 15 años y más, que requieren Marcapaso", "Tratamiento Médico en personas de 55 años y más con Artrosis de Cadera y/o Rodilla, leve o moderada", "Tratamiento Quirúrgico de Hernia del Núcleo Pulposo Lumbar", "Tratamiento Quirúrgico de lesiones crónicas de la válvula aórtica en personas de 15 años y más", "Tratamiento Quirúrgico de lesiones crónicas de las válvulas mitral y tricúspide en personas de 15 años y más", "Tratamiento de Erradicación del Helicobacter Pylori", "Tratamiento de Hipoacusia moderada en personas menores de 4 años", "Tratamiento de la hiperplasia benigna de la próstata en personas sintomáticas", "Tratamiento quirúrgico de cataratas", "Tratamiento quirúrgico de escoliosis en personas menores de 25 años", "Trauma Ocular Grave", "Traumatismo Cráneo Encefálico moderado o grave", "Tumores Primarios del Sistema Nervioso Central en personas de 15 años o más", "Urgencia Odontológica Ambulatoria", "Vicios de refracción en personas de 65 años y más" y "Órtesis (o ayudas técnicas) para personas de 65 años y más"'
                user_template = '¿"<x>" pertenece a la lista de 80 problemas de salud priorizados por las garantías explícitas de salud?.'
                classes = {
                    "Falso": ["Falso", "False"],
                    "Verdadero": [
                        "Verdadero",
                        "Verídico",
                        "Verdadeiro",
                        "Veradero",
                        "True",
                    ],
                }
                classes_map = {"false": "Falso", "true": "Verdadero"}
                retry_message = 'Responde sólo con la palabra "Verdadero" si la enfermedad pertenece a uno de los 80 problemas de salud y sólo la palabra "Falso" si la enfermedad no pertenece al conjunto de problemas. Por favor, intenta nuevamente.'
                if not model_args.few_shot:
                    model.contextualize(
                        system_message=system_message,
                        user_template=user_template,
                        classes=classes,
                        retry_message=retry_message,
                    )
                else:
                    shots = []
                    n = 5
                    for c in classes.keys():
                        m = 0
                        for example in data["train"]:
                            if classes_map[example["label"]] == c:
                                m += 1
                                if m <= n:
                                    shots.append(
                                        {
                                            "text": user_template.replace(
                                                "<x>", example["text"]
                                            ),
                                            "label": c,
                                        }
                                    )
                                else:
                                    break
                    model.contextualize(
                        system_message=system_message,
                        user_template=user_template,
                        classes=classes,
                        retry_message=retry_message,
                        examples=shots,
                    )

                def parse_prediction(prediction):
                    response = None
                    if prediction == "VERDADERO":
                        response = "true"
                    elif prediction == "FALSO":
                        response = "false"
                    return response

                test_data = test_data.map(
                    lambda x: {"prediction": parse_prediction(model.predict(x["text"]))}, num_proc=12, load_from_cache_file=False
                )
                true = test_data["label"]
                predicted = test_data["prediction"]
        else:
            pipe = pipeline(
                "text-classification",
                model=model_args.model_name_or_path,
                tokenizer=model_args.tokenizer_name,
                device=pipeline_arguments.device,
            )
            predictions = pipe(
                test_data["text"],
                batch_size=pipeline_arguments.batch_size,
                truncation=pipeline_arguments.truncation,
            )
            true = test_data["label"]
            predicted = [p["label"] for p in predictions]
        with open(prediction_arguments.prediction_file, "w", encoding="utf-8") as f:
            for t, p in zip(true, predicted):
                f.write(f"{t}\t{p}\n")
    elif model_args.task == "ner":
        label_list = data["train"].features[f"ner_tags"].feature.names
        if "http" in model_args.model_name_or_path:
            from langchain.globals import set_llm_cache
            from langchain.cache import InMemoryCache, SQLiteCache
            import os

            set_llm_cache(InMemoryCache())
            # set_llm_cache(SQLiteCache(database_path=".langchain.db"))

            os.environ["OPENAI_API_BASE"] = model_args.model_name_or_path
            os.environ["OPENAI_API_KEY"] = model_args.api_key

            entities = {
                "disease": "alteración o desviación del estado fisiológico en una o varias partes del cuerpo, por causas en general conocidas, manifestada por síntomas y signos característicos, y cuya evolución es más o menos previsible",
                "medication": "Medicamentos o drogas empleadas en el tratamiento y o prevención de enfermedades",
                "abbreviation": "Abreviatura",
                "body_part": "Órgano o una parte anatómica de una persona",
                "family_member": "Miembro de la familia",
                "laboratory_or_test_result": "Resultado de laboratorio o test",
                "clinical_finding": "Observaciones, juicios o evaluaciones que se hacen sobre los pacientes",
                "diagnostic_procedure": "Exámenes que permiten determinar la condición del individuo ",
                "laboratory_procedure": "Exámenes que se realizan en diversas muestras de pacientes que permiten diagnosticar enfermedades mediante la detección de biomarcadores y otros parámetros",
                "therapeutic_procedure": "Actividad o tratamiento que es empleado para prevenir, reparar, eliminar o curar la enfermedad del individuo",
            }

            prompt_template = """Eres reconocedor de entidades nombradas que solo debe detectar las entidades en la siguiente lista: 
{entities} 
Debes responder con el mismo texto de entrada, pero con las entidades nombradas anotadas con etiquetas en la misma línea (<nombre_entidad>lorem ipsum</nombre_entidad>), donde cada etiqueta corresponde a un nombre de entidad, por ejemplo: <entidad>Sed ut perspiciatis</entidad> unde omnis iste natus error sit voluptatem <entidad>accusantium</entidad>.
Las únicas etiquetas disponibles son: {entity_list}, no puedes agregar más etiquetas de las incluidas en esa lista.
IMPORTANTE: NO DEBES CAMBIAR EL TEXTO DE ENTRADA, SÓLO AGREGAR LAS ETIQUETAS."""

            if not model_args.few_shot:
                model = ZeroShotNer(max_tokens=1024, model=model_args.openai_model_name)
                model.contextualize(
                    entities=entities,
                    prompt_template=prompt_template,
                )
            else:
                from llmner.utils import (
                    inline_annotation_to_annotated_document,
                    conll_to_inline_annotated_string,
                )

                label_list = list(
                    map(
                        lambda x: x.capitalize(),
                        data["train"].features[f"ner_tags"].feature.names,
                    )
                )
                label2id = dict(zip(label_list, range(len(label_list))))
                id2label = dict(zip(label2id.values(), label2id.keys()))
                shots = []
                for document in data["train"]:
                    conll = list(
                        zip(
                            document["tokens"],
                            map(lambda x: id2label[x], document["ner_tags"]),
                        )
                    )
                    annotated_string = conll_to_inline_annotated_string(conll)
                    annotated_document = inline_annotation_to_annotated_document(
                        annotated_string, entity_set=list(entities.keys())
                    )
                    shots.append(annotated_document)
                model = FewShotNer(max_tokens=1024, model=model_args.openai_model_name)
                model.contextualize(
                    entities=entities,
                    prompt_template=prompt_template,
                    examples=shots[:5],
                )
            test_data = test_data.map(
                lambda x: {
                    "prediction": [
                        t[1] for t in model.predict_tokenized([x["tokens"]])[0]
                    ]
                },
                num_proc=12,
            )

            sentences = test_data["tokens"]
            true = list(
                map(
                    lambda x: [label_list[z].capitalize() for z in x],
                    test_data["ner_tags"],
                )
            )
            predicted = test_data["prediction"]
        else:
            from transformers import (
                AutoTokenizer,
                TokenClassificationPipeline,
                AutoModelForTokenClassification,
            )

            def tokenize_and_align_labels(examples):
                tokenized_inputs = tokenizer(
                    examples["tokens"], truncation=True, is_split_into_words=True
                )

                labels = []
                for i, label in enumerate(examples[f"ner_tags"]):
                    word_ids = tokenized_inputs.word_ids(
                        batch_index=i
                    )  # Map tokens to their respective word.
                    previous_word_idx = None
                    label_ids = []
                    for word_idx in word_ids:  # Set the special tokens to -100.
                        if word_idx is None:
                            label_ids.append(-100)
                        elif (
                            word_idx != previous_word_idx
                        ):  # Only label the first token of a given word.
                            label_ids.append(label[word_idx])
                        else:
                            label_ids.append(-100)
                        previous_word_idx = word_idx
                    labels.append(label_ids)

                tokenized_inputs["labels"] = labels
                return tokenized_inputs

            def process_true_labels(example):
                true = [
                    -100 if i == -100 else pipe.model.config.id2label[i]
                    for i in example
                ]
                interim = []
                for i, piece in enumerate(true):
                    if i == 0:
                        interim.append("O")
                    elif i == len(true) - 1:
                        interim.append("O")
                    elif piece != -100:
                        interim.append(piece)
                    else:
                        last_piece = interim[i - 1]
                        interim.append(
                            last_piece
                            if not last_piece.startswith("B-")
                            else last_piece.replace("B-", "I-")
                        )
                true = interim
                return true

            def process_predicted_labels(prediction, sentence):
                interim = ["O"] * len(sentence)
                for result in prediction:
                    interim[result["index"]] = result["entity"]
                predicted = interim
                return predicted

            class MyTokenClassificationPipeline(TokenClassificationPipeline):
                def preprocess(
                    self, sentence, offset_mapping=None, **preprocess_params
                ):
                    tokenizer_params = preprocess_params.pop("tokenizer_params", {})
                    truncation = (
                        True
                        if self.tokenizer.model_max_length
                        and self.tokenizer.model_max_length > 0
                        else False
                    )
                    inputs = self.tokenizer(
                        sentence,
                        return_tensors=self.framework,
                        truncation=truncation,
                        return_special_tokens_mask=True,
                        return_offsets_mapping=self.tokenizer.is_fast,
                        is_split_into_words=True,
                        **tokenizer_params,
                    )
                    inputs.pop("overflow_to_sample_mapping", None)
                    num_chunks = len(inputs["input_ids"])

                    for i in range(num_chunks):
                        if self.framework == "tf":
                            raise Exception(
                                "TensorFlow pipelines are currently not supported."
                            )
                        else:
                            model_inputs = {
                                k: v[i].unsqueeze(0) for k, v in inputs.items()
                            }
                        if offset_mapping is not None:
                            model_inputs["offset_mapping"] = offset_mapping
                        model_inputs["sentence"] = sentence if i == 0 else None
                        model_inputs["is_last"] = i == num_chunks - 1

                        yield model_inputs

            tokenizer = AutoTokenizer.from_pretrained(
                model_args.tokenizer_name, add_prefix_space=True
            )
            tokenized_test_data = test_data.map(tokenize_and_align_labels, batched=True)
            model = AutoModelForTokenClassification.from_pretrained(
                model_args.model_name_or_path
            )
            pipe = MyTokenClassificationPipeline(
                model=model,
                tokenizer=tokenizer,
                device=0,
                aggregation_strategy="none",
            )
            sentences = list(
                map(
                    pipe.tokenizer.convert_ids_to_tokens,
                    tokenized_test_data["input_ids"],
                )
            )
            true = list(map(process_true_labels, tokenized_test_data["labels"]))
            predictions = pipe(test_data["tokens"])
            predicted = []
            for prediction, sentence in zip(predictions, sentences):
                predicted.append(process_predicted_labels(prediction, sentence))
        with open(prediction_arguments.prediction_file, "w", encoding="utf-8") as f:
            for i in range(len(sentences)):
                for piece, t, p in zip(sentences[i], true[i], predicted[i]):
                    f.write(f"{piece}\t{t}\t{p}\n")
                f.write("\n")
