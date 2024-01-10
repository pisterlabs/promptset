import json
from langchain.llms import HuggingFaceTextGenInference
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import re


class chatsearcher:
    def __init__(self, embedding, skillfit_model):
        self.embedding = embedding
        self.skillfit_model = skillfit_model

    def predict(
        self,
        skilldb,
        skill_taxonomy,
        doc,
        los,
        known_skills,
        filterconcepts,
        top_k,
        strict,
        trusted_score,
        temperature,
        use_llm,
        request_timeout,
        llm_validation,
        skillfit_validation,
    ):
        if len(los) > 0:
            learningoutcomes = "\n".join(los)
            searchterms = los
        elif use_llm:
            print("Extracting los from doc using llm")
            prompt = "Du bist ein KI-Assistent, der auf Kursbeschreibungen spezialisiert ist. Deine Aufgabe ist es, eine Liste von Lernzielen aus einer gegebenen Kursbeschreibung zu extrahieren. Benannte Vorraussetzunge und Zielgruppen sollen ignoriert werden. Gehe dabei Schritt für Schritt vor. Erfasse zuerst die groben Themen, dann identifiziere zu jedem Thema die vermittelten Fähigkeiten."
            prompt += "\n\nBitte identifiziere die Lernziele in der folgenden Kursbeschreibung:"
            prompt += "\n\n" + doc
            prompt += "\n\nErstelle eine Liste der Lernziele, wobei jedes Lernziel in einer neuen Zeile stehen soll. Die Antwort sollte nur die Lernziele selbst enthalten, ohne Einleitungen oder zusätzliche Worte. Nutze kurze und einfache Sprache sowie BLOOM-Verben für Fähigkeiten."

            max_tokens = 4000
            max_new_tokens = 512
            max_input_tokens = max_tokens - max_new_tokens
            prompt = prompt[:max_input_tokens]
            llm = HuggingFaceTextGenInference(
                inference_server_url="https://em-german-13b.llm.mylab.th-luebeck.dev/",
                max_new_tokens=max_new_tokens,
                top_k=10,
                top_p=0.95,
                typical_p=0.95,
                temperature=temperature,
                repetition_penalty=1.03,
            )
            learningoutcomes = llm(prompt)

            # Remove String " ASSISTANT: " from start of learningoutcomes
            learningoutcomes = learningoutcomes.replace("ASSISTANT: ", "").strip()
            # Remove list decorations using regular expressions
            learningoutcomes = re.sub(
                r"^ *[\d.-]+ *|^ *\* *|^ *- *", "", learningoutcomes, flags=re.MULTILINE
            )

            searchterms = learningoutcomes.split("\n")
        else:
            learningoutcomes = doc
            searchterms = []

        predictions = []

        embedded_doc = self.embedding.embed_documents([learningoutcomes])

        known_skill_uris = [skill["uri"] for skill in known_skills]
        known_skill_labels = [skill["title"] for skill in known_skills]
        doc = " ".join(known_skill_labels) + " " + doc

        if skill_taxonomy != "ESCO":
            relevant_skills = skilldb.similarity_search_with_score(doc, top_k)
        if len(filterconcepts):
            relevant_skills = skilldb.similarity_search_with_score(doc, top_k * 5)
        else:
            relevant_skills = skilldb.similarity_search_with_score(doc, top_k * 2)

        for relevant_skill in relevant_skills:
            if skill_taxonomy == "ESCO":
                if relevant_skill[0].metadata["conceptUri"] in known_skill_uris:
                    continue
                # If filterconcepts are set, exlcude all terms that are not a child of either of the cocepts.
                if (
                    len(filterconcepts)
                    and relevant_skill[0].metadata["broaderHierarchyConcepts"]
                ):
                    broaderconcepts = json.loads(
                        relevant_skill[0].metadata["broaderHierarchyConcepts"]
                    )
                    is_part_of_concept = False
                    for broaderconcept in broaderconcepts:
                        if broaderconcept["uri"] in filterconcepts:
                            is_part_of_concept = True
                            break

                    if not is_part_of_concept:
                        continue

                embedded_skill = self.embedding.embed_documents(
                    [relevant_skill[0].metadata["preferredLabel"]]
                )

                predictions.append(
                    {
                        "uri": relevant_skill[0].metadata["conceptUri"],
                        "title": relevant_skill[0].metadata["preferredLabel"],
                        "className": "Skill",
                        "score": relevant_skill[1],
                        "fit": True,
                        "skill_embedding": embedded_skill[0],
                        # 'broaderConcepts': relevant_skill[0].metadata["broaderHierarchyConcepts"]
                    }
                )
            else:
                embedded_skill = self.embedding.embed_documents(
                    [relevant_skill[0].page_content]
                )

                predictions.append(
                    {
                        "uri": relevant_skill[0].metadata["id"],
                        "title": relevant_skill[0].page_content,
                        "className": "Skill",
                        "score": relevant_skill[1],
                        "fit": True,
                        "skill_embedding": embedded_skill[0],
                        # 'broaderConcepts': relevant_skill[0].metadata["broaderHierarchyConcepts"]
                    }
                )

        if not llm_validation and not skillfit_validation:
            # Define artificial threshholds for relevancy by identifying where the similarity rating decreases the fastest.
            if strict > 0 and len(predictions) > 2:
                # Identify the biggest and second biggest gap between the skills with scores higher than 0.2.
                gaps = []
                for i in range(len(predictions) - 1):
                    gaps.append(predictions[i + 1]["score"] - predictions[i]["score"])

                # Get the idecies of the two largest gaps.
                max_gap_skill_index = gaps.index(max(gaps)) + 1
                if strict == 3:
                    predictions = predictions[:max_gap_skill_index]
                elif strict <= 2:
                    max_gap = 0
                    max_gap_skill_index_2 = 0
                    for i in range(max_gap_skill_index + 1, len(predictions) - 1):
                        if predictions[i]["score"] < trusted_score:
                            continue
                        gap = predictions[i + 1]["score"] - predictions[i]["score"]
                        if gap > max_gap:
                            max_gap = gap
                            max_gap_skill_index_2 = i

                    if strict == 1:
                        max_gap = 0
                        max_gap_skill_index_3 = 0
                        for i in range(max_gap_skill_index_2 + 1, len(predictions) - 1):
                            if predictions[i]["score"] < trusted_score:
                                continue
                            gap = predictions[i + 1]["score"] - predictions[i]["score"]
                            if gap > max_gap:
                                max_gap = gap
                                max_gap_skill_index_3 = i

                        predictions = predictions[: max_gap_skill_index_3 + 1]
                    else:
                        predictions = predictions[: max_gap_skill_index_2 + 1]

        if skill_taxonomy == "ESCO":
            # Predictions base on the known skills.
            for known_skill in known_skills:
                predictions += self.predict_for_skill(known_skill, embedded_doc)

        # Remove knwon skills and duplicates from predictions.
        seen = []
        todelete = []
        for i in range(len(predictions)):
            if (
                predictions[i]["uri"] in seen
                or predictions[i]["uri"] in known_skill_uris
            ):
                todelete.append(i)
            else:
                seen.append(predictions[i]["uri"])

        results = []
        for i in range(len(predictions)):
            if i not in todelete:
                results.append(predictions[i])

        validated = []
        if llm_validation:
            print("Validating search results using LLM-Chat")
            skilllabels = []
            filtered = []
            for prediction in results:
                skilllabels.append(prediction["title"])

            prompt = "Es soll geprüft werden welche Lernziele von dem folgenden Kurs vermittelt werden. \nKursbeschreibung: "
            prompt += "\n\n" + doc
            prompt += (
                "\n\n"
                + "Welche der folgenden Kompetenzen werden nicht im Kurs vermittelt:"
            )
            prompt += "\n\n".join(skilllabels)
            prompt += (
                "\n\nDie Kompetenzen, die nicht zur Kursbeschreibung passen, sind:"
            )
            max_tokens = 4000
            max_new_tokens = 512
            max_input_tokens = max_tokens - max_new_tokens
            prompt = prompt[:max_input_tokens]
            llm = HuggingFaceTextGenInference(
                inference_server_url="https://em-german-70b.llm.mylab.th-luebeck.dev/",
                max_new_tokens=max_new_tokens,
                top_k=10,
                top_p=0.95,
                typical_p=0.95,
                temperature=temperature,
                repetition_penalty=1.03,
            )
            chatresponse = llm(prompt)

            # from chatresponse get all lines that start with -
            lines = chatresponse.split("\n")
            # Remove "- " from start of every line if exists
            errors = [line[2:] for line in lines if line.startswith("- ")]

            for i in range(len(results)):
                fit = results[i]["title"] in errors
                results[i]["fit"] = fit

                if fit:
                    if strict > 1:
                        validated.append(results[i])
                    if strict > 0:
                        # Score reward for skills that are marked as fitting.
                        results[i]["score"] -= 0.06
        elif skillfit_validation and skill_taxonomy == "ESCO":
            print("Validating search results using Skillfit-Model")
            if use_llm or len(los) > 0:
                skillfit = self.skillfit_model.predict(
                    self.prepare_data_for_prediction(
                        self.embedding.embed_documents([learningoutcomes]), predictions
                    )
                )
            else:
                skillfit = self.skillfit_model.predict(
                    self.prepare_data_for_prediction(embedded_doc, predictions)
                )

            # Set fit value for results to 1 if skillfit is 1
            for i in range(len(results)):
                if i not in range(len(skillfit)):
                    raise Exception("Index out of bounds")

                fit = bool(skillfit[i][0])
                results[i]["fit"] = fit

                if fit:
                    if strict > 1:
                        validated.append(results[i])
                    if strict > 0:
                        # Score reward for skills that are marked as fitting.
                        results[i]["score"] -= 0.06

        if (
            (llm_validation or skillfit_validation)
            and strict > 1
            and len(validated) > 0
        ):
            results = validated

        # Unset skill_embedding param for every object in results
        for result in results:
            del result["skill_embedding"]

        results = sorted(results, key=lambda x: x["score"], reverse=False)

        return {"status": "200", "searchterms": searchterms, "results": results[:top_k]}

    def predict_for_skill(self, skill, embedded_doc):
        predictions = []
        relevant_skills = skilldb.similarity_search_with_score(skill["title"], 3)
        for relevant_skill in relevant_skills:
            if relevant_skill[0].metadata["conceptUri"] == skill["uri"]:
                continue

            embedded_skill = self.embedding.embed_documents(
                [relevant_skill[0].metadata["preferredLabel"]]
            )
            similarities = cosine_similarity(embedded_doc, embedded_skill)

            predictions.append(
                {
                    "uri": relevant_skill[0].metadata["conceptUri"],
                    "title": relevant_skill[0].metadata["preferredLabel"],
                    "className": "Skill",
                    "score": 1 - similarities[0][0].item(),
                    "fit": True,
                    "skill_embedding": embedded_skill[0],
                    # 'broaderConcepts': relevant_skill[0].metadata["broaderHierarchyConcepts"]
                }
            )

        return predictions

    def prepare_data_for_prediction(self, embedded_doc, skill_data):
        # Extract skill embeddings and similarity scores
        skill_embeddings = [item["skill_embedding"] for item in skill_data]
        similarity_scores = [item["score"] for item in skill_data]

        # Convert to numpy arrays
        skill_embeddings = np.array(skill_embeddings)
        similarity_scores = np.array(similarity_scores)

        # Reshape similarity_scores to be a column vector
        similarity_scores = similarity_scores.reshape(-1, 1)

        # Repeat the embedded_doc and similarity_scores for each skill_embedding
        num_skills = len(skill_embeddings)
        embedded_doc_repeated = np.repeat(embedded_doc, num_skills, axis=0)

        # Flatten the arrays and concatenate them to form the input data
        prepared_data = np.concatenate(
            (embedded_doc_repeated, skill_embeddings, similarity_scores), axis=1
        )

        return prepared_data
