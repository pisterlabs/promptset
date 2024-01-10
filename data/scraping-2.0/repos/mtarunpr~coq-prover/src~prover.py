from openai import OpenAI

from alectryon.serapi import annotate, Sentence, Text
from tenacity import retry, stop_after_attempt, wait_random_exponential
from joblib import Memory
from contextlib import redirect_stderr
from dotenv import load_dotenv
import os
import re
import argparse

load_dotenv()

memory = Memory("cachegpt", verbose=0)

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
MODEL = "gpt-4-1106-preview"
MAX_LEMMA_DEPTH = 5
MAX_THEOREM_ERROR_COUNT = 20


# Caching process ported from https://github.com/metareflection/gpt-call
@retry(wait=wait_random_exponential(min=10, max=30), stop=stop_after_attempt(25))
def generate(messages, model):  # "gpt-3.5-turbo", "gpt-4"
    print("calling GPT... model=" + model)
    return client.chat.completions.create(model=model, messages=messages)


@memory.cache
def ask(messages, model):
    response = generate(messages, model)
    return response.choices[0].message.content


def prove_using_gpt(context, theorem_or_lemma, model, prev_attempt_with_error=None):
    messages = [
        {
            "role": "system",
            "content": "You are an automated theorem prover that can prove theorems and lemmas in Coq. Your entire response must be valid Coq code. You should explain your reasoning (what the proof steps are attempting to do), but only in comments inside the Coq code. The following messages will all consist of a theorem statement (possibly preceded by necessary definitions, imports, etc.), and your response must be a valid Coq proof of that theorem. Your response must be in this format: ```coq\n Proof.\n<proof>. Qed.\n```. Remember: do not add any other text besides Coq code and do not repeat any imports, definitions, lemmas, etc. provided in the prompt.",
        },
        {"role": "user", "content": context + "\n\n" + theorem_or_lemma},
    ]
    if prev_attempt_with_error is not None:
        prev_attempt, error = prev_attempt_with_error
        messages += [
            {"role": "assistant", "content": "```coq" + prev_attempt + "\n```"},
            {
                "role": "user",
                "content": "This is incorrect; Coq produced the following error message: "
                + error
                + "\n\nPlease try again.",
            },
        ]
    response = ask(messages, model)
    proof_contents = response.split("Proof.")[1].split("Qed.")[0]
    return "Proof.\n" + proof_contents + "\nQed."


def annotate_and_fetch_error(context, theorem_with_proof):
    first_error_idx = -1
    annotated_proof = annotate([context + "\n\n" + theorem_with_proof])
    # A Fragment is a Sentence (proof step) or a Text (comment)
    annotated_proof_fragments = []
    i = 0
    for step in annotated_proof[0]:
        if isinstance(step, Sentence) and len(step.messages) > 0:
            if first_error_idx == -1 and not all(
                "deprecated" in message.contents for message in step.messages
            ):
                first_error_idx = i
        annotated_proof_fragments.append(step)
        i += 1
    return annotated_proof_fragments, first_error_idx


def create_lemma_name(lemma, suffix):
    messages = [
        {
            "role": "system",
            "content": "You are a proof helper for Coq that can come up with descriptive names for lemmas and theorems based on the statement of the proposition. Specifically, Replace `helper_lemma` with a better, more descriptive, name for the following lemma(s) in Coq. Your entire response must be valid Coq code. Your response must be in this format: ```coq\nLemma <new_lemma_name> : <lemma_statement>.\n```.",
        },
        {"role": "user", "content": lemma},
    ]
    response = ask(messages, MODEL)
    new_lemma_name = response.split("Lemma ")[1].split(":")[0].strip()
    return new_lemma_name + "_" + suffix


def proof_state_to_lemma(lemma_name_suffix, hypotheses, conclusion):
    lemma = f"Lemma helper_lemma : "
    if len(hypotheses) > 0:
        for hypothesis in hypotheses:
            lemma += (
                "forall " + " ".join(hypothesis.names) + " : " + hypothesis.type + ", "
            )
    lemma += conclusion + ".\n"

    # Replace "helper_lemma" with a better name
    lemma_name = create_lemma_name(lemma, lemma_name_suffix)
    lemma = lemma.replace("Lemma helper_lemma : ", f"Lemma {lemma_name} : ")

    return lemma


def recursively_prove_lemma(
    context,
    lemma,
    depth=0,
    prev_attempt_lemma_with_proof=None,
    prev_attempt_error_message=None,
    prev_attempt_error_idx=None,
):
    # If a previous attempt had an error, print it
    if prev_attempt_error_message is not None:
        print(f"ERROR MESSAGE IN LEMMA PROOF (FRAGMENT #{prev_attempt_error_idx})")
        print(prev_attempt_error_message)
        print()

    # Break out of recursion if we've reached the max depth
    if depth > MAX_LEMMA_DEPTH:
        print("MAX LEMMA DEPTH REACHED. GIVING UP.")
        exit(1)

    # If this is the first attempt, try to prove the lemma
    if depth == 0:
        proof = prove_using_gpt(context, lemma, MODEL)
    # Otherwise, try to prove the lemma again using the previous attempt's error message
    else:
        print(f"LEMMA PROOF IS INVALID. TRYING AGAIN... (ATTEMPT {depth})")
        print()
        proof = prove_using_gpt(
            context,
            lemma,
            MODEL,
            (
                prev_attempt_lemma_with_proof,
                prev_attempt_error_message,
            ),
        )

    # Print the lemma's proof
    lemma_with_proof = lemma + "\n" + proof
    print("GPT PROOF OF LEMMA")
    print(lemma_with_proof)
    print()

    # Check if lemma's proof is valid
    annotated_proof_fragments, first_error_idx = annotate_and_fetch_error(
        context, lemma_with_proof
    )

    # If invalid, try again recursively
    if first_error_idx != -1:
        # Get the closest Sentence before the error
        for i in range(first_error_idx - 1, -1, -1):
            if isinstance(annotated_proof_fragments[i], Sentence):
                prev_sentence = annotated_proof_fragments[i]
                break
        # Get first non-"deprecated" error message
        for message in annotated_proof_fragments[first_error_idx].messages:
            if "deprecated" not in message.contents:
                error_message = f'Error in step "{annotated_proof_fragments[first_error_idx].contents}".\nMessage: {message.contents}.\nGoal: {prev_sentence.goals[0].conclusion}.'
                break
        return recursively_prove_lemma(
            context,
            lemma,
            depth + 1,
            lemma_with_proof,
            error_message,
            first_error_idx,
        )
    # Otherwise, return the lemma's proof
    else:
        print("LEMMA IS VALID")
        print()
        return lemma_with_proof


def check_theorem_proof_and_maybe_reprove_using_lemmas(
    context, theorem, proof, depth=0
):
    # Break out of recursion if we've reached the max depth
    if depth > MAX_THEOREM_ERROR_COUNT:
        print("MAX THEOREM ERROR COUNT REACHED. GIVING UP.")
        exit(1)

    print(f"ATTEMPTED THEOREM PROOF (LEMMAS USED: {depth})")
    print(context + "\n\n" + theorem + "\n\n" + proof)
    print()

    # Check if proof is valid and get error index if any
    annotated_proof_fragments, first_error_idx = annotate_and_fetch_error(
        context, theorem + "\n" + proof
    )

    # If there is an error, extract the proof state before the error
    # and try to prove that goal separately as a lemma
    if first_error_idx != -1:
        # Get the closest Sentence before the error
        for i in range(first_error_idx - 1, -1, -1):
            if isinstance(annotated_proof_fragments[i], Sentence):
                prev_sentence = annotated_proof_fragments[i]
                break
        # Get first non-"deprecated" error message
        for message in annotated_proof_fragments[first_error_idx].messages:
            if "deprecated" not in message.contents:
                error_message = f'Error in step "{annotated_proof_fragments[first_error_idx].contents}".\nMessage: {message.contents}.\nGoal: {prev_sentence.goals[0].conclusion}.'
                break
        print(f"ERROR MESSAGE IN THEOREM PROOF (FRAGMENT #{first_error_idx})")
        print(error_message)
        print()

        lemma = proof_state_to_lemma(
            str(depth),
            prev_sentence.goals[0].hypotheses,
            prev_sentence.goals[0].conclusion,
        )
        # String containing a space-separated list of hypothesis names, passed when applying the lemma
        lemma_args = " ".join(
            [
                " ".join(hypothesis.names)
                for hypothesis in prev_sentence.goals[0].hypotheses
            ]
        )

        lemma_with_proof = recursively_prove_lemma(context, lemma)

        # Now that we have a valid lemma, we can use it to complete the proof
        # Convert sentences to Coq code
        proof_using_lemma = ""
        for i, fragment in enumerate(annotated_proof_fragments):
            if i == first_error_idx:
                proof_using_lemma += (
                    "apply (@"
                    + lemma.split("Lemma ")[1].split(" ")[0]
                    + " "
                    + lemma_args
                    + ").\n"
                )
                still_in_same_goal = True
            elif i > first_error_idx:
                # If this line is trying to prove the same goal as the line that caused the error,
                # skip it
                if isinstance(fragment, Text) or not re.match(
                    r"^[\+\-\*]+$", fragment.contents
                ):
                    if still_in_same_goal:
                        continue
                    else:
                        proof_using_lemma += fragment.contents
                # The first time we reach a new bullet point, we know that we've reached the end
                # of what our helper lemma has taken care of
                # TODO: This isn't reliable, e.g. if the proof doesn't use bullet points
                # and simply continues to prove the next goal instead (as the proof of the following
                # goals will have been deleted).
                else:
                    proof_using_lemma += fragment.contents
                    still_in_same_goal = False
            else:
                proof_using_lemma += fragment.contents
        # Only keep proof (and discard theorem statement, etc. before it)
        proof_using_lemma = (
            "Proof.\n"
            + proof_using_lemma.split("Proof.")[-1].split("Qed.")[0]
            + "\nQed."
        )

        return check_theorem_proof_and_maybe_reprove_using_lemmas(
            context + "\n" + lemma_with_proof, theorem, proof_using_lemma, depth + 1
        )

    # Otherwise, our proof is valid, so return the entire code
    else:
        full_coq_code = context + "\n\n" + theorem + "\n\n" + proof
        return full_coq_code


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-e",
        "--example",
        help="name of example to prove",
        required=True,
        type=str,
    )
    args = parser.parse_args()

    with open(f"examples/{args.example}/context.v", "r") as f:
        context = f.read()
    with open(f"examples/{args.example}/theorem.v", "r") as f:
        theorem = f.read()

    proof = prove_using_gpt(
        context,
        theorem,
        MODEL,
    )

    with open(f"examples/{args.example}/stderr.txt", "w") as f:
        with redirect_stderr(f):
            full_coq_code = check_theorem_proof_and_maybe_reprove_using_lemmas(
                context, theorem, proof
            )

            print("PROOF IS VALID")
            with open(f"examples/{args.example}/proof.v", "w") as f:
                f.write(full_coq_code)
