from reader import read_pdf
from LLM_api import OpenAILInker
import argparse

# Get cpt codes of procedure
# Checl if we should conintue
# run the through the guideline

def reformat_medical_records(text, open_ai_linker):
    # This function will reformat the medical records to contain headers and paragraphs
    # This will make it easier for future questions
    message = 'Reformat the following to document to include headings such as medical history, if there are dates' \
              'start each line with the date. Also if there are dates of birth of people mentioned, start the line with DOB. ' \
              'Group the information using the topics given. This document' \
              'contains artifacts, if there are words that are missing letters, write out the complete words. Separate' \
              'each header with ##. The following document is: %s' % text
    medical_record = open_ai_linker.pass_message(
        sys='You are an expert at reading medical documents and extracting and reformating information. Medical records are'
            'usually divided up into the following categories: patients bio which includes personal information such '
            'as age, past medical history which sometimes includes dates, symtpoms, family medical history, assesment and'
            'plan of requested procedure and other topics. You can create heading and group the information accordingly.'
            'You can understand dates and who are family members. Todays date is 29/12/023, if any refernce to time is used,'
            'calculate the date.',
        message=message
    )
    return medical_record


def get_cpt_codes(text, open_ai_linker):
    # Here we use an LLM to ask for the CPT codes requested by the doctor
    message = 'Get CPT codes of procedures suggested in the following passage. Ignore past procedures: %s' % text
    cpt2 = open_ai_linker.pass_message(
        sys='You are an expert at reading medical documents and extracting information. Medical records contain'
            'information on a patients age, medical history, assessment and plans. Records include the CPT of previous'
            'procedures and also the requested / suggested procedure',
        message=message
    )
    print(cpt2)

def get_previous_treatment(text, open_ai_linker):
    # We find if any previous treatments were used. Dividing the medical record in its paragraphs
    #  Query each pararaph and check if treatments were used.
    topics = text.replace('\n', ' ').replace('  ', ' ').split('##')
    i = 0
    found = False
    while i < len(topics) and not found:
        paragraph = topics[i].replace('#', '')
        message = 'Did any previous conservative treatments work or improved the condition relating to the colonoscopy for this patient. Use the following document and give a YES or NO answer, if answer' \
                  'is YES write out the reason why, if NO do not mention anything: %s' %paragraph

        cpt2 = open_ai_linker.pass_message(
            sys='You are an expert at reading medical documents and extracting information. Medical records contain'
                'information on a patients age, medical history, assessment and plans. Sometimes the medical records contain'
                'information about previous treatments and sometimes not. Before giving an answer'
                'quote the exact text given to you and explain your reason. If you are unsure or unable to answer type out: NO',
            message=message
        )
        if 'YES' in cpt2:
            print(cpt2)
            return True
        else:
            i+=1
    return False


def get_answer_to_guidlines(text, guidlines, open_ai_linker):
    # Augment the guidelines so that we have steps we can follow for the LLM to execute

    message = f'Create steps to determine if the patient meets the criteria using the following guidelines: {guidlines}. ' \
              f'Do not use nested steps, and number each step. At the end of each step, state which steps to procede to, if the ' \
              f'criteria has been matched then stop.' \
              f'Separate each step with ##'
    cpt2 = open_ai_linker.pass_message(
        sys='You are an expert in medical procedures and guidlines. You are able to determine given a set of guidlines'
            'whether a patient meets a criteria for a procedure. You are able to create step by step instructions and use '
            'the outcome of those steps to determine if the patient meets the criteria. Sometimes we can jump to certain other'
            'steps based on the result of the current step. Not all points need to be met to pass the criteria, determine how'
            'many points needs to be met to pass the criteria. Do not use nested steps, number each step and determine '
            'which steps to move to.',
        message=message
    )
    message = f'Given the following steps: {cpt2}. Perform these steps on this medical record {text} and determine' \
              f'if the patient meets the criteria for the medical procedure. Before each step, repeat the step and the' \
              f'outcome of the step'

    cpt2 = open_ai_linker.pass_message(
        sys='You are an expert in reading medical guidelines and procedures and following a step by step instruction to come'
            'to a decision. As you follow the steps, read out each step and explain your answer'
            'by quoting the medical record given to you. You also know how to calculate age, the current date is 2023-12-29. If there'
            'is no mention of previous treatments, assume that they did not happen and procede.',
        message=message
    )
    print(cpt2)

def extract_data(record_fn, guidlines_fn):
    # Main function that reads the medical record and guidlines and follows the pipeline
    record = read_pdf(record_fn)
    with open(guidlines_fn, 'r') as file:
        guidlines = file.read()
    open_ai_linker = OpenAILInker()
    medical_record = reformat_medical_records(record, open_ai_linker)
    get_cpt_codes(medical_record, open_ai_linker)
    prev_treat = get_previous_treatment(medical_record, open_ai_linker)
    if prev_treat:
        print('We have found previous treatments that have worked. As a result the patient is not eligible for the treament')
    else:
        print("No previous conservative treatments have worked. Now we will check the guidelines.")
        get_answer_to_guidlines(medical_record, guidlines, open_ai_linker)

if __name__ == "__main__":
    # Create the parser
    parser = argparse.ArgumentParser(description='Process two files.')

    # Add arguments
    parser.add_argument('-m',  '--medical_record', required=True, help='The first filename is for medical record')
    parser.add_argument('-g', '--guidelines', default='data/guidelines.txt', help='The second filename is to pass the guidelines')

    # Parse arguments
    args = parser.parse_args()

    # Pass the arguments to the function
    extract_data(args.medical_record, args.guidelines)
