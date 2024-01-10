import os
import re
import fitz
from oauth2client.service_account import ServiceAccountCredentials
from langchain.text_splitter import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
import gspread
import regex
import json


# Regex pattern to remove page numbers from parsed PDF text
page_number_pattern = r'^[\n\s]*\d+[\n\s]*(?!.)'
# Regex pattern to remove extra spaces from parsed PDF text
spaces_regex = r"(?<!\n\s)\n(?!\n| \n)"


# Function to get full sections data from the google sheets (Karnataka and Central)
def get_data_from_google_sheets(required_sheet_name: str):
    credentials_path = os.environ["GOOGLE_APPLICATION_CREDENTIALS"]
    scope = ['https://spreadsheets.google.com/feeds', 'https://www.googleapis.com/auth/drive']
    creds = ServiceAccountCredentials.from_json_keyfile_name(credentials_path, scope)
    client = gspread.authorize(creds)
    sheet_url = os.environ["GOOGLE_SHEET_URL"]
    sheet = client.open_by_url(sheet_url)

    sheet_names = sheet.worksheets()
    for worksheet in sheet_names:
        if worksheet.title == required_sheet_name:
            data = worksheet.get_all_values()
            keys = data[0]
            values = data[1:]
            section_names_data = [dict(zip(keys, row)) for row in values]
            return section_names_data


# Fuction to get only the section names from the extracted google sheets' sections data
def get_data_from_sheet_data(filename: str, section_names_data: list):
    # Regex pattern to find section names which starts with digits
    pattern = r'(\n*\d+-?\[?[A-Z]{0,3}\..*)'
    for section in section_names_data:
        if section['Filename'] == filename:
            matches = re.findall(pattern, section['Section'])
            section_list = [match.replace("\n", "") for match in matches]
            return section_list


# Function to find the substring with fuzzy search
def fuzzy_substring_search(minor: str, major: str, errs: int = 10):
    errs_ = 0
    s = regex.search(f"({minor}){{e<={errs_}}}", major)
    while s is None and errs_ <= errs:
        errs_ += 1
        s = regex.search(f"({minor}){{e<={errs_}}}", major)
    return s


# Function to find matching strings with fuzzy search
def find_string_with_fuzzy(A_string: str, B_string: str,
                           large_string: str, is_first_section: bool, is_last_section: bool):
    match_a = re.search(A_string, large_string, re.IGNORECASE)
    if match_a is None:
        match_a = fuzzy_substring_search(A_string, large_string)
    if is_last_section is False:
        match_b = re.search(B_string, large_string, re.IGNORECASE)
        if match_b is None:
            match_b = fuzzy_substring_search(B_string, large_string)
    else:
        match_b = re.search(B_string, large_string)
        if match_b is None:
            return large_string[match_a.start():]
        else:
            return [large_string[match_a.start():match_b.start()], large_string[match_b.start():]]

    if is_first_section:
        return [large_string[0: match_a.start()], large_string[match_a.start():match_b.start()]]

    result = large_string[match_a.start():match_b.start()]
    return result


# Function to split the parsed PDF text section wise for the Karnataka Acts
def get_karnataka_section_splits(root_dir: str, section_names_data: list):
    section_dict = {}
    file_names = os.listdir(root_dir)
    # Regex pattern to find the table of contents' section data in the first page of parsed PDF text
    pattern = r'statement(?:s)?\s*of\s*object(?:s)?\s*and\s*reason(?:s)?'
    for filename in file_names:
        # Skipping the files which can not be split into sections by this logic
        if filename not in ["THE KARNATAKA STAMP ACT, 1957-ENG.pdf", "21 of 1964 (E).pdf", "32 of 1963 (E).pdf",
                            "THE KARNATAKA SCHEDULED CASTES SUB-ALLOCATION AND TRIBAL...ETC 2013-ENG.pdf", "THE KARNATAKA SALES TAX ACT, 1957-ENG.pdf", "11of1959(E).pdf",
                            "A1860-45 (E).pdf", "THE KARNATAKA LEGISLATURE SALARIES, PENSIONS AND ALLOWANCES ACT, 1956 -ENG.pdf",
                            "THE KARNATAKA HOUSING BOARD ACT, 1962-ENG.pdf", "THE KARNATAKA LAND REVENUE ACT, 1964-ENG.pdf", "A1908-05 (E).pdf", "27 of 1966 (E) emblem.pdf",
                            "19 of 1979 Rules (E) Debt.pdf", "17 of 2019 Rules (E).pdf", "23 of 2013 Rules (E).pdf", "A1974-02 (E).pdf", "COI (E).pdf"]:
            print("\nFilename:", filename)
            doc = fitz.open(os.path.join(root_dir, filename))
            content = "\n"
            content_list = []
            # Iterating through all the pages of parsed PDF text
            for i in range(len(doc)):
                flag = False
                page = doc[i]
                text = page.get_text("text", textpage=None, sort=False)
                text = re.sub(page_number_pattern, '', text)
                if i == 0:
                    matches = list(re.finditer(pattern, text, re.IGNORECASE))
                    # Checking if the table of contents is fully present in the first page of parsed PDF text
                    if len(matches) == 2:
                        split_text = re.split(pattern, text, flags=re.IGNORECASE)
                        new_text = split_text.pop(-1)
                        text = " ".join(split_text)
                        # Flag to stop the table of contents sections' data from getting added to the text content
                        flag = True

                if flag is False:
                    split_text = re.split(pattern, text, flags=re.IGNORECASE)
                    # Check if the text is split into two parts. If yes, then the first part is the section names' table of content
                    if len(split_text) == 2:
                        # The last part of the split_text is the rest of the act information
                        new_text = split_text.pop(-1)
                        text = " ".join(split_text)
                        # Flag to stop the table of contents sections' data from getting added to the text content
                        flag = True

                content += text
                if flag is True:
                    # Append the table of contents sections' data to the content_list
                    content_list.append(content)
                    # Make the content from the last page as the split_text
                    content = new_text

            # Append rest of the full act information from the content to the content_list
            content_list.append(content)
            # Checking if the content_list has more than one element
            if len(content_list) > 1:
                # The last element of the content_list is the needed rest of the act information
                text = content_list.pop(-1)
                text = re.sub(spaces_regex, '', text)
                sections_list = [" ".join(content_list)]
            else:
                # The only element of the content_list is the needed act information
                text = re.sub(spaces_regex, '', content_list[0])
                sections_list = []

            # Extracting only the section names from the google sheets' sections data
            data = get_data_from_sheet_data(filename, section_names_data)
            # Iterating through all the section names
            for i in range(len(data)):
                exception_encounter = False
                if i == 0:
                    is_first_section = True
                else:
                    is_first_section = False

                # Taking the previous section name as the A_section_name
                A_section_name = data[i].split(" ")
                A_section_string = r"\s*".join(A_section_name)
                if i == len(data)-1:
                    is_last_section = True
                    B_section_string = "SCHEDULE"
                else:
                    is_last_section = False
                    # Taking the next section name as the B_section_name
                    B_section_name = data[i+1].split(" ")
                    B_section_string = r"\s*".join(B_section_name)
                try:
                    # Calling the find_string_with_fuzzy function to find the section data between the given A_section_name and B_section_name
                    extracted_text = find_string_with_fuzzy(A_section_string.strip(" "),
                                                            B_section_string.strip(" "),
                                                            text, is_first_section, is_last_section)
                    # Checking if the extracted_text is a list or not
                    if isinstance(extracted_text, list):
                        sections_list += extracted_text
                    else:
                        sections_list.append(extracted_text)
                except Exception:
                    # If any exception occurs, then the exception_encounter flag is set to True
                    exception_encounter = True
                    print(filename, "is problematic and not fully done")
                    print("Remaining Uncompleted Sections:", len(data)-i)
                    # Break the loop
                    break

            print("Total Completed Sections:", len(sections_list))
            # If exception_encounter is False, then add the sections_list to the section_dict with filename as key
            if exception_encounter is False:
                section_dict[filename] = sections_list

    # Finally dump the section_dict for all Karnataka acts to a json file
    with open("karnataka_section_splits.json", 'w') as json_file:
        json.dump(section_dict, json_file)


# Function to split the parsed PDF text section wise for the Central Acts
def get_central_section_splits(root_dir: str, section_names_data: list):
    section_dict = {}
    # Regex pattern to find the table of contents sections' data in the first page of parsed PDF text
    pattern = r"ARR\D{1,3}EMENT OF SECT\D{0,2}NS{0,1}"
    file_names = os.listdir(root_dir)
    # Iterating through all the files in the central root_dir
    for filename in file_names:
        print("\nFilename:", filename)
        data_list = []
        doc = fitz.open(os.path.join(root_dir, filename))
        # Extracting the first page of parsed PDF text
        first_page = doc[0].get_text("text", textpage=None, sort=False)
        # Checking if the pattern is present in the first page of parsed PDF text
        if re.search(pattern, first_page):
            # Splitting the first page of parsed PDF text into title and sections
            title, sections = re.split(pattern, first_page)
            # Removing the page numbers and other outliers from the title
            title = re.sub(r'(?:^|\s|\n)\d{1,2}(?=\s|\n|$)', '', title)
            title = title.replace("_", "").strip()
            title = title.replace("\n", "")
            sections = title + sections
            sections = sections.replace("SECTIONS", "").replace("_", "").strip()
            # Iterating through rest of the pages of parsed PDF text
            for i in range(1, len(doc)):
                page = doc[i]
                text = page.get_text("text", textpage=None, sort=False)
                text = re.sub(page_number_pattern, '', text)
                # Checking if title is present in the parsed text
                if title in text:
                    # Appending the so far collected sections (table of contents data) to the data_list
                    data_list.append(sections)
                    # Making the sections as empty string for rest of the act information
                    sections = ""
                sections += text
            # Appending the rest of the act information to the data_list
            data_list.append(sections)
        else:
            print("CANNOT FIND TITLES INSIDE THIS FILE: ", filename)
            continue

        # Checking if the data_list has more than one element
        if len(data_list) > 1:
            # The last element of the data_list is the needed rest of the act information
            text = data_list.pop(-1)
            sections_list = [" ".join(data_list)]
        else:
            # The only element of the data_list is the needed act information
            text = data_list[0]
            sections_list = []

        # Removing extra spaces from the text
        text = re.sub(spaces_regex, '', text)
        # Extracting only the section names from the google sheets' sections data
        data = get_data_from_sheet_data(filename, section_names_data)
        for i in range(len(data)):
            if i == 0:
                is_first_section = True
            else:
                is_first_section = False

            # Taking the previous section name as the A_section_name
            A_section_name = data[i].split()
            A_section_string = r"\s*".join(A_section_name)
            if i == len(data)-1:
                is_last_section = True
                B_section_string = "THE SCHEDULE"
            else:
                is_last_section = False
                # Taking the next section name as the B_section_name
                B_section_name = data[i+1].split()
                B_section_string = r"\s*".join(B_section_name)
            try:
                # Calling the find_string_with_fuzzy function to find the section data between the given A_section_name and B_section_name
                extracted_text = find_string_with_fuzzy(A_section_string.strip(" "),
                                                        B_section_string.strip(" "),
                                                        text, is_first_section, is_last_section)
                # Checking if the extracted_text is a list or not
                if isinstance(extracted_text, list):
                    sections_list += extracted_text
                else:
                    sections_list.append(extracted_text)
            except Exception:
                print(filename, "is problematic and not fully done")
                print("Remaining Uncompleted Sections:", len(data)-i)
                # Break the loop
                break

        print("Total Completed Sections:", len(sections_list))
        section_dict[filename] = sections_list

    # Finally dump the section_dict for all Central acts to a json file
    with open("central_section_splits.json", 'w') as json_file:
        json.dump(section_dict, json_file)


# Function to chunk the sections which are more than 4000 characters and also group the sections which are less than 4000 characters
def sections_chunking():
    # Reading the Karnataka section splits json file
    with open("karnataka_section_splits.json", 'r') as file:
        json_data = file.read()

    sections_dict = json.loads(json_data)
    # Regex pattern to find section numbers paragraph from parsed PDF text
    digit_pattern = r'^(\n*\d+-?\[?[A-Z]{0,3}\..*)'
    # Regex pattern to find section numbers from parsed PDF text
    section_number_pattern = r'^(\n*\d+-?\[?[A-Z]{0,3}\.)'
    # The following sentences are added as a precursor to the chunks
    precursor_sentence_one = "The following contents are part of the {}"
    precursor_sentence_two = "The following sections are part of the {}"
    precursor_sentence_three = "The following section is part of the {}"
    precursor_sentence_four = "The following contents are continuation of section {} of the {}"

    # Initializing the RecursiveCharacterTextSplitter which will split the chunks based on the given separators
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=4000,
        chunk_overlap=0,
        separators=["\n \n", "\n\n", "\n", ".", " "]
    )

    # Reading the google sheets' metadata for Karnataka acts
    meta_data_list = get_data_from_google_sheets("All Acts")
    result_dict = {}
    for key, value in sections_dict.items():
        print("Filename:", key)
        for meta_data in meta_data_list:
            # Getting the title of the act from the google sheets' metadata for the given filename
            if meta_data['File Name'] == key:
                title = meta_data['Doc title']

        # Store the sections data list in section_doc
        section_doc = value
        # Initializing the new_chunks list with the first element of section_doc as it is the title and table of contents of the act
        new_chunks = [section_doc[0]]
        i = 1
        # Iterating through the rest of the sections data list
        while i < len(section_doc):
            section = section_doc[i]
            # Checking if the section data starts with a digit
            matches = re.findall(digit_pattern, section)
            # Checking if there is only one section paragraph in the section data
            if len(matches) == 1:
                # Checking if the section data is less than 4000 characters
                if len(section) < 4000:
                    flag = False
                    # Adding the precursor sentence to the section data as it is a section paragraph
                    section = precursor_sentence_two.format(title) + "\n\n\nSection " + section
                    # Checking if the section data is the last element of the section_doc
                    if i == len(section_doc)-1:
                        new_chunks.append(section)
                        # Break the loop
                        break

                    new_section = section
                    j = 1
                    # Iterating through the rest of the section_doc to find the sections which when combined with existing section, are less than 4000 characters
                    while True:
                        # Checking if the combined index is greater than or equal to the length of section_doc
                        if i+j >= len(section_doc):
                            flag = True
                            break
                        # Adding 'Section' word to the front of the section data
                        new_section += "\n\n\nSection " + section_doc[i+j]
                        # Checking if the combined section data is greater than 4000 characters
                        if len(new_section) > 4000:
                            # Removing the last section data from the new_section as it is greater than 4000 characters
                            new_section = new_section.replace("\n\n\nSection " + section_doc[i+j], "")
                            flag = True
                            j -= 1
                            break
                        j += 1
                    # Checking if the flag is True
                    if flag is True:
                        # Appending the new_section to the new_chunks list
                        new_chunks.append(new_section)
                        i += j
                    else:
                        # Appending the section to the new_chunks list
                        new_chunks.append(section)
                else:
                    # Getting the section paragraph data from the section data
                    section_number_match = re.search(section_number_pattern, section)
                    section_number = section[section_number_match.start():section_number_match.end()]
                    # Splitting the section data into chunks of 4000 characters
                    section_splits = text_splitter.split_text(section)
                    sections_list = []
                    for k in range(len(section_splits)):
                        if k == 0:
                            # Adding the precursor sentence and 'Section' word to the first chunk of the section data
                            section_split = precursor_sentence_three.format(title) + "\n\n\nSection " + section_splits[k]
                        else:
                            # Adding the precursor sentence to the rest of the chunks of the section data
                            section_split = precursor_sentence_four.format(section_number, title) + "\n\n\n" + section_splits[k]
                        sections_list.append(section_split)
                    new_chunks += sections_list
            else:
                # Checking if the section data is greater than 4000 characters
                if len(section) > 4000:
                    # Splitting the section data into chunks of 4000 characters
                    section_splits = text_splitter.split_text(section)
                    section_splits = [precursor_sentence_one.format(title) + "\n\n\n" + section_split for section_split in section_splits]
                    new_chunks += section_splits
                else:
                    section = precursor_sentence_one.format(title) + "\n\n\n" + section
                    new_chunks.append(section)
            i += 1
        # Adding the new_chunks list to the result_dict with filename as key
        result_dict[key] = new_chunks

    # Finally dump the result_dict for all Karnataka acts chunks to a json file
    with open("karnataka_docs_chunks.json", 'w') as json_file:
        json.dump(result_dict, json_file)


if __name__ == "__main__":
    load_dotenv()
    root_dir = os.environ["ROOT_DIR"]
    section_names_data = get_data_from_google_sheets("Sheet Name")
    # get_karnataka_section_splits(root_dir, section_names_data)
    # get_central_section_splits(root_dir, section_names_data)
    # sections_chunking()
