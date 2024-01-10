import re
import uuid
from typing import List, Tuple

from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from pathlib import Path

class DocumentParser():
    # TAGS_REGX  = r'<\?%\s*.*?%\s*>'
    # TAGS_REGX  = r'<\?%\s*(.*?)%\s*>'
    TAGS_REGX = r'<\?%\s*type=(.*?),\s*object_id=(.*?)\s*%\s*>'
    TAGS_MAX_LEN = 50

    @staticmethod
    def parse_pdf(file:Path)->str:
        """Parse the PDF file and return the text content."""
        reader = PdfReader(file)
        raw_text = ""
        for i, page in enumerate(reader.pages):
            text = page.extract_text()
            if text:
                raw_text += text
        return raw_text
    
    @staticmethod
    def parse_raw_texts(raw_texts:str)->Tuple[List[str],List[str]]:
        """
        Accepts the entire text in string format with <%? %> tags included
        and parses it.
        """
        output_text = []
        output_object_ids = []

        # Split the text into chunks
        texts = DocumentParser.split_texts(raw_texts)

        for text in texts:
            # Extract tags
            broken_tags = DocumentParser.check_broken_tags(text)
            text = DocumentParser.fix_broken_tags(text, raw_texts, broken_tags)
            text = DocumentParser.trim_broken_text(text)
          
            tags = DocumentParser.extract_tags(text)
            if len(tags) == 0:
                output_text.append(text)
                output_object_ids.append("")
            else:
                # Replace tags with empty string
                text = re.sub(DocumentParser.TAGS_REGX, "", text)
                output_text.append(text)

                # Combine tags to be a string 
                # ChromaDB metadata does not accept list
                tag_string = ""
                for tag in tags:
                    tag_string += f"{tag[1].strip()},"
                tag_string = tag_string[:-1]
                output_object_ids.append(tag_string)
  
        return output_text, output_object_ids


    @staticmethod
    def split_texts(raw_text: str)->List[str]:
        """Split the text into chunks of text."""
        text_splitter = CharacterTextSplitter(
            separator = "",
            chunk_size = 750, # need to research on the right value to use for chunk_size and chunk_overlap
            chunk_overlap = 50,
            length_function = len,
        )
        
        texts = text_splitter.split_text(raw_text)
     
        return texts
    
    @staticmethod
    def extract_tags(text:str)->List[Tuple[str,str]]:
        """
        Given an excerpt, extract out the selected tags
        """
        output = []
        matches = re.finditer(DocumentParser.TAGS_REGX, text)
        for match in matches:
            type, object_id = match.groups()
            assert len(type) != 0 and len(object_id) != 0, f"Invalid tag format, {type, object_id}!" 
            output.append((type, object_id))

        return output
    
    @staticmethod
    def check_broken_tags(text:str, tag_max_length:int=TAGS_MAX_LEN) -> List[bool]:
        """
        Given a text, scans to see if there are broken text.
        Looks out for <%? and %>
        If we are using UUID, the expected length of the text will be 36 character. 

        Returns:
            [bool,bool]: True indicates the tag is broken, False indicates the tag is unbroken,
                        First element refers to start, second element refers to end.
        """
        
        output = [False, False]

        text_start = text[:tag_max_length]
        text_end = text[-tag_max_length:]

        first_index = text_start.find(">")
        #Trouble shoot or   "> here, another tag here"
        if "<?%" in text_start and "%>" in text_start:
            output[0] = False
        elif first_index != -1: 
            # Means there might be broken tag at the start
            if first_index -1 < 0 : #This means the sentence starts with ">", cannot identify if it is broken. Will just say its true
                output[0] = True
            elif text_start[first_index-1] == "%": # %>
                if text_start.find("<?%") != -1:
                    # This means the tag is complete
                    output[0] = False
                else:
                    output[0] = True
            else: # This means the start starts with " > ..." which is not an end tag. 
                output[0] = False
        
        # Find all occurences of "<" in end
        last_index = text_end.rfind("<")
        if last_index != -1:
            if last_index == len(text_end) - 1: # This means the last character is "<", cannot identify if it is broken. Will just say its true
                output[1] = True
            # Means there might be broken tag at the end
            # Find index of "<"
            # Check if the next character is "%?"
            elif last_index + 1 < len(text_end) and text_end[last_index+1] == "?": # <? 
                if last_index + 2 < len(text_end) and text_end[last_index+2] == "%": # <?%
                    # Check if from the last_index to the end, is there a "%>"
                    if "%>" in text_end[last_index:]:
                        # This means the tag is unbroken
                        output[1] = False
                    else:
                        output[1] = True
            else: # This means that "< " which is not a closing tag
                output[1] = False
                
        return output
    
    @staticmethod
    def fix_broken_tags(
        excerpt:str,
        full_text:str,
        indicator:List[bool],
        tag_max_length:int=TAGS_MAX_LEN
        )-> str:
        # Start is broken
        index = full_text.find(excerpt) # Find the index of this in full step
        if indicator[0]: 
            first_index = excerpt.find(">") # Assumption here is that everything on the left of this is the broken tag
            # Add in the tags max length in front
            if index - tag_max_length > 0: # 
                excerpt = full_text[index - tag_max_length: first_index+1] + excerpt[first_index:]
            else:
                excerpt = full_text[:first_index+1] + excerpt[first_index:]
        # End is broken
        if indicator[1]:
            last_index = excerpt.rfind("<")
            index += len(excerpt) # Get the end index in full text
            if index + tag_max_length < len(full_text):
                excerpt = excerpt[:last_index+1] + full_text[index+len(excerpt): index + len(excerpt) + tag_max_length]
            else:
                excerpt = excerpt[:last_index+1] + full_text[index+len(excerpt): index:]
        # Sometimes when you extend the excerpt, it might have another broken part. 
        # In order to prevent this, we will trim 

        return excerpt
    
    @staticmethod
    def trim_broken_text(text:str):
        opening_tag = text.find("<?%")
        ending_tag = text.find("%>")
        if ending_tag < opening_tag: # Means "%> <?%...%>" 
            text = text[ending_tag+2:]
        elif opening_tag == -1 and ending_tag != -1: # Means "%>...%>"
            text = text[ending_tag+2:]

        opening_tag = text.rfind("<?%")
        ending_tag = text.rfind("%>")
        if opening_tag > ending_tag:
            text = text[:opening_tag]
        return text

if __name__ == "__main__":
    # text = f"of life's cruelties. <?% type=image,object_id={str(uuid.uuid4())[:-6]} %> <?% type=image,object_id={str(uuid.uuid4())[:-6]} %> Were they necessary for growth, like the way heat transforms sugar into the sweetest candies? Or were they simply random, like a child's choice of which jelly bean to eat next? <?% type=text,object_id={str(uuid.uuid4())[:-6]} %> In that moment, as it melted away in the warmth of a child's mouth, Jello found a bitte"
    # texts, ids = DocumentParser.parse_raw_texts(text)
    # print(texts)
    # print(ids)
    
    # Asserts broken tag is working
    text1 = "This is a sample text without a broken tag: <?% type=broken,object_id=123 %>. Lorem ipsum dolor sit amet."
    text2 = "<<<<?% type=extra<,object_id=456 %>>. Nulla < >euismod massa vel lectus."
    text3 = "This is a sample text with a broken tag: <?% type=broken,object_id=123"
    text4 = "ect_id=123 %>This is a sample text with a broken tag:<?% type=broken,object_id=123"

    assert DocumentParser.check_broken_tags(text1) == [False, False], f"Expected [False, False], got {DocumentParser.check_broken_tags(text1)}!"
    assert DocumentParser.check_broken_tags(text2) == [False, False], f"Expected [False, False], got {DocumentParser.check_broken_tags(text2)}!"
    assert DocumentParser.check_broken_tags(text3) == [False, True], f"Expected [False, True], got {DocumentParser.check_broken_tags(text3)}!"
    assert DocumentParser.check_broken_tags(text4) == [True, True], f"Expected [True, True], got {DocumentParser.check_broken_tags(text4)}!"
    
    # Test fix_broken_tag
    fulltext = "This is the full text with the tags <?% type=extra,object_id=456 %> here, another tag here <?% type=extra,object_id=456 %>, one more tag here!"
    excerpt1 = "pe=extra,object_id=456 %> here, another tag here <?% type=extra,object_id=45"
    excerpt2 = "This is the full text with the tags <"
    excerpt3 = "> here, another tag here"

    fixed_excerpt1 = DocumentParser.trim_broken_text(DocumentParser.fix_broken_tags(excerpt1, fulltext, DocumentParser.check_broken_tags(excerpt1)))
    fixed_excerpt2 = DocumentParser.trim_broken_text(DocumentParser.fix_broken_tags(excerpt2, fulltext, DocumentParser.check_broken_tags(excerpt2)))
    fixed_excerpt3 = DocumentParser.trim_broken_text(DocumentParser.fix_broken_tags(excerpt3, fulltext, DocumentParser.check_broken_tags(excerpt3)))

    print(fixed_excerpt1)
    print(fixed_excerpt2)
    print(fixed_excerpt3)
   

