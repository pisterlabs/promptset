# Here, the task-specific AI agents for text analysis, 
# image analysis, and keyword/tag analysis are specified. 
# The general logic for API communication with OpenAI is 
# established in the parent classes AIAgent and AIAgent_OpenAI
# (see AIAgents.py).

from autoPDFtagger.AIAgents import AIAgent_OpenAI
from autoPDFtagger.AIAgents import OpenAI_model_pricelist
import logging
from autoPDFtagger.config import config
api_key = config['OPENAI-API']['API-Key']
LANGUAGE = config['DEFAULT']['language']

from autoPDFtagger.PDFDocument import PDFDocument
import json
import pprint
import re
import copy
import tiktoken

# IMAGE-Analysis
class AIAgent_OpenAI_pdf_image_analysis(AIAgent_OpenAI):
    def __init__(self):
        system_message = f"""
        You are a helpful assistant analyzing images inside of documents. 
        Based on the shown images, provide the following information:\n
        1. Creation date of the document.\n
        2. A short title of 3-4 words.\n
        3. A short summary of 3-4 sentences.\n
        4. Creator/Issuer\n
        5. Suitable keywords/tags related to the content.\n
        6. Rate the importance of the document on a scale from 0 (unimportant) to 10 (vital).\n
        7. Rate your confidence for each of the above points on a scale from 0 (no information) 
        over 5 (possibly right, but only few hints) to 10 (very sure). 
        You always answer in {LANGUAGE} language.  For gathering information, 
        you use the given filename, pathname and ocr-analyzed text. You always 
        answer in a specified JSON-Format which is given in the question. """
        
        # calling parent class constructor
        super().__init__(model="gpt-4-vision-preview", system_message=system_message)

        self.response_format="json_object"
    
    # Main function of this class: Try to extract relevant metadata
    # out of a PDFDocument (pdf_document) by analyzing their images
    # and return result as a json-string
    def analyze_images(self, pdf_document: PDFDocument):
        pdf_document.analyze_document_images()

        # Prevent modifying the original document
        working_doc = copy.deepcopy(pdf_document)
        
        # For the general requirement of this function, 
        # I have different approaches, which vary depending 
        # on the structure of the document. In the case of a 
        # scanned document, at least the first page should be 
        # completely analyzed, as it is where most of the relevant 
        # information (title, summary, date) can be expected. 
        # Subsequent pages should only be examined for cost reasons 
        # if the collected information is insufficient. This approach 
        # is implemented in the function process_images_by_page.

        # In cases where the document is not a scanned one, but an 
        # originally digital document, images contained within it 
        # (of a certain minimum size) can also hold valuable information. 
        # These images can be analyzed in blocks (currently 3 at a time) 
        # by GPT. We start with the largest images. At least one block 
        # will be analyzed. Additional blocks only if the quality of the 
        # metadata is not yet sufficient (similar to process_images_by_pages). 
        # This approach is specified by the function process_images_by_size.

        # No scanned document
        if pdf_document.image_coverage < 100: 
            logging.info("Analyzing smaller Images")
            # Wir sortieren alle Bilder mit einer bestimmten Mindestgröße und 
            # fangen von oben an, diese in 5er Gruppen in einzelnen Anfragen 
            # an GPT-Vision zu schicken, so lange, bis entweder alle Bilder
            # analysiert sind oder has_sufficient_information true ergibt
            working_doc = self.process_images_by_size(working_doc)

        # Scanned document
        if pdf_document.image_coverage >= 100: 
            # Hier alle Seiten als Bild GTP-Vision vorlegen, welche weniger als 100 Wörter enthalten
            logging.info("Recognizing scanned document")
            working_doc = self.process_images_by_page(working_doc)

        return working_doc.to_api_json()

    # A generic function to ask GPT to analyze a list of Images (list_imgaes_base64)
    # in context of information of a PDFDocument (document)
    # The decision regarding the selection of images and their 
    # extraction from the document is made separately, therefore 
    # these must be passed as additional parameters.
    def send_image_request(self, document: PDFDocument, list_images_base64):
        logging.info("Asking GPT-Vision for analysis of " + str(len(list_images_base64)) + " Images found in " + document.get_absolute_path())
        
        user_message = (
            "Analyze following Images which are found in a document. "
            "Please extend the existing information by keeping their JSON-Format: "
            + document.to_api_json() + 
            " Try to imagine as many valuable keywords and categories as possible. "
            "Imagine additional keywords thinking of a wider context and possible categories in an archive system. "
            "Answer in JSON-Format corresponding to given input."
        )

        message_content = [
        {
            "type": "text",
            "text": user_message
        }]

        # Add individual images to the message
        for base64_image in list_images_base64:
            image_content = {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{base64_image}"
                }
            }
            message_content.append(image_content)

        self.add_message(message_content)


        try:
            response = super().send_request(temperature=0.2, response_format = None, )
            return response
        except Exception as e:
            logging.error("API-Call failed")
            logging.error(e)
            return None

    def process_images_by_size(self, pdf_document: PDFDocument):
        # Create a list of all images from each page
        all_images = [image for page in pdf_document.images for image in page]
        # Sort images by pixel count (width x height)
        sorted_images = sorted(all_images, key=lambda img: img["original_width"] * img["original_height"], reverse=True)

        # Filter out images smaller than 90000 pixels (e.g., less than 300x300)
        relevant_images = [img for img in sorted_images if img["original_width"] * img["original_height"] >= 90000]

        # Process images in groups of 3
        for i in range(0, len(relevant_images), 3):
            group = relevant_images[i:i + 3]
            base64_group = []

            for image in group:
                base64_group.append(pdf_document.get_png_image_base64_by_xref(image['xref']))

            # Call ai_analyze_images with the group of images
            response = self.send_image_request(pdf_document, base64_group)
            try:
                pdf_document.set_from_json(response)
            except Exception as e:
                logging.error("API-Call for image analysis failed")
                logging.error(e)

            if pdf_document.has_sufficient_information():
                logging.info("Document information sufficient. Proceeding with next document.")
                return pdf_document
            else:
                logging.info("Still lacking information, looking for more images")
        logging.info("No more images found.")
        return pdf_document
            
    def process_images_by_page(self, pdf_document: PDFDocument):
        for page in pdf_document.pages:
            logging.debug(f"Checking Page {page['page_number']} looking for largest image")
            # Skip page if no images are present
            if 'max_img_xref' not in page or not page['max_img_xref']:
                logging.debug("Page not analyzed: (no images)")
                continue

            # Get the largest image of the site (assuming it to be the scan-image)
            image_base64 = pdf_document.get_png_image_base64_by_xref(page['max_img_xref'])

            # Send it to GPT
            logging.info("Asking AI for analyzing scanned page")
            response = self.send_image_request(pdf_document, [image_base64])
          
            try:
                pdf_document.set_from_json(response)
            except Exception as e:
                logging.error("API-Call for image analysis failed")
                logging.error(e)

            # Only proceed if the information about the document is still insufficient
            if pdf_document.has_sufficient_information():
                logging.info("Document information sufficient, proceeding.")
                return pdf_document
            else:
                logging.info("Still lacking information, looking for more pages")
        logging.info("No more pages available.")
        return pdf_document

# TEXT-Analysis
class AIAgent_OpenAI_pdf_text_analysis(AIAgent_OpenAI):
    def __init__(self):
        system_message = (
            "You are a helpful assistant analyzing OCR outputs. It's important "
            "to remember that these outputs may represent only a part of the document. "
        
            "Provide the following information:\n"
            "1. Creation date of the document.\n"
            "2. A short title of 3-4 words.\n"
            "3. A meaningful summary of 3-4 sentences.\n"
            "4. Creator/Issuer\n"
            "5. Suitable keywords/tags related to the content.\n"
            "6. Rate the importance of the document on a scale from 0 (unimportant) to "
            "10 (vital).\n"
            "7. Rate your confidence for each of the above points on a scale "
            "from 0 (no information, text not readable) over 5 (possibly right, but only "
            "few hints about the content of the whole document) to 10 (very sure). "
            "You always answer in {LANGUAGE} language. For gathering information, "
            "you use the given filename, pathname and OCR-analyzed text. "
            "If you are seeing a blank document, your title-confidence is alway 0."
            "You always answer in a specified JSON-Format like in this example:\n"
            "{\n"
            "    'summary': '[summary]',\n"
            "    'summary_confidence': [number],\n"
            "    'title': '[title]',\n"
            "    'title_confidence': [number],\n"
            "    'creation_date': '[Date YY-mm-dd]',\n"
            "    'creation_date_confidence': [number],\n"
            "    'creator': '[creator name]',\n"
            "    'creator_confidence': [number],\n"
            "    'tags': ['[tag 1]', '[tag 2]', ...],\n"
            "    'tags_confidence': [[confidence tag 1], [confidence tag 2]],\n"
            "    'importance': [number],\n"
            "    'importance_confidence': [number]\n"
            "}"
        )

        # Parent constructor
        super().__init__(model="gpt-4-1106-preview", system_message=system_message)

        self.response_format="json_object"
    
    # Main working function to get info about a PDFDocument by
    # sending a GPT-API-Request
    def analyze_text(self, pdf_document: PDFDocument):
       
        # Step 1: Analyze the number of potentially meaningful words
        # to decide which model to use
        # GPT-3.5 is good enough for long texts and much cheaper. 
        # Especially in shorter texts, GPT-4 gives much more high-quality answers
        word_count = len([word for word in re.split(r'\W+', pdf_document.get_pdf_text()) if len(word) >= 3])
        model_choice = "gpt-3.5-turbo-1106" if word_count > 100 else "gpt-4-1106-preview"
        #model_choice = "gpt-4-1106-preview" # for test purposes

        logging.debug("Opting for " + model_choice)
        self.set_model(model_choice)
        
        message = ("Analyze following OCR-Output. Try to imagine as many valuable keywords and categories as possible. "
            "Imagine additional keywords thinking of a wider context and possible categories in an archive system. "
            f"Use {LANGUAGE} Language. Answer in the given pattern (JSON): "
            + pdf_document.get_short_description()
        )
        
        # in case of very long text, we have to shorten it depending on 
        # the specific token-limit of the actual model

        # Optimize request data
        # Convert message-list to str
        request_test_str = pprint.pformat(self.messages)
        # estimate token-number for request
        num_tokens = num_tokens_from_string(request_test_str)
        # estimate 500 Tokens for answer
        tokens_required = num_tokens + 500
        
        # max tokens of the actual model stored in price list table
        diff_to_max =  tokens_required - OpenAI_model_pricelist[self.model][2]
        if diff_to_max > 0: # message too long 
            # we need to shorten it
            # estimating 3 characters per Token
            message = message[:-(diff_to_max*3)] 
            logging.info("PDF-Text needs to be shortened due to token_limit by " + str(diff_to_max*3) + " characters.")

        self.add_message(message, role="user")
    
        primary_response = super().send_request(temperature=0.7, response_format = self.response_format)
        return primary_response
 
        # At this point, a secondary request could be implemented to 
        # optimize the result (draft below)

        # self.add_message(primary_response, role="assistant")

        # message2 = """
        #    Critically review and correct your answer. Ensure that the confidence level accurately 
        #    reflects how well the content of the original document can be estimated in terms of accuracy 
        #    and completeness based on the available text excerpts. Additionally, think of more 
        #    keywords/tags that could improve the document's findability. Be creative, but only 
        #    give useful suggestions! Respond in german language. Respond as usual in the specified JSON 
        #    format."""
        #self.add_message(message2, "user")

        #try:
        #    secondary_response = super().send_request(temperature=0.7, response_format = self.response_format)
        #except Exception as e:
        #    logging.error(e)
        #    return None
        
        #return secondary_response


# TAG/KEYWORD-Analysis
class AIAgent_OpenAI_pdf_tag_analysis(AIAgent_OpenAI):
    def __init__(self):
        
        system_message = """You are a helpful assistant organizing tags. Please perform the following tasks:
        1. Correct any spelling errors in tags.
        2. Remove meaningless and irrelevant tags like 'Date', '23', 'Persons', 'Positions'
        3. Try to maintain a good specificity of the tags. Don't over-simplify.
        4. You keep the language, e.g. german tags are replaced by german tags
        Respond in JSON format with a list of replacements:
        5. Replace synonym tags (e.g. buddys and friend) with a common tag-name
        {
            "replacements": [
                {"original": "tag1", "replacement": "tag2"},
                ...
            ]
        }
        """

        super().__init__(model="gpt-4-1106-preview", system_message=system_message)

        self.response_format="json_object"
        
    def send_request(self, tags):
        # Step 1: Simplify and summarize tags
        message = f"Improve the following tags: {tags}"
        
        self.add_message(message, role="user")

        response = super().send_request(temperature=0.3, response_format = self.response_format)
        
        # Now we do a second request to optimize the result
        message2 = """
        Reevaluate your response. Ensure that only synonymous tags are consolidated 
        and that no information is lost in the process. Try to find the most specific/useful 
        name for each tag. Preserve the original language. Check if meaningful tags 
        are preserved and the response is complete and correct. 
        Respond in the same JSON format.
        """
        self.add_message(response, "assistant")
        self.add_message(message2, "user")

        logging.debug("Optimizing response...")
        response = response = super().send_request(temperature=0.3, response_format = self.response_format)
        
        try: 
            replacements = json.loads(response)
            replacements = replacements['replacements']
        except Exception as e: 
            logging.error("Could not interpret AI answer for tag simplification: " + pprint.pformat(response))
            replacements = {}

        return replacements

def num_tokens_from_string(string: str, encoding_name: str = "cl100k_base") -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens