from django.test import TestCase
from django.urls import reverse
from django.test import TestCase
from django.core.files.uploadedfile import SimpleUploadedFile
import re
from api.views import consume_file
import openai
import PyPDF2



class ConsumeFileTestCase(TestCase):
    
    
    
    def test_non_pdf_file(self):
        # Send a POST request with a non-PDF file
        with open('C:\\Users\\Jaime\\Desktop\\SUMMARYDSD.txt', 'rb') as file:
            response = self.client.post(reverse('consume_file'), {'file': file})
        # Check that the response status code is 400 (bad request)
        self.assertEqual(response.status_code, 400)
        
        
        

    def test_pdf_file(self):
        # Send a POST request with a PDF file
        with open('C:\\Users\\Jaime\\Downloads\\article_with_multi_picture_figures.pdf', 'rb') as file:
            response = self.client.post(reverse('consume_file'), {'file': file})
        # Check that the response status code is 200 (success)
        self.assertEqual(response.status_code, 200)
        
        
        

    def test_text_extraction(self):
        # Read the PDF file
        with open('C:\\Users\\Jaime\\Downloads\\article_with_multi_picture_figures.pdf', 'rb') as file:
            expected_text = file.read().decode('utf-8')

        # Send a POST request with the PDF file
        with open('C:\\Users\\Jaime\\Downloads\\article_with_multi_picture_figures.pdf', 'rb') as file:
            response = self.client.post(reverse('consume_file'), {'file': file})
        # Get the extracted text from the response
        extracted_text = response.content.decode('utf-8')

        # Check that the extracted text is the same as the expected text
        self.assertEqual(expected_text, extracted_text)



    def test_image_removal(self):
        with open('C:\\Users\\Jaime\\Downloads\\article_with_multi_picture_figures.pdf', 'rb') as file:
            file_content = file.read()
            response = self.client.post('/consume_file/', {'file': file})

        file = SimpleUploadedFile("C:\\Users\\Jaime\\Downloads\\article_with_multi_picture_figures.pdf", file_content)

        # Here we can check the content of the file and make sure that it does not contain images
        self.assertEqual(response.status_code, 200)
        response_text = response.content.decode('utf-8')
        self.assertNotIn('\x00', response_text)

        
        
    
    def test_link_removal(self):
    # Make a POST request with a file containing links
        with open('C:\\Users\\Jaime\\Downloads\\article_with_multi_picture_figures.pdf', 'rb') as file:
            response = self.client.post('/consume_file/', {'file': file})
    # Get the text extracted from the file
        extracted_text = response.content.decode('utf-8')
    # Search for links in the extracted text
        links = re.findall(r'(http|https).+?(?=\s|$)', extracted_text)
    # Check that no links were found
        self.assertEqual(len(links), 0)
        

    
    def test_reference_removal(self):
        with open('C:\\Users\\Jaime\\Downloads\\article_with_multi_picture_figures.pdf', 'rb') as file:
            file_content = file.read()
            response = self.client.post('/consume_file/', {'file': file})
        file = SimpleUploadedFile("C:\\Users\\Jaime\\Downloads\\article_with_multi_picture_figures.pdf", file_content)

        # Aquí podríamos comprobar el contenido del texto extraído y asegurarnos de que no contiene secciones de referencias
        self.assertEqual(response.status_code, 200)
        response_text = response.content.decode('utf-8')
        self.assertNotIn('References', response_text)



    

    def test_remove_urls(self):
# Send a POST request with a PDF file containing URLs
        with open('C:\\Users\\Jaime\\Downloads\\article_with_multi_picture_figures.pdf', 'rb') as file:
            response = self.client.post(reverse('consume_file'), {'file': file})
# Check that the response status code is 200 (OK)
        self.assertEqual(response.status_code, 200)
# Check that the text of the response does not contain any URL
        response_text = response.content.decode('utf-8')

        self.assertNotIn('http', response_text)
        
        
        
        
    def test_replace_curly_quotes(self):
# Send a POST request with a PDF file containing curly quotes
        with open('C:\\Users\\Jaime\\Downloads\\article_with_multi_picture_figures.pdf', 'rb') as file:
            response = self.client.post(reverse('consume_file'), {'file': file})
# Check that the response status code is 200 (OK)
        self.assertEqual(response.status_code, 200)
# Check that the text of the response does not contain any curly quotes
        response_text = response.content.decode('utf-8')
        self.assertNotIn('’', response_text)
        
        
        
        
    def test_remove_special_characters(self):
    # Send a POST request with a PDF file containing special characters
        with open('C:\\Users\\Jaime\\Downloads\\article_with_multi_picture_figures.pdf', 'rb') as file:
            response = self.client.post(reverse('consume_file'), {'file': file})
    # Check that the response status code is 200 (OK)
        self.assertEqual(response.status_code, 200)
        response_text = response.content.decode('utf-8')

    # Check that the text of the response does not contain any special characters
        self.assertNotRegex(response_text, r'[^a-zA-Z0-9\'\"():;,.!?— ]+')

        
        
        
    def test_remove_empty_parentheses(self):    
    # Send a POST request with a PDF file containing empty parentheses
        with open('C:\\Users\\Jaime\\Downloads\\article_with_multi_picture_figures.pdf', 'rb') as file:
            response = self.client.post(reverse('consume_file'), {'file': file})
    # Check that the response status code is 200 (OK)
        self.assertEqual(response.status_code, 200)
        response_text = response.content.decode('utf-8')

    # Check that the text of the response does not contain any empty parentheses
        self.assertNotRegex(response_text, r'\(\s*\)')



    def test_remove_bracketed_numbers(self):
    # Send a POST request with a PDF file containing bracketed numbers
        with open('C:\\Users\\Jaime\\Downloads\\article_with_multi_picture_figures.pdf', 'rb') as file:
            response = self.client.post(reverse('consume_file'), {'file': file})
    # Check that the response status code is 200 (OK)
        self.assertEqual(response.status_code, 200)
        response_text = response.content.decode('utf-8')

    # Check that the text of the response does not contain any bracketed numbers
        self.assertNotRegex(response_text, r'\[[0-9]*\]')
        
        
        
    def test_replace_multiple_spaces(self):
# Send a POST request with a PDF file containing multiple spaces
        with open('C:\\Users\\Jaime\\Downloads\\article_with_multi_picture_figures.pdf', 'rb') as file:
            response = self.client.post(reverse('consume_file'), {'file': file})
# Check that the response status code is 200 (OK)
        self.assertEqual(response.status_code, 200)
        response_text = response.content.decode('utf-8')

# Check that the text of the response does not contain any multiple spaces
        self.assertNotRegex(response_text, r'\s{2,}')
        
        

    def test_summary(self):
        # Set the API key and model for OpenAI
        openai.api_key = "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"
        model_engine = "text-davinci-003"

        # Read the PDF file
        with open('C:\\Users\\Jaime\\Downloads\\article_with_multi_picture_figures.pdf', 'rb') as file:
            doc = PyPDF2.PdfFileReader(file)
            # Extract the text from the PDF
            text = doc.extract_text()

        # Use the OpenAI API to generate a summary of the text
        summary = openai.Completion.create(
            engine=model_engine,
            prompt=f"Please summarize the following text:\n{text}",
            temperature=0.5,
            max_tokens=1024,
            top_p=1,
            frequency_penalty=1,
            presence_penalty=1
        )
        # Get the summary generated by the API
        summary = summary.text

        # Check that the summary is not empty
        self.assertTrue(summary)
