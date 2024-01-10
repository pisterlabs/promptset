from controllers.file_processing.file_processor import FileProcessor

from controllers.invoice_processing.text_processor import TextProcessor
from controllers.openai.openai_connector import OpenAIConnector

class InvoiceExtractor:
    def __init__(self, config, translations, openai_config, txtEvents, invoice_processor):
        self.config = config  # Configuración
        self.txtEvents = txtEvents  # Eventos de texto
        self.translations = translations  # Traducciones
        self.invoice_processor = invoice_processor  # Procesador de facturas
        self.text_processor = TextProcessor(translations)  # Procesador de texto
        self.file_processor = FileProcessor(config, translations, self.text_processor)  # Procesador de archivos
        self.openai_connector = OpenAIConnector(openai_config, translations, txtEvents, self.text_processor, self.invoice_processor)  # Conector de OpenAI

    def extract_data_from_file(self, n_request, file_path ):
        if file_path.endswith('.msg'):  # Si el archivo termina en .msg
            email_subject, email_body, email_content = self.file_processor.process_msg_file(file_path)  # Procesa el archivo .msg
        elif file_path.endswith('.eml'):  # Si el archivo termina en .eml
            email_subject, email_body, email_content = self.file_processor.process_eml_file(file_path)  # Procesa el archivo .eml
        else:
            self.txtEvents.insertPlainText(f"{self.translations['ui']['InvalidFileFormatError']} {file_path}\n")  # Inserta un error de formato de archivo no válido

        n_request, headeri_data, detaili_data, headero_data, detailo_data, all_info = self.openai_connector.extract_data(n_request, email_subject, email_body, email_content)  # Extrae los datos
        return n_request, headeri_data, detaili_data, headero_data, detailo_data, all_info  # Devuelve los datos extraídos