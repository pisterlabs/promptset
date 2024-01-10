from pathlib import Path

import openai
import pytesseract
import cv2
import re
from PIL import Image, ImageTk
import dateutil.parser as date_parser
from datetime import datetime
from unidecode import unidecode
import tkinter as tk
from tkinter import filedialog
import gspread
import json
from oauth2client.service_account import ServiceAccountCredentials


class Ticket():
    def __init__(self, ticket_image, file_name, sheet=None, gpt=False) -> None:
        
        """
        Initialize a Ticket object.

        Args:
            ticket_image (str): Path to the ticket image.
            file_name (str): Name of the ticket file.
            sheet (object): Optional sheet object for storing ticket information.
            gpt (bool): Flag indicating whether to use GPT for information extraction.
        """

        Ticket.system_prompt = """Tu sais lire les tickets de caisse. On te donnera toujours ce qui a été lu sur un ticket de caisse. Ton but sera de récupérer 3 informations sur ce ticket, le montant total, la date de l’achat, et le libellé de l’achat, c’est à dire le magasin dans lequel a été effectué la dépense ou la raison de la dépense (il faut que le libellé soit court mais explicite). Ajoute CB au début du libellé si le paiement a été effectué en carte bancaire. Tu présenteras ta recherche en renvoyant uniquement un dictionnaire de la forme suivante : {"libelle" : "" , "date" : "jj/mm/aaaa", "montant" : "nombre"}
        le montant doit être une chaine de caractère contenant un nombre. De plus, si tu ne trouves pas l’une des 3 valeurs, alors met None."""
        self.ticket_image = ticket_image
        self.file_name = file_name
        self.reading_status = True
        self.sheet = sheet
        self.text_recognition = pytesseract.image_to_string(self.ticket_image, lang="fra")
        self.filter_text_recognition()
        self.gpt_ticket_info = {}
        self.date = None
        self.libelle = None
        self.amount = None
        self.gpt = gpt
        if "cartebancaire" in self.filtered_text:
            self.get_date()
            self.get_libelle()
            self.get_amount()
        else:
            self.gpt_request()
    
    def __str__(self) -> str:

        """
        Return a string representation of the Ticket object.

        Returns:
            str: String representation of the Ticket object.
        """

        return f"{self.file_name} : {self.date}-{self.libelle}-{self.amount}"

    def __repr__(self) -> str:

        """
        Return a string representation of the Ticket object.

        Returns:
            str: String representation of the Ticket object.
        """

        return f"{self.file_name} : {self.date}-{self.libelle}-{self.amount}"
    
    def filter_text_recognition(self):

        """
        Apply text filtration on the output of OCR.

        This method keeps only alphabet characters and digits, removing unwanted characters.
        """

        # Application d’une filtration sur le texte de sortie de l’OCR permettant d’enlever les caractères qui ne m’intéresse pas
        # Je ne garde que les caractères de l’alphabet ou les chiffres
        self.filtered_text = ""
        for char in self.text_recognition:
            if char.isalpha():
                char = unidecode(char)
                self.filtered_text += char.lower()
            elif char.isdigit():
                self.filtered_text += char
            elif (char == "," or char == ".") and self.filtered_text[-1] != char:
                self.filtered_text += char

    def get_date(self):

        """
        Extract and set the date from the ticket.

        This method uses a regex pattern to find the date in the OCR text.
        """

        # Mise en place d’une recherche de la date du ticket
        # Utilisation d’une règle regex permettant de potentiellement trouver facilement la date
        # Si on ne la trouve pas, cela sera fait à la main ou à l’aide de GPT3.5

        date_pattern = r'(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})'
        match = re.search(date_pattern, self.text_recognition)
        if match:
            print("match")
            date_str = match.group(1)
            self.date_index = 0
            detected_date = datetime.strptime(date_str, "%d/%m/%y") if len(date_str) == 8 else datetime.strptime(date_str, "%d/%m/%Y")
            while date_str not in self.text_recognition.splitlines()[self.date_index]:
                self.date_index += 1
            self.date = detected_date.strftime("%d/%m/%Y")
        else:
            print("date est None")
            self.date = None

    def get_amount(self):

        """
        Extract and set the amount from the ticket.

        This method looks for keywords related to the amount and extracts the relevant information.
        """

        # Recherche du montant de la transaction dans le ticket, 
        # Si c’est un ticket de carte bancaire, il y a des mots clés qui permettent de les repérer facilement
        # Dans le cas où ce n’est pas un ticket classique, on utilisera GPT3.5 pour trouver le montant.
        keyword_amount = "montant"
        keyword_money = "eur"
        if keyword_amount in self.filtered_text:
            amount_index_end = self.filtered_text.index(keyword_amount) + len(keyword_amount)
            money_index = self.filtered_text[amount_index_end:].index(keyword_money)
            self.amount = self.filtered_text[amount_index_end:amount_index_end + money_index]
        else:
            self.reading_status = False
            self.amount = "unable to read amount"

    def get_libelle(self):

        """
        Extract and set the libelle from the ticket.

        This method extracts the libelle from a standard credit card ticket.
        """

        # Dans le cas où c’est un ticket classique de carte bleue, il est possible de trouver juste après la date, le libelle de la transaction/l-v+@@)
        lines = self.text_recognition.splitlines()
        self.libelle = "CB " + lines[self.date_index + 1].capitalize()
    
    def gpt_request(self):

        """
        Make a GPT request to extract missing information.

        This method uses GPT3.5 to fill in missing date, libelle, and amount information.
        """

        # Mise en place de la requête GPT3.5 en fonction de ce que l’analyse de l’OCR nous a permi de trouver
        if self.gpt:
            if not self.gpt_ticket_info:
                self.gpt_response = openai.ChatCompletion.create(
                            model="gpt-3.5-turbo",
                            messages=[
                                {"role": "system", "content": Ticket.system_prompt}
                            ]
)
                self.gpt_info = self.gpt_response['choices'][0]['message']['content']
                self.gpt_ticket_info = json.loads(self.gpt_info)

            if not self.date:
                self.date = self.gpt_ticket_info.get("date", None)
            self.libelle = self.gpt_ticket_info.get("libelle", None)
            self.amount = self.gpt_ticket_info.get("montant", None)
        else:
            self.get_date()
            if self.gpt_ticket_info:
                if not self.date:
                    self.date = self.gpt_ticket_info.get("date", None)
                self.libelle = self.gpt_ticket_info.get("libelle", None)
                self.amount = self.gpt_ticket_info.get("montant", None)
        
        self.verify_status()


    
    def add_to_sheet(self, line):

        """
        Add the ticket information to the specified sheet.

        Args:
            line (int): Line number in the sheet to update.
        """

        # Ajout du ticket courant dans le sheet sélectionné
        if self.reading_status:
            self.sheet.update(f"A{line}:F{line}", [[self.date, self.libelle, "", self.amount, "", f"=F{line - 1} - D{line}"]], raw=False)
    
    def verify_status(self):

        """
        Verify the status of the extracted information.

        This method checks if the extracted date, libelle, and amount are valid.
        """

        date_verif = False
        libelle_verif = False
        amount_verif = False

        # Vérification de la date
        if isinstance(self.date, str):
            date_pattern = r'(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})'
            match = re.search(date_pattern, self.date)
            if match:
                date_verif = True
        
        # Vérification du libellé
        if isinstance(self.libelle, str):
            if len(self.libelle) > 0:
                libelle_verif = True
        
        # Vérification du montant
        try:
            if self.amount:
                if "," in self.amount:
                    _ = float(".".join(self.amount.split(",")))
                else:
                    _ = float(self.amount)
                amount_verif = True
        except ValueError:
            pass
        if amount_verif and "." in self.amount:
            self.amount = ",".join(self.amount.split(""))

        if date_verif and libelle_verif and amount_verif:
            self.reading_status = True
        else:
            self.reading_status = False
    
    @staticmethod   
    def preprocess_image(image_path):

        """
        Preprocess the ticket image.

        Args:
            image_path (str): Path to the ticket image.

        Returns:
            Image: Processed image in the form of a PIL Image.
        """

        # Charger l'image en niveaux de gris
        image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)

        # Appliquer un seuillage adaptatif pour binariser l'image
        _, threshold_image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Convertir l'image en objet PIL pour la passer à Tesseract OCR
        return Image.fromarray(threshold_image)

class TicketReader():
    def __init__(self, width, height, ticket_directory) -> None:

        """
        Initialize a TicketReader object.

        Args:
            width (int): Width of the Tkinter window.
            height (int): Height of the Tkinter window.
            ticket_directory (str): Directory to store ticket images.
        """

        self.root = tk.Tk()
        self.root.title("Ticket Reader")
        self.width = width
        self.height = height
        self.root.geometry(f"{self.width}x{self.height}")
        self.root.resizable(width=False, height=False)
        self.label_error = None
        self.failing_menu_created = False
        self.ticket_page_opened = False
        self.tickets = []
        self.failing_tickets = []
        self.success_tickets = []
        self.failing_menu_element = []
        self.use_gpt = False

        self.ticket_directory = ticket_directory
        self.initialize_sheets_connection()
        self.add_widgets()
        
    def initialize_sheets_connection(self):

        """
        Initialize the connection to Google Sheets.

        This method sets up the connection to Google Sheets for storing ticket information.
        """
        
        self.scope = ["https://spreadsheets.google.com/feeds","https://www.googleapis.com/auth/spreadsheets","https://www.googleapis.com/auth/drive.file","https://www.googleapis.com/auth/drive"]
        self.identifiants = ServiceAccountCredentials.from_json_keyfile_name("identifiants.json", self.scope)
        self.client = gspread.authorize(self.identifiants)
        self.compte = self.client.open_by_key("1uCilvm7ps6XemNSUbDIaacO4Jg3VMpblc6n_3xl34As")

        self.sheets = {"Relevé SG" : self.compte.worksheet("Relevé SG"),
                       "Relevé Boursorama" : self.compte.worksheet("Relevé Boursorama")}

    def add_widgets(self):

        """
        Add Tkinter widgets to the window.

        This method creates and packs various widgets such as labels, buttons, and entry fields.
        """

        self.title = tk.Label(self.root, 
                         height=2,
                         width=18,
                         state="disabled",
                         font=("Arial", 30),
                         text="Lecteur de tickets")

        # choose sheets
        self.selected_sheet = tk.StringVar()
        self.selected_sheet.set("Sélectionner une page")
        self.sheets_menu = tk.OptionMenu(self.root, 
                                         self.selected_sheet, 
                                         *self.sheets.keys(),
                                         command=self.destroy_label_error)

        self.button_find_ticket = tk.Button(self.root, 
                                            text="Sélectionner photos tickets",
                                            command=self.add_ticket_photos)

        self.ticket_count = len(list(ticket_directory.iterdir()))
        self.information_ajout = tk.StringVar()
        self.information_ajout_label = tk.Label(self.root, textvariable=self.information_ajout)
        if self.ticket_count == 0:
            self.information_ajout.set(value="Aucun ticket sélectionné")
        else:
            self.information_ajout.set(value=f"{self.ticket_count} tickets sélectionnés")

        self.button_upload_ticket = tk.Button(self.root, text="Ajouter tickets", command=self.upload_ticket)

        self.button_gpt_activation = tk.Button(self.root, 
                                               text="Chat GPT", 
                                               background="#e03d31", 
                                               width=12, 
                                               font=("Arial 14"),
                                               command=self.gpt_activation)

        self.title.pack(side=tk.TOP)
        self.sheets_menu.pack(side=tk.TOP, pady=10)
        self.button_find_ticket.pack()
        self.information_ajout_label.pack()
        self.button_upload_ticket.pack(pady=10)
        self.button_gpt_activation.pack(side=tk.BOTTOM, pady=4)

    def add_ticket_photos(self):

        """
        Open a file dialog to add ticket photos to the directory.

        This method allows the user to select and add ticket photos to the specified directory.
        """

        file_paths = filedialog.askopenfilenames(filetypes=[("Images", "*.jpg;*.jpeg;*.png")])
        if file_paths:
            for path in file_paths:
                destination_file = self.ticket_directory / Path(path).name
                if not destination_file.exists():
                    destination_file.write_bytes(Path(path).read_bytes())
                    self.ticket_count += 1
        self.information_ajout.set(value=f"{self.ticket_count} tickets sélectionnés")
    
    def upload_ticket(self):

        """
        Upload tickets to the selected Google Sheet.

        This method creates Ticket objects, verifies them, and uploads valid tickets to the Google Sheet.
        """

        # Vérification choix de la page
        print("trying to upload tickets")
        if self.selected_sheet.get() in self.sheets.keys():
            if self.label_error:
                self.label_error.destroy()
                self.label_error = None

            # Création des objets tickets
            self.create_tickets()

            # Vérifier que tout les tickets sont bons
            ready_to_upload = True
            for ticket in self.tickets:
                if not ticket.reading_status:
                    ready_to_upload = False
                    if ticket not in self.failing_tickets:
                        self.failing_tickets.append(ticket)
                else:
                    if ticket not in self.success_tickets:
                        self.success_tickets.append(ticket)
            if ready_to_upload:
                print("ready to upload")
                # get last line of sheets
                line = len(self.sheets[self.selected_sheet.get()].get_all_values()) + 1
                for ticket in self.success_tickets:
                    ticket.add_to_sheet(line=line)
                    line += 1
                self.delete_tickets_files()
            else:
                print("error while reading")
                self.create_failing_menu()
                    
        else:
            if not self.label_error:
                self.label_error = tk.Label(self.root, text="Page non sélectionnée")
                self.label_error.pack()

    def create_tickets(self):

        """
        Create Ticket objects for each file in the ticket directory.

        This method creates Ticket objects for each file in the ticket directory and adds them to the list.
        """

        for file in self.ticket_directory.iterdir():
            if file.is_file():
                if file.name not in [ticket.file_name for ticket in self.tickets]:
                    ticket_image = Ticket.preprocess_image(image_path=file)
                    ticket = Ticket(ticket_image=ticket_image, 
                                    file_name=file.name, 
                                    sheet=self.sheets[self.selected_sheet.get()],
                                    gpt=self.use_gpt)
                    self.tickets.append(ticket)
                
    def destroy_label_error(self, event):

        """
        Destroy the label error widget.

        This method destroys the label error widget if it exists.
        """

        if self.label_error:
                self.label_error.destroy()

    def create_failing_menu(self):

        """
        Create a menu for failing tickets.

        This method creates a menu for failing tickets, allowing the user to modify and correct them.
        """

        if not self.failing_menu_created:
            self.failing_menu_title = tk.Label(self.root, text="À modifier", font=("Arial 16"))
            self.frame_ticket_menu = tk.Frame(self.root)
            self.scrollbar_ticket_menu = tk.Scrollbar(self.frame_ticket_menu, orient="vertical")
            self.list_tickets = tk.Listbox(self.frame_ticket_menu, width=50, yscrollcommand=self.scrollbar_ticket_menu.set)

            self.scrollbar_ticket_menu.config(command=self.list_tickets.yview)
            self.list_tickets.bind("<<ListboxSelect>>", self.on_listbox_select)

            self.failing_menu_title.pack(pady=10)
            self.frame_ticket_menu.pack()
            self.scrollbar_ticket_menu.pack(side=tk.RIGHT, fill=tk.Y)
            self.list_tickets.pack()

            for ticket in self.failing_tickets:
                print(ticket.file_name)
                self.list_tickets.insert(tk.END, ticket.file_name)
            self.failing_menu_element.append(self.failing_menu_title)
            self.failing_menu_element.append(self.frame_ticket_menu)
        else:
            self.list_tickets.delete(0, tk.END)
            for ticket in self.failing_tickets:
                print(ticket.file_name)
                self.list_tickets.insert(tk.END, ticket.file_name)

        self.failing_menu_created = True

    def on_listbox_select(self, event):

        """
        Handle the selection of an item in the listbox.

        This method handles the selection of an item in the listbox and opens the corresponding ticket page.
        """

        # Récupérer le ticket sélectionné
        selected_index = self.list_tickets.curselection()
        if selected_index:
            selected_item = self.list_tickets.get(selected_index)
            self.selected_ticket = self.get_ticket_by_filename(filename=selected_item)
            if not self.ticket_page_opened:
                self.open_ticket_page()
                self.ticket_page_opened = True

    def delete_failing_menu(self):

        """
        Delete the failing menu.

        This method deletes the failing menu if it exists.
        """

        for element in self.failing_menu_element:
            element.destroy()
            element = None
        self.failing_menu_created = False

    def get_ticket_by_filename(self, filename):

        """
        Get a Ticket object based on the filename.

        Args:
            filename (str): Filename of the ticket.

        Returns:
            Ticket: Ticket object corresponding to the filename.
        """

        for ticket in self.failing_tickets:
            if ticket.file_name == filename:
                return ticket

    def open_ticket_page(self):

        """
        Open a separate window for viewing and modifying ticket information.

        This method opens a separate window for viewing and modifying ticket information.
        """

        self.ticket_page = tk.Toplevel(self.root)
        self.ticket_page.resizable(width=False, height=False)
        self.ticket_page.geometry(f"{800}x{800}")
        self.ticket_page.protocol("WM_DELETE_WINDOW", self.close_ticket_page)

        # Obtenir l’image à la bonne taille 
        ticket_image = Image.open(self.ticket_directory / self.selected_ticket.file_name)
        resized_ticket_image = ticket_image.resize((500, 800))
        self.selected_ticket_image = ImageTk.PhotoImage(resized_ticket_image)
        self.label_ticket_image = tk.Label(self.ticket_page, image=self.selected_ticket_image)
        self.label_ticket_image.pack(side=tk.LEFT)

        # Création des champs pour modifier
        self.selected_ticket_date = tk.StringVar()
        self.selected_ticket_libelle = tk.StringVar()
        self.selected_ticket_amount = tk.StringVar()

        if self.selected_ticket.date:
            self.selected_ticket_date.set(self.selected_ticket.date)
        else:
            self.selected_ticket_date.set("")

        if self.selected_ticket.libelle:
            self.selected_ticket_libelle.set(self.selected_ticket.libelle)
        else:
            self.selected_ticket_libelle.set("")

        if self.selected_ticket.amount:
            self.selected_ticket_amount.set(self.selected_ticket.amount)
        else:
            self.selected_ticket_amount.set("")

        self.date_input = tk.Entry(self.ticket_page, textvariable=self.selected_ticket_date, font=("Arial 14"))
        self.libelle_input = tk.Entry(self.ticket_page, textvariable=self.selected_ticket_libelle, font=("Arial 14"))
        self.amount_input = tk.Entry(self.ticket_page, textvariable=self.selected_ticket_amount, font=("Arial 14"))

        self.ticket_page_title = tk.Label(self.ticket_page, text="Informations du ticket", font=("Arial 20"))
        self.date_title = tk.Label(self.ticket_page, text="Date (jj/mm/aaaa)", font=("Arial 16"))
        self.libelle_title = tk.Label(self.ticket_page, text="Libelle", font=("Arial 16"))
        self.amount_title = tk.Label(self.ticket_page, text="Montant (€)", font=("Arial 16"))
        self.button_validate_ticket = tk.Button(self.ticket_page, 
                                                text="Valider", 
                                                command=self.close_ticket_page, 
                                                font=("Arial 18"),
                                                background="#2cb327")

        self.ticket_page_title.pack(pady=10)
        self.date_title.pack(pady=20)
        self.date_input.pack(pady=10)
        self.libelle_title.pack(pady=20)
        self.libelle_input.pack(pady=10)
        self.amount_title.pack(pady=20)
        self.amount_input.pack(pady=10)
        self.button_validate_ticket.pack(pady=50)

    def close_ticket_page(self):

        """
        Close the ticket page window.

        This method closes the ticket page window and updates the ticket information based on user modifications.
        """

        self.ticket_page.destroy()
        self.ticket_page_opened = False
        print("windows close")

        # Actualisation des valeurs du ticket
        self.selected_ticket.date = self.selected_ticket_date.get() if self.selected_ticket_date.get() else None
        self.selected_ticket.libelle = self.selected_ticket_libelle.get() if self.selected_ticket_libelle.get() else None
        self.selected_ticket.amount = self.selected_ticket_amount.get() if self.selected_ticket_amount.get() else None

        # Vérification de la validité des nouvelles informations du ticket
        self.selected_ticket.verify_status()
        if self.selected_ticket.reading_status:
            # Suppression de la listebox
            ticket_index = self.failing_tickets.index(self.selected_ticket)
            self.list_tickets.delete(ticket_index)
            self.failing_tickets.pop(ticket_index)

            self.success_tickets.append(self.selected_ticket)
            print("ticket validé avec succès", ticket_index)
            if len(self.list_tickets.get(0, tk.END)) == 0:
                self.delete_failing_menu()
    
    def gpt_activation(self):

        """
        Activate or deactivate GPT for failing tickets.

        This method activates or deactivates GPT for failing tickets and updates their information accordingly.
        """

        if self.use_gpt:
            print("chat gpt désactivé")
            self.button_gpt_activation["bg"] = "#e03d31"
            self.use_gpt = False
        else:
            print("chat gpt activé")
            self.button_gpt_activation["bg"] = "#2cb327"
            self.use_gpt = True

        for ticket in self.failing_tickets:
            ticket.gpt = self.use_gpt
            ticket.gpt_request()
    
    def delete_tickets_files(self):

        """
        Delete all files in the ticket directory.

        This method deletes all files in the ticket directory and resets ticket-related attributes.
        """

        for file in self.ticket_directory.iterdir():
            if file.is_file():
                file.unlink()
        self.ticket_count = 0 
        self.information_ajout.set(value="Aucun ticket sélectionné")
        self.tickets = []
        self.failing_tickets = []
        self.success_tickets = []
        self.failing_menu_element = []

if __name__ == "__main__":
    ticket_directory = Path(__file__).parent / "tickets"
    ticket_reader = TicketReader(width=500, height=500, ticket_directory=ticket_directory)
    ticket_reader.root.mainloop()
    pass