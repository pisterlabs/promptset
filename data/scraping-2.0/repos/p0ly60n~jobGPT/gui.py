"""A graphical user interface for the job scraper and cover letter generator."""
import webbrowser
import csv
import os
import subprocess
from threading import Thread

import customtkinter as ctk
from customtkinter import filedialog
from CTkMessagebox import CTkMessagebox
from pypdf import PdfReader
from openai import AuthenticationError

from selenium.common.exceptions import NoSuchElementException, TimeoutException

import src.scraper as scraper
import src.gpt as gpt
from src.job import Job

class App(ctk.CTk):
    """A graphical user interface for a job scraper and cover letter generator."""

    def __init__(self):
        """Initializes the GUI and sets up the main window."""
        super().__init__()
        self.title("Job-Scraper")
        ctk.set_default_color_theme("blue")
        ctk.set_appearance_mode("system")
        self.resizable(False, False)

        self.website_choices = ["Stepstone", "Indeed"]
        self.website_choice = ctk.StringVar()
        self.website_choice.set(self.website_choices[0])

        self.interest_label = ctk.CTkLabel(self, text="Interest")
        self.location_label = ctk.CTkLabel(self, text="Location")
        self.radius_label = ctk.CTkLabel(self, text="Radius")
        self.no_of_jobs_label = ctk.CTkLabel(self, text="#Jobs")
        self.resume_label = ctk.CTkLabel(self, text="Resume")
        self.website_label = ctk.CTkLabel(self, text="Website")

        self.interest_field = ctk.CTkEntry(self)
        self.location_field = ctk.CTkEntry(self)
        self.radius_field = ctk.CTkEntry(self)
        self.no_of_jobs_field = ctk.CTkEntry(self)
        self.resume_field = ctk.CTkButton(self, text="Choose File", command=self.file_open)

        self.interest_label.grid(row=0, column=0, padx=5, pady=5)
        self.location_label.grid(row=1, column=0, padx=5, pady=5)
        self.radius_label.grid(row=2, column=0, padx=5, pady=5)
        self.no_of_jobs_label.grid(row=3, column=0, padx=5, pady=5)
        self.resume_label.grid(row=4, column=0, padx=5, pady=5)
        self.website_label.grid(row=5, column=0, padx=5, pady=5)

        self.interest_field.grid(row=0, column=1, padx=5, pady=5)
        self.location_field.grid(row=1, column=1, padx=5, pady=5)
        self.radius_field.grid(row=2, column=1, padx=5, pady=5)
        self.no_of_jobs_field.grid(row=3, column=1, padx=5, pady=5)
        self.resume_field.grid(row=4, column=1, padx=5, pady=5)

        ctk.CTkOptionMenu(self, variable=self.website_choice, values=self.website_choices).grid(row=5, column=1, padx=5, pady=5)

        self.scrape_button = ctk.CTkButton(self, text="Scrape", command=self.scrape, fg_color="green", hover_color="dark green")
        self.scrape_button.grid(row=6, column=1, padx=5, pady=5)

        # getting the working root directory
        self.directory = os.getcwd()
        self.output_directory = os.path.join(self.directory, "output")
        # creating the output directory if it does not exist
        if not os.path.exists(self.output_directory):
            os.mkdir(self.output_directory)
            print(f"Directory added at {self.output_directory}")

        self.csv_file_name = ""
        self.personal_info = ""
        self.jobs = []
        self.resume_file = ""

        self.mainloop()

    def scrape(self):
        """Scrapes job data from a website based on user input and saves it to a CSV file."""
        interest = self.interest_field.get().strip()
        location = self.location_field.get().strip()
        try:
            radius = int(self.radius_field.get())
            no_of_jobs = int(self.no_of_jobs_field.get())
        except ValueError:
            CTkMessagebox(title="Error", message="Please enter a valid number!", icon="error")
            return
        resume_file = self.resume_file

        print(f"\nExtracting data of {resume_file}...\n")
        self.extract_personal_info(resume_file)

        print("\nScraping...\n")

        # create a scraper object based on the user's choice
        website_scraper = None
        choice = self.website_choice.get()
        if (choice == "Stepstone"):
            website_scraper = scraper.StepstoneScraper(interest.replace(" ", "%20"), \
            location.replace(" ", "%20"), radius, no_of_jobs)
        elif (choice == "Indeed"):
            website_scraper = scraper.IndeedScraper(interest.replace(" ", "%20"), \
            location.replace(" ", "%20"), radius, no_of_jobs)
        else:
            print("Website not supported yet, edit gui.py!")

        # run the scraper to extract the job data from the website
        if website_scraper is not None:
            try:
                self.jobs = website_scraper.scrape()
            except NoSuchElementException:
                CTkMessagebox(title="Error", message="Reading of one job element failed", icon="error")
                return
            except TimeoutException:
                CTkMessagebox(title="Error", message="Connection to the website failed", icon="error")
                return
            except Exception as e:
                CTkMessagebox(title="Error", message=f"An unknown error occurred: {e}", icon="error")
                return
        else:
            CTkMessagebox(title="Error", message="Website scraper is not initialized", icon="error")
            return

        print("\nScraping done!\n")

        self.csv_file_name = interest.title() + "_" + location.title() + "_Jobs.csv"

        self.save_csv()

        JobWindow(self.jobs, self.output_directory, self.personal_info)

    def save_csv(self):
        """Saves the scraped job data to a CSV file."""
        with open(os.path.join(self.output_directory, self.csv_file_name), mode="w", encoding="utf8") as csv_file:
            writer = csv.writer(csv_file, delimiter=",", lineterminator="\n")
            writer.writerow(["TITLE", "COMPANY", "LOCATION", "LINK"])

        for job in self.jobs:
            job.write_to_file(os.path.join(self.output_directory, self.csv_file_name))

    def file_open(self):
        """Opens a file dialog to select a file."""
        filename = filedialog.askopenfilename(initialdir=self.directory, title="Select a file", filetypes=(("PDF files", "*.pdf"), ("TXT files", "*.txt")))
        if (os.path.basename(filename)) != "":
            self.resume_field.configure(text=os.path.basename(filename))
            self.resume_file = filename

    def extract_personal_info(self, resume_file: str):
        """Extracts personal information from a cover letter PDF or TXT file."""
        # if a resume is available, extract the text from it
        if (resume_file != ""):
            if (os.path.splitext(resume_file)[1] == ".txt"):
                with open(resume_file, mode="r", encoding="utf8") as txt_file:
                    self.personal_info = txt_file.read()
            elif (os.path.splitext(resume_file)[1] == ".pdf"):
                reader = PdfReader(resume_file)
                for page in reader.pages:
                    self.personal_info += page.extract_text()
            else:
                print("File format not supported yet, edit gui.py!")
        else:
            print("No resume selected!")

class JobWindow(ctk.CTkToplevel):
    def __init__(self, jobs: list[Job], output_directory: str, personal_info: str=""):
        super().__init__()
        self.title("Jobs")
        self.grid_columnconfigure(0, minsize=200, weight=1)
        self.grid_rowconfigure(0, minsize=400, weight=1)

        self.jobs = jobs
        self.selected = []
        self.output_directory = output_directory
        self.personal_info = personal_info

        self.frame = ctk.CTkScrollableFrame(self)

        self.resume_needed_label = ctk.CTkLabel(self.frame, text="RESUME?")
        self.title_label = ctk.CTkLabel(self.frame, text="TITLE")
        self.company_label = ctk.CTkLabel(self.frame, text="COMPANY")
        self.location_label = ctk.CTkLabel(self.frame, text="LOCATION")
        self.link_label = ctk.CTkLabel(self.frame, text="LINK")

        self.resume_needed_label.grid(row=0, column=0, padx=40, sticky="WE", pady=5)
        self.title_label.grid(row=0, column=1, padx=40, sticky="WE", pady=5)
        self.company_label.grid(row=0, column=2, padx=40, sticky="WE", pady=5)
        self.location_label.grid(row=0, column=3, padx=40, sticky="WE", pady=5)
        self.link_label.grid(row=0, column=4, padx=40, sticky="WE", pady=5)

        for idx, job in enumerate(self.jobs, start=0):
            var = ctk.IntVar()
            self.selected.append(var)
            ctk.CTkCheckBox(self.frame, variable=var, text="", width=30).grid(row=idx+1, column=0, padx=5, pady=5)
            ctk.CTkLabel(self.frame, text=job.job_title[0:40] + ("..." if len(job.job_title) >= 40 else "")).grid(row=idx+1, column=1, sticky="W", padx=5, pady=5)
            ctk.CTkLabel(self.frame, text=job.job_company[0:40] + ("..." if len(job.job_company) >= 40 else "")).grid(row=idx+1, column=2, sticky="W", padx=5, pady=5)
            ctk.CTkLabel(self.frame, text=job.job_location[0:40] + ("..." if len(job.job_location) >= 40 else "")).grid(row=idx+1, column=3, sticky="W", padx=5, pady=5)
            ctk.CTkButton(self.frame, text="click here", command=lambda link=job.job_link: self.callback(link)).grid(row=idx+1, column=4, padx=5, pady=5)

        self.generate_button = ctk.CTkButton(self.frame, text="Generate", command=self.generate_letter, fg_color="green", hover_color="dark green")
        self.generate_button.grid(row=len(self.jobs) + 1, column=4, padx=5, pady=5)

        self.frame.grid(row=0, column=0, sticky="nsew")

        # update the window size to fit the frame
        self.update_idletasks()
        width = self.frame.winfo_reqwidth()
        height = self.winfo_height()
        self.geometry(f'{width+20}x{height}')

    def callback(self, link: str):
        """Opens a web browser with the provided link."""
        webbrowser.open_new(link)


    def start_gpt_generation_threaded(self, job: Job, job_index: int, personal_info: str):
        """Generates a cover letter using GPT-3 for a specific job in a separate thread."""
        with open(os.path.join(self.output_directory, f"job{job_index}@{job.job_company}.txt"), mode="w", encoding="utf8") as txt_file:
            # get the resume
            try:
                gpt_data = gpt.get_letter(job, personal_info)
            except AuthenticationError:
                CTkMessagebox(title="Error", message="Authentication failed, try to check your api_key", icon="error")
                return
            except Exception as e:
                CTkMessagebox(title="Error", message=f"An unknown error occured: {e}", icon="error")
                return
            txt_file.write(gpt_data["message"])
            print(f"Cover-Letter for job{job_index} ({job}) generated!")
            print(f"-> Tokens: in:{gpt_data['input_tokens']}, out:{gpt_data['output_tokens']}")

    def generate_letter(self):
        """Generates cover letters for selected jobs using GPT-3."""
        threads = []
        amount_selected = 0

        while (amount_selected == 0):
            for idx, job in enumerate(self.jobs, start=0):
                if self.selected[idx].get() == 1:
                    threads.append(Thread(target=self.start_gpt_generation_threaded, args=(job, idx, self.personal_info)))
                    amount_selected += 1
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()
            msg = CTkMessagebox(title="Success", message="Cover letters generated! Find them in the output folder.", \
                icon="check", option_1="Open Folder", option_2="Thanks")
            if (msg == "Open Folder"):
                subprocess.run(['open', self.output_directory], check=True)
                    
            self.destroy()