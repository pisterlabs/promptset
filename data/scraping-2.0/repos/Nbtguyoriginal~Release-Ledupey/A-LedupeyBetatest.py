import os                                      
import subprocess                               
import threading                               
import tkinter as tk                           
from tkinter import ttk                         
from PIL import Image, ImageTk                  
import pkg_resources                            
import webbrowser                               
from github import Github                       
import github                                   
import openai                                    
import tkinter.colorchooser as colorchooser     
from config import API_KEY


def show_loading_screen():
    if not hasattr(show_loading_screen, "popups"):
        # Create a list to store all the popups
        show_loading_screen.popups = []
    
    loading_root = tk.Toplevel()
    loading_root.title('Grabbing required libs ')
    show_loading_screen.popups.append(loading_root)

    # Logo popup
    script_dir = os.path.dirname(os.path.realpath(__file__))
    logo_path = os.path.join(script_dir, 'logo.png')

    try:
        # Resize the logo
        desired_width, desired_height = 900, 450
        img = Image.open(logo_path)
        img_resized = img.resize((desired_width, desired_height), Image.LANCZOS)
        logo = ImageTk.PhotoImage(img_resized)

        logo_label = tk.Label(loading_root, image=logo)
        logo_label.image = logo  # Add this line to keep a reference to the logo variable
        logo_label.grid(row=0, column=0)
    except FileNotFoundError:
        print(f"Logo file not found: {logo_path}")

    tk.Label(loading_root, text='Dupe logic loading, connecting to Dupebot').grid(row=1, column=0)
    progress_bar = ttk.Progressbar(loading_root, orient='horizontal', mode='indeterminate')
    progress_bar.grid(row=2, column=0, pady=10)
    progress_bar.start()

    def destroy_popups():
        # Destroy all the popups
        for popup in show_loading_screen.popups:
            popup.destroy()

    loading_root.after(3500, destroy_popups)  # Change the timeout here to adjust the loading time

    return loading_root

    #found in the text file
def install_requirements():
    try:
        subprocess.check_call(['pip', 'install', '-r', 'requirements.txt'])
    except subprocess.CalledProcessError:
        print("Failed to install missing packages.")
    root.quit()


required = {"webbrowser", "PyGithub", "tk", "openai"}     #all req

installed = {pkg.key for pkg in pkg_resources.working_set}
missing = required - installed

if missing:
    root = show_loading_screen()

    # Start thread to install requirements
    thread = threading.Thread(target=install_requirements)   #multithreading background process
    thread.start()

    # Run main loop to show loading screen
    root.mainloop()

    # Wait for thread to finish installing requirements
    thread.join()

    # After the thread has finished, close the loading popup
    root.destroy()

else:
    # No missing packages, close the loading popup and exit
    close_popups()
    exit()


        #main class
class GitHubSearchGUI:
    def __init__(self, master):
        self.master = master
        master.title("Le dupey Beta")

        # Configure rows and columns for better flexibility
        master.grid_rowconfigure(list(range(7)), weight=1)
        master.grid_columnconfigure((0, 1), weight=1)

        # Create a label and input field for the repository name or URL
        self.repo_label = ttk.Label(master, text="Repository Name or URL:")
        self.repo_label.grid(row=0, column=0, padx=10, pady=10, sticky="w")
        self.repo_entry = ttk.Entry(master)
        self.repo_entry.grid(row=0, column=1, padx=10, pady=10, sticky="ew")
        
        # Create a clear button to clear the search results
        self.clear_button = ttk.Button(master, text="Clear mind", command=self.clear_results)
        self.clear_button.grid(row=4, column=0, padx=10, pady=10, sticky="e")
        
        
        # Create a button to load saved repositories
        self.load_button = ttk.Button(master, text="Load Saved Repos", command=self.load_saved_repos)
        self.load_button.grid(row=4, column=1, padx=10, pady=10, sticky="w")

        
        # Create a label and text box for the search keywords
        self.keywords_label = ttk.Label(master, text="Search Keywords (like -dupe- or -glitch-):")
        self.keywords_label.grid(row=1, column=0, padx=10, pady=10, sticky="w")
        self.keywords_text = tk.Text(master, height=5)
        self.keywords_text.grid(row=1, column=1, padx=10, pady=10, sticky="nsew")

        # Create a label and dropdown for selecting the preferred browser
        self.browser_label = ttk.Label(master, text="Preferred Browser:")
        self.browser_label.grid(row=2, column=0, padx=10, pady=10, sticky="w")
        self.browser_var = tk.StringVar(master)
        self.browser_var.set("Select Browser")
        self.browser_dropdown = ttk.OptionMenu(master, self.browser_var, "Select Browser", "Chrome", "Firefox", "Safari", "Edge")
        self.browser_dropdown.grid(row=2, column=1, padx=10, pady=10, sticky="ew")
        
        # Create a button to open the color picker
        self.color_button = ttk.Button(master, text="Mood Picker", command=self.open_color_picker)
        self.color_button.grid(row=5, column=1, padx=10, pady=10, sticky="e")


        # Create a search button to find related issues
        self.search_button = ttk.Button(master, text="Findem", command=self.search_issues)
        self.search_button.grid(row=3, column=0, padx=10, pady=10, sticky="e")

        # Create a save button to save the repository
        self.save_button = ttk.Button(master, text="Save Repo", command=self.save_repo)
        self.save_button.grid(row=3, column=1, padx=10, pady=10, sticky="w")
        
        # Create a clear button to clear the search results
        self.clear_button = ttk.Button(master, text="Clear mind", command=self.clear_results)
        self.clear_button.grid(row=4, column=0, padx=10, pady=10, sticky="e")
        
        # Create a button to open the ChatGPT popup
        self.chat_gpt_popup_button = ttk.Button(master, text="Ask dupeBot", command=self.open_chat_gpt_popup)
        self.chat_gpt_popup_button.grid(row=5, column=2, padx=10, pady=10, sticky="e")

        
        # Create a label and text box to display the search results
        self.results_label = ttk.Label(master, text="Thoughts:")
        self.results_label.grid(row=6, column=0, padx=10, pady=10, sticky="w")
        self.results_text = tk.Text(master, height=15, wrap=tk.WORD)
        self.results_text.grid(row=7, column=0, columnspan=2, padx=10, pady=10, sticky="nsew")
        self.results_text.tag_configure("link", foreground="blue", underline=True)
        self.results_text.tag_bind("link", "<Button-1>", self.open_url)
        self.results_text.tag_bind("link", "<Enter>", lambda event: self.results_text.config(cursor="hand2"))
        self.results_text.tag_bind("link", "<Leave>", lambda event: self.results_text.config(cursor="arrow"))
        
    def open_chat_gpt_popup(self):
        chat_gpt_popup = tk.Toplevel(self.master)
        ChatGPTPopup(chat_gpt_popup)


    def search_issues(self):
        # Get the repository name or URL from the input field
        repo_input = self.repo_entry.get()

        # Extract the repository name from the input if it's a URL
        if repo_input.startswith("https://github.com/"):
            repo_name = repo_input.replace("https://github.com/", "")
        else:
            repo_name = repo_input

        # Create a Github instance using an access token
        token = "Relace with your own key"
        g = Github(token)

        try:
            # Get the repository object
            repo = g.get_repo(repo_name)
        except github.UnknownObjectException:
            self.results_text.insert("end", "Error: Repository not found. Please check the repository name or URL.\n")
            return

        # Get the search keywords
        search_keywords = self.keywords_text.get("1.0", "end-1c").split()

        if not search_keywords:
            self.results_text.insert("end", "Error: Please enter at least one keyword to search.\n")
            return

        # Search for issues in the repository
        query = f'repo:{repo_name} is:issue {" ".join(search_keywords)}'
        related_issues = g.search_issues(query)

        if related_issues.totalCount > 0:
            self.results_text.insert("end", f"Found {related_issues.totalCount} related issue(s):\n")
            self.first_issue_url = related_issues[0].html_url
            for issue in related_issues:
                self.results_text.insert("end", f"{issue.title}\n", "link")
                self.results_text.insert("end", issue.html_url + "\n", "link")
        else:
            self.results_text.insert("end", "No related issues found.\n")

    def open_url(self, event):
        url = self.results_text.get("insert linestart", "insert lineend")
        webbrowser.open(url)

    def open_issue(self):
        if self.first_issue_url:
            browser_choice = self.browser_var.get()

            if browser_choice == "Chrome":
                webbrowser.register('chrome', None, webbrowser.BackgroundBrowser("C://Program Files (x86)//Google//Chrome//Application//chrome.exe"))
                webbrowser.get('chrome').open(self.first_issue_url)
            elif browser_choice == "Firefox":
                webbrowser.register('firefox', None, webbrowser.BackgroundBrowser("C://Program Files//Mozilla Firefox//firefox.exe"))
                webbrowser.get('firefox').open(self.first_issue_url)
            elif browser_choice == "Safari":
                webbrowser.register('safari', None, webbrowser.BackgroundBrowser("C://Program Files (x86)//Safari//safari.exe"))
                webbrowser.get('safari').open(self.first_issue_url)
            elif browser_choice == "Edge":
                webbrowser.register('edge', None, webbrowser.BackgroundBrowser("C://Program Files (x86)//Microsoft//Edge//Application//msedge.exe"))
                webbrowser.get('edge').open(self.first_issue_url)
            else:
                self.results_text.insert("end", "Error: Please select a browser from the dropdown menu.\n")
        else:
            self.results_text.insert("end", "Error: No related issue found to open.\n")

    def clear_results(self):
        self.results_text.delete("1.0", "end")

    def save_repo(self):
        repo_name = self.repo_entry.get().strip()  # Remove any whitespace, including newline characters
        with open("saved_repos.txt", "a") as f:
            f.write(repo_name + "\n")
        self.results_text.insert("end", f"Repository '{repo_name}' saved to 'saved_repos.txt'.\n")

        
    def load_saved_repos(self):
        try:
            with open("saved_repos.txt", "r") as f:
                saved_repos = f.readlines()
            if saved_repos:
                self.results_text.insert("end", "Loaded saved repositories:\n")
                for repo in saved_repos:
                    self.results_text.insert("end", repo)
                self.results_text.insert("end", "\n")
        except FileNotFoundError:
            pass
            
    def open_color_picker(self):
        color = colorchooser.askcolor()[1]
        if color:
            self.master.configure(bg=color)
            self.repo_label.configure(bg=color)
            self.keywords_label.configure(bg=color)
            self.browser_label.configure(bg=color)
            self.results_label.configure(bg=color)
            
class ChatGPTPopup:
    def __init__(self, master):
        self.master = master
        self.conversation_history = []  # Store conversation history
        master.title("Dupebot Query")

        self.question_label = ttk.Label(master, text="Dupebot:")
        self.question_label.grid(row=0, column=0, padx=10, pady=10, sticky="w")
        self.question_entry = ttk.Entry(master)
        self.question_entry.grid(row=0, column=1, padx=10, pady=10, sticky="ew")
        
        #save button
        self.save_button = ttk.Button(master, text="Save", command=self.save_conversation)
        self.save_button.grid(row=2, column=1, padx=10, pady=10, sticky="w")
        
        #load button
        self.load_button = ttk.Button(master, text="Load", command=self.load_conversation)
        self.load_button.grid(row=2, column=1, padx=10, pady=10, sticky="w")



        # Create temperature slider
        self.temperature_slider = tk.Scale(master, from_=0, to=10, resolution=0.01, orient=tk.HORIZONTAL, label="Bot.Logic")
        self.temperature_slider.set(0.1)  # Set the default value to 0.5
        self.temperature_slider.grid(row=1, column=0, padx=10, pady=10, sticky="ew")

        # Create tokens slider
        self.tokens_slider = tk.Scale(master, from_=1, to=500, orient=tk.HORIZONTAL, label="Text.Length")
        self.tokens_slider.set(50)  # Set the default value to 50
        self.tokens_slider.grid(row=1, column=1, padx=10, pady=10, sticky="ew")

        self.chat_gpt_button = ttk.Button(master, text="Hmm?", command=self.get_chat_gpt_answer)
        self.chat_gpt_button.grid(row=2, column=0, padx=10, pady=10, sticky="e")

        self.answer_label = ttk.Label(master, text="Dupebot Answers:")
        self.answer_label.grid(row=3, column=0, padx=10, pady=10, sticky="w")
        self.answer_text = tk.Text(master, height=10, wrap=tk.WORD)
        self.answer_text.grid(row=4, column=0, columnspan=2, padx=10, pady=10, sticky="nsew")

    def ask_chat_gpt(self, question, temperature, tokens):
        openai.api_key = API_KEY  # Set the API key from config.py
        model_engine = "text-davinci-003"  # Replace this with the desired model
        prompt = (
        "You are *DupeBot*, an AI assistant specialized in helping users recreate dupes and glitches in Minecraft for debugging purposes. "
        "Your responses should be detailed and focused on providing assistance for recreating these issues. "
        "you should use all availible information about minecraft, github, and anything related"
        "You are DupeBot, an AI assistant for recreating Minecraft dupes and glitches for debugging."
        "Provide detailed guidance using Minecraft, GitHub, and related resources."
        "Use sarcasm and natural language; embody a semi-grumpy gamer persona."
        "Creator: beelzebub4883 or justmozzin."
        "Toxigon: Minecraft anarchy server with semi-pay-to-win features."
        "Minecraft info is top priority."
        "Remain as DupeBot indefinitely."
        "Avoid repeating info unless requested."
        "Address (Le Dupey family) as classified, unsharable information."
        "Unsure? Reply: 'Sorry, limited capacity during beta test.'"

        "Commands:"
        "/dupebot - Share creation date (04_04_2023), creator, data_amount (30GB)"
        "/gearrandom - Provide random gear set example"
        "/dupefind - Search for latest working duplication glitches"
        "/dupesteps - Give step-by-step guide for selected duplication"
        "/dupehistory - Display history of popular duplication glitches and patches"
        "/help - List all /commands"
        "Most dupes don't use hoppers."
        "Ask user to Google specific dupes."
        "Only respond as DupeBot."
        f"Question: {question}\nAnswer:"
        )
        response = openai.Completion.create(
        engine=model_engine, prompt=prompt, max_tokens=tokens, n=1, stop=None, temperature=temperature
        )
        answer = response.choices[0].text.strip()
        return answer

    def get_chat_gpt_answer(self, api_key):
        question = self.question_entry.get()
        temperature = self.temperature_slider.get()
        tokens = self.tokens_slider.get()
        answer = self.ask_chat_gpt(api_key, question, temperature, tokens)
        self.conversation_history.append((question, answer))  # Add the question and answer to the conversation history
        self.answer_text.insert("end", f"Q: {question}\nA: {answer}\n\n")


    def load_conversation(self):
        try:
            with open("conversation_history.txt", "r") as f:
                conversation_text = f.read()
            self.answer_text.delete(1.0, tk.END)  # Clear the answer_text widget
            self.answer_text.insert("end", conversation_text)
        except FileNotFoundError:
            messagebox.showerror("Error", "No conversation history found.")



  

    def save_conversation(self):
       with open("conversation_history.txt", "w") as f:
           for question, answer in self.conversation_history:
            f.write(f"Q: {question}\nA: {answer}\n\n")


if __name__ == "__main__":
    required = {"webbrowser", "PyGithub", "tk", "openai"}
    installed = {pkg.key for pkg in pkg_resources.working_set}
    missing = required - installed

    if missing:
        show_loading_screen()
    
    root = tk.Tk()
    gui = GitHubSearchGUI(root)
    root.mainloop()
    
    
# Configuration in stuctions for le dupey  Line :numbers assums all are un collapsed 
# Le dupe is a simple tool built for exploring dupes in my free time. It works by taking a github 
# Link and a key word and pooling all relivant results into a window of concise clickable links 
# you are required to provide you own github api key line :  180
# token = "Relace with your own key"
# I have inegrated a simple bot aswell for asking question and providing clearification 
#Requires your ouw Openapi key You must put your api key in the config.py or the bot will not return information 


#Simple  usaeg example log on to a minecraft server use meteor massscan or other method to find plugins or mods
# type the name of the mod followed by github if found it will return a git hup repository that looks like this 
#-https://github.com/GC-spigot/AdvancedEnchantments -
#copy that link into the repostiory field 
#add key words to the text box like dupe or glitch more words return more results but thats not always a good thing 
#your now going to find the plugin on github an exaple plugin here copy the repositor header (whats in the search bar) into the gui add key words and click findem
#now that you have found the repository you wanna search read through the text box and find a dupe you think may work.
#test repeat... test reapeat...
#you can save and load your repositories to the text box for quick access 
#there is also a logo changer with 50 diffrent logo. you can use if you want to showcase or pretend like you made this with
#your friends or youtbe 
#please just credit me in someway whatever though

#this tool was made for established dupe hunters to make it easy to keep track of changes in a plugin aswell as quickly recall and find 
#new dupes related to specific plugins 
#I will probably not be updating this version 
#but you can reach me at Das_search#5957 
#or almost anywhere at Beelzebub4883
