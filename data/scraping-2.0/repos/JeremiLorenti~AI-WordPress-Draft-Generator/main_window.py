import customtkinter as ctk
from settings_window import SettingsWindow
from newPost import on_submit
import threading
import webbrowser
import time
import asyncio
import json
from utils import load_settings  # Import load_settings from utils
from wordpress_api import create_draft_post  # Import create_draft_post from wordpress_api
from bs4 import BeautifulSoup  # Import BeautifulSoup for parsing HTML
from openAI import generate_title  # Import generate_title from openAI
from tkhtmlview import HTMLLabel  # Import HTMLLabel from tkhtmlview
import tkinter as tk  # Import tkinter for Listbox widget
from wordpress_api import get_draft_posts, delete_draft_post  # Import get_draft_posts and delete_draft_post from wordpress_api

def center_window(window, width, height):
    # Get the screen width and height
    screen_width = window.winfo_screenwidth()
    screen_height = window.winfo_screenheight()
    # Calculate the x and y coordinates for the Tk root window
    x = (screen_width // 2) - (width // 2)
    y = (screen_height // 2) - (height // 2)
    # Set the window's geometry 
    window.geometry(f'{width}x{height}+{x}+{y}')

class MainWindow:
    def __init__(self):
        self.root = ctk.CTk()
        self.root.title("AI WordPress Draft Generator")
        self.root.configure(bg="dark gray")
        self.root.tk_setPalette(background='dark gray')
        ctk.set_appearance_mode("Dark")
        ctk.set_default_color_theme("dark-blue")
        self.root.geometry('800x700+50+50')  # Adjusted width from 400 to 800 and height from 600 to 700
        self.settings = load_settings()  # Load settings when MainWindow is initialized
        self.generated_url = ""  # Instance variable to store the generated URL
        self.debug_mode = ctk.BooleanVar()  # Variable to store the state of the debug checkbox
        self.preview_window = None  # Initialize the preview_window attribute
        self.root.after(0, lambda: asyncio.run(self.list_draft_posts()))
        self.create_widgets()

    def create_widgets(self):
        # Settings Button with Gear Icon
        gear_icon = "⚙️"  # Unicode character for a gear icon
        settings_button = ctk.CTkButton(self.root, text=gear_icon, width=40, height=40, corner_radius=10,
                                        command=self.open_settings_window, fg_color="gray")
        settings_button.pack(side='top', anchor='ne', padx=10, pady=10)

        # Welcome label (moved to the top)
        welcome_label = ctk.CTkLabel(self.root, text="Welcome to your Personal WordPress Draft Generator!", font=('Lato', 20, 'bold'))
        welcome_label.pack(side='top', pady=20)

        # Divider after welcome label
        divider1 = ctk.CTkFrame(self.root, height=2, fg_color="gray75")
        divider1.pack(side='top', fill='x', padx=10, pady=5)

        # App description label
        description_label = ctk.CTkLabel(self.root, text="This app helps you create draft posts for your blog with the power of AI. Simply enter the number of articles and let the AI do the rest.", font=("Lato", 10), wraplength=400)
        description_label.pack(side='top', padx=10, pady=5)

        # Divider after app description label
        divider2 = ctk.CTkFrame(self.root, height=2, fg_color="gray75")
        divider2.pack(side='top', fill='x', padx=10, pady=5)

        num_articles_frame = ctk.CTkFrame(self.root)
        num_articles_frame.pack(pady=10)
        num_articles_label = ctk.CTkLabel(num_articles_frame, text="Number of Articles:", font=("Lato", 10))
        num_articles_label.pack(side='left', padx=5)
        self.num_articles = 1
        self.counter = ctk.CTkEntry(num_articles_frame, width=3, font=("Lato", 10))
        self.counter.insert(0, self.num_articles)
        self.counter.pack(side='left')

        # RSS feed selection
        with open('rss_feeds.json') as f:
            rss_feed_options = json.load(f)['feeds']
        self.rss_feed_dropdown = ctk.CTkComboBox(self.root, values=rss_feed_options, width=325)
        self.rss_feed_dropdown.pack()

        # Tone selection checkboxes
        tones = ["Formal", "Friendly", "Professional", "Casual", "Enthusiastic", "Neutral"]
        self.tone_vars = {tone: ctk.BooleanVar() for tone in tones}
        tone_frame = ctk.CTkFrame(self.root)
        tone_frame.pack(pady=20)  # Increase vertical padding for the tone frame

        # Use a grid layout with increased padding
        tone_columns = 3  # Number of columns for the checkboxes
        for i, (tone, var) in enumerate(self.tone_vars.items()):
            checkbox = ctk.CTkCheckBox(tone_frame, text=tone, variable=var)
            checkbox.grid(row=i // tone_columns, column=i % tone_columns, padx=10, pady=5)

        create_button = ctk.CTkButton(self.root, text="Create New Draft", command=self.create_new_draft, font=("Lato", 10))
        create_button.pack(pady=10)

                # Add a button to list draft posts
        self.list_drafts_button = ctk.CTkButton(self.root, text="List Draft Posts", command=lambda: asyncio.run(self.list_draft_posts()))
        self.list_drafts_button.pack(pady=10)

        # Add a Listbox to display draft posts
        self.draft_posts_list = tk.Listbox(self.root)
        self.draft_posts_list.pack(pady=10)

        # Add a button to delete the selected draft post
        self.delete_draft_button = ctk.CTkButton(self.root, text="Delete Selected Draft", command=lambda: asyncio.run(self.delete_selected_draft_post()))
        self.delete_draft_button.pack(pady=10)

        # Progress Indicator Label
        self.progress_label = ctk.CTkLabel(self.root, text="")
        self.progress_label.pack(side='top', fill='x', expand=True)

        self.spinner_label = ctk.CTkLabel(self.root, text="")
        self.spinner_label.pack(side='top', fill='x', expand=True, pady=5)

        # URL Display Label
        self.url_label = ctk.CTkLabel(self.root, text="")
        self.url_label.pack(side='top', fill='x', expand=True)

        # Debug Mode Checkbox
        debug_checkbox = ctk.CTkCheckBox(self.root, text="Debug Mode", variable=self.debug_mode)
        debug_checkbox.pack(side='bottom', anchor='sw', padx=10, pady=10)

        # Instructional text label (moved to the bottom)
        instructions_label = ctk.CTkLabel(self.root, text="Please click the settings button in the top right corner to enter the required information before creating a new draft.", font=("Lato", 10), wraplength=760)
        instructions_label.pack(side='bottom', padx=10, pady=5)

    async def list_draft_posts(self):
        draft_posts = await get_draft_posts()
        if 'error' not in draft_posts:
            self.draft_posts_list.delete(0, tk.END)  # Clear the current list
            for post in draft_posts:
                self.draft_posts_list.insert(tk.END, f"{post['id']} - {post['title']['rendered']}")
        else:
            print(draft_posts['error'])
    async def delete_selected_draft_post(self):
        selection = self.draft_posts_list.curselection()
        if selection:
            post_id = self.draft_posts_list.get(selection[0]).split(' - ')[0]
            result = await delete_draft_post(post_id)
            if 'success' in result:
                self.list_draft_posts()  # Refresh the list after deletion
            else:
                print(result['error'])

    def open_settings_window(self):
        settings_window = SettingsWindow()
        settings_window.show()

    def create_new_draft(self):
        num_articles = self.counter.get()
        if num_articles.isdigit() and int(num_articles) > 0:
            # Initialize the timer
            self.start_time = time.time()
            # Show loading indicator and start the timer
            self.spinner_label.configure(text="Loading... 0 seconds")
            # Start the update_timer method and store the reference
            self.timer_job = self.root.after(1000, self.update_timer)
            # Get selected tones from checkboxes
            selected_tones = [tone for tone, var in self.tone_vars.items() if var.get()]
            # Check if settings is None, if so, initialize with default values
            if self.settings is None:
                self.settings = {'ARTICLE_PREVIEW': False}
            # Call the on_submit function with the number of articles in a separate thread
            threading.Thread(target=self.on_submit_thread, args=(self.spinner_label, num_articles, 1, self.progress_label, None), kwargs={'tones': selected_tones, 'preview_enabled': self.settings.get('ARTICLE_PREVIEW', False)}).start()
        else:
            # Show an error message if the input is not a positive integer
            ctk.messagebox.showerror("Error", "Please enter a valid number of articles.")

    def update_timer(self):
        # Calculate the elapsed time
        elapsed_time = int(time.time() - self.start_time)
        # Update the spinner label with the elapsed time
        self.spinner_label.configure(text=f"Loading... {elapsed_time} seconds")
        # Schedule the update_timer method to be called after 1 second and store the reference
        self.timer_job = self.root.after(1000, self.update_timer)

    def on_submit_thread(self, spinner_label, num_articles, num_paragraphs, progress_label, feedback, tones, preview_enabled):
        selected_rss_feed = self.rss_feed_dropdown.get()  # This gets the selected RSS feed from the dropdown
        # Check if debug mode is enabled
        if self.debug_mode.get():
            # If debug mode is enabled, use static predefined test articles
            result = ["Test Article 1", "Test Article 2", "Test Article 3"]
        else:
            # If debug mode is not enabled, call the on_submit function and get the result
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            result = loop.run_until_complete(on_submit(spinner_label, num_articles, num_paragraphs, progress_label, selected_rss_feed, feedback, tones=tones, preview_enabled=preview_enabled))
        # Stop the timer by canceling the scheduled update_timer method
        if self.timer_job:
            self.root.after_cancel(self.timer_job)
        # Check if the result is an error message
        if isinstance(result, str) and result.startswith("Error:"):
            # Display the error message
            ctk.messagebox.showerror("Error", result)
        else:
            # Check if the article preview feature is enabled in the settings
            if self.settings.get('ARTICLE_PREVIEW', False):
                # Set the article_html attribute with the first article's HTML content
                self.article_html = result[0] if result else ""
                # Generate the article title
                soup = BeautifulSoup(self.article_html, 'html.parser')
                self.article_title = generate_title(soup.get_text())  # Generate title using AI
                # Schedule display_article_preview to be called on the main thread
                self.root.after(0, lambda: self.display_article_preview(self.article_html))
            else:
                # If the article preview feature is not enabled, display the URLs as clickable links
                for post_url in result:
                    spinner_label.configure(text=str(post_url), cursor="hand2")
                    spinner_label.bind("<Button-1>", lambda e: webbrowser.open_new(str(post_url)))

    def display_article_preview(self, article_html):
        # Create a new preview window
        self.preview_window = ctk.CTkToplevel(self.root)
        self.preview_window.title("Article Preview")
        
        # Get the main window's position and size
        main_x = self.root.winfo_x()
        main_y = self.root.winfo_y()
        main_width = self.root.winfo_width()
        main_height = self.root.winfo_height()
        
        # Calculate the center position for the preview window
        preview_width = 800
        preview_height = 600
        x = main_x + (main_width - preview_width) // 2
        y = main_y + (main_height - preview_height) // 2
        
        # Set the preview window position and size
        self.preview_window.geometry(f"{preview_width}x{preview_height}+{x}+{y}")
        
        # Add a HTMLLabel widget to display the article content
        article_html_label = HTMLLabel(self.preview_window, html="<h1>" + self.article_title + "</h1>" + article_html)
        article_html_label.pack(fill='both', expand=True)  # Ensure HTMLLabel expands and fills the preview window
        
        # Add Approve and Disapprove buttons
        approve_button = ctk.CTkButton(self.preview_window, text="Approve", command=lambda: self.handle_article_approval(True, article_html))
        approve_button.pack(side='left', padx=(50, 10), pady=10)
        
        disapprove_button = ctk.CTkButton(self.preview_window, text="Disapprove", command=lambda: self.handle_article_approval(False, article_html))
        disapprove_button.pack(side='right', padx=(10, 50), pady=10)

        # Add Cancel button
        cancel_button = ctk.CTkButton(self.preview_window, text="Cancel", command=self.preview_window.destroy)
        cancel_button.pack(side='right', padx=(50, 10), pady=10)
        
        # Bring the preview window to the front and keep it there
        if self.preview_window is not None:
            self.preview_window.lift()
            self.preview_window.attributes('-topmost', True)  # Keep the window on top
            self.preview_window.focus_force()  # Add this line to force focus on the preview window
            self.preview_window.update()  # Add this line to ensure the preview window is updated before proceeding
        else:
            raise Exception("Preview window is not created.")

    def handle_article_approval(self, approved, article_html):
        if approved:
            # If the user approves, schedule post_article_to_wordpress to be called on the main thread
            self.article_content = article_html
            self.root.after(0, self.post_article_to_wordpress)
        else:
            # If the user disapproves, schedule collect_feedback to be called on the main thread
            self.root.after(0, self.collect_feedback)

    def post_article_to_wordpress(self):
        # Assume that the article data is available as instance variables of the MainWindow class
        post_data = {
            'title': self.article_title,
            'content': self.article_content,
            'status': 'draft'
        }
        # Call the create_draft_post function from wordpress_api.py with the post data
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        self.generated_url = loop.run_until_complete(create_draft_post(post_data))
        
        # Temporarily set the main window as topmost before showing the message box
        self.root.attributes('-topmost', True)
        self.root.update()
        
        # Create Success message window using tkinter.messagebox
        import tkinter.messagebox as msgbox
        
        # Create a success message window using tkinter.messagebox
        msgbox.showinfo("Success", "Article generation was successful!")
        
        # Revert the main window to its normal state (not topmost)
        self.root.attributes('-topmost', False)

        # After message is closed, close the preview window
        self.preview_window.destroy()

        # Update the URL label in the main window and make it clickable
        self.url_label.configure(text=f"Generated URL: {self.generated_url}", cursor="hand2")
        self.url_label.bind("<Button-1>", lambda e: webbrowser.open_new(self.generated_url))

    def collect_feedback(self):
        # Open a feedback window and collect feedback from the user
        feedback_window = ctk.CTkToplevel(self.root)
        feedback_window.title("Provide Feedback")
        feedback_window.geometry('400x200')  # Adjust the size as needed
        # Center the new window in front of the preview window
        center_window(feedback_window, 400, 200)

        # Add a Text widget to collect the feedback
        feedback_label = ctk.CTkLabel(feedback_window, text="Please provide feedback for the AI:", font=("Lato", 10))
        feedback_label.pack(pady=(10, 0))
        self.feedback_entry = ctk.CTkEntry(feedback_window, width=50, font=("Lato", 10))
        self.feedback_entry.pack(pady=(0, 10))

        # Add a Submit button to submit the feedback
        submit_button = ctk.CTkButton(feedback_window, text="Submit Feedback", command=lambda: self.submit_feedback(feedback_window), font=("Lato", 10))
        submit_button.pack(pady=10)

        # Set the feedback window to not be topmost when it is closed
        feedback_window.protocol("WM_DELETE_WINDOW", lambda: self.on_feedback_window_close(feedback_window))

        # Delay setting the feedback window as topmost to ensure it stays on top
        self.root.after(100, lambda: feedback_window.attributes('-topmost', True))

    def on_feedback_window_close(self, feedback_window):
        # Revert the feedback window to its normal state (not topmost)
        feedback_window.attributes('-topmost', False)
        feedback_window.destroy()

    def submit_feedback(self, feedback_window):
        # Get the feedback from the feedback_entry widget
        feedback = self.feedback_entry.get()
        # Close the feedback window
        self.on_feedback_window_close(feedback_window)
        # Show a message indicating that the revised article generation is in progress
        self.spinner_label.configure(text="Your feedback has been received. Generating a revised article...")
        # Call the on_submit function again with the feedback
        threading.Thread(target=self.on_submit_thread, args=(self.spinner_label, self.num_articles, 1, self.progress_label, feedback), kwargs={'tones': self.tone_vars, 'preview_enabled': self.settings.get('ARTICLE_PREVIEW', False)}).start()

    def show(self):
        center_window(self.root, 800, 700)  # Adjusted width from 600 to 800 and height from 600 to 700
        self.root.mainloop()

        # Progress Indicator Label
        self.progress_label = ctk.CTkLabel(self.root, text="")
        self.progress_label.pack(side='top', fill='x', expand=True)

        self.spinner_label = ctk.CTkLabel(self.root, text="")
        self.spinner_label.pack(side='top', fill='x', expand=True)

        # URL Display Label
        self.url_label = ctk.CTkLabel(self.root, text="")
        self.url_label.pack(side='top', fill='x', expand=True)

        # Debug Mode Checkbox
        debug_checkbox = ctk.CTkCheckBox(self.root, text="Debug Mode", variable=self.debug_mode)
        debug_checkbox.pack(side='bottom', anchor='sw', padx=10, pady=10)

        # Instructional text label (moved to the bottom)
        instructions_label = ctk.CTkLabel(self.root, text="Please click the settings button in the top right corner to enter the required information before creating a new draft.", font=("Lato", 10), wraplength=760)
        instructions_label.pack(side='bottom', padx=10, pady=5)

    def open_settings_window(self):
        settings_window = SettingsWindow()
        settings_window.show()

    def create_new_draft(self):
        num_articles = self.counter.get()
        if num_articles.isdigit() and int(num_articles) > 0:
            # Initialize the timer
            self.start_time = time.time()
            # Show loading indicator and start the timer
            self.spinner_label.configure(text="Loading... 0 seconds")
            # Start the update_timer method and store the reference
            self.timer_job = self.root.after(1000, self.update_timer)
            # Get selected tones from checkboxes
            selected_tones = [tone for tone, var in self.tone_vars.items() if var.get()]
            # Check if settings is None, if so, initialize with default values
            if self.settings is None:
                self.settings = {'ARTICLE_PREVIEW': False}
            # Call the on_submit function with the number of articles in a separate thread
            threading.Thread(target=self.on_submit_thread, args=(self.spinner_label, num_articles, 1, self.progress_label, None), kwargs={'tones': selected_tones, 'preview_enabled': self.settings.get('ARTICLE_PREVIEW', False)}).start()
        else:
            # Show an error message if the input is not a positive integer
            ctk.messagebox.showerror("Error", "Please enter a valid number of articles.")

    def update_timer(self):
        # Calculate the elapsed time
        elapsed_time = int(time.time() - self.start_time)
        # Update the spinner label with the elapsed time
        self.spinner_label.configure(text=f"Loading... {elapsed_time} seconds")
        # Schedule the update_timer method to be called after 1 second and store the reference
        self.timer_job = self.root.after(1000, self.update_timer)

    def on_submit_thread(self, spinner_label, num_articles, num_paragraphs, progress_label, feedback, tones, preview_enabled):
        selected_rss_feed = self.rss_feed_dropdown.get()  # This gets the selected RSS feed from the dropdown
        # Check if debug mode is enabled
        if self.debug_mode.get():
            # If debug mode is enabled, use static predefined test articles
            result = ["Test Article 1", "Test Article 2", "Test Article 3"]
        else:
            # If debug mode is not enabled, call the on_submit function and get the result
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            result = loop.run_until_complete(on_submit(spinner_label, num_articles, num_paragraphs, progress_label, selected_rss_feed, feedback, tones=tones, preview_enabled=preview_enabled))
        # Stop the timer by canceling the scheduled update_timer method
        if self.timer_job:
            self.root.after_cancel(self.timer_job)
        # Check if the result is an error message
        if isinstance(result, str) and result.startswith("Error:"):
            # Display the error message
            ctk.messagebox.showerror("Error", result)
        else:
            # Check if the article preview feature is enabled in the settings
            if self.settings.get('ARTICLE_PREVIEW', False):
                # Set the article_html attribute with the first article's HTML content
                self.article_html = result[0] if result else ""
                # Generate the article title
                soup = BeautifulSoup(self.article_html, 'html.parser')
                self.article_title = generate_title(soup.get_text())  # Generate title using AI
                # Schedule display_article_preview to be called on the main thread
                self.root.after(0, lambda: self.display_article_preview(self.article_html))
            else:
                # If the article preview feature is not enabled, display the URLs as clickable links
                for post_url in result:
                    spinner_label.configure(text=str(post_url), cursor="hand2")
                    spinner_label.bind("<Button-1>", lambda e: webbrowser.open_new(str(post_url)))

    def display_article_preview(self, article_html):
        # Create a new preview window
        self.preview_window = ctk.CTkToplevel(self.root)
        self.preview_window.title("Article Preview")
        
        # Get the main window's position and size
        main_x = self.root.winfo_x()
        main_y = self.root.winfo_y()
        main_width = self.root.winfo_width()
        main_height = self.root.winfo_height()
        
        # Calculate the center position for the preview window
        preview_width = 800
        preview_height = 600
        x = main_x + (main_width - preview_width) // 2
        y = main_y + (main_height - preview_height) // 2
        
        # Set the preview window position and size
        self.preview_window.geometry(f"{preview_width}x{preview_height}+{x}+{y}")
        
        # Add a HTMLLabel widget to display the article content
        article_html_label = HTMLLabel(self.preview_window, html="<h1>" + self.article_title + "</h1>" + article_html)
        article_html_label.pack(fill='both', expand=True)  # Ensure HTMLLabel expands and fills the preview window
        
        # Add Approve and Disapprove buttons
        approve_button = ctk.CTkButton(self.preview_window, text="Approve", command=lambda: self.handle_article_approval(True, article_html))
        approve_button.pack(side='left', padx=(50, 10), pady=10)
        
        disapprove_button = ctk.CTkButton(self.preview_window, text="Disapprove", command=lambda: self.handle_article_approval(False, article_html))
        disapprove_button.pack(side='right', padx=(10, 50), pady=10)

        # Add Cancel button
        cancel_button = ctk.CTkButton(self.preview_window, text="Cancel", command=self.preview_window.destroy)
        cancel_button.pack(side='right', padx=(50, 10), pady=10)
        
        # Bring the preview window to the front and keep it there
        if self.preview_window is not None:
            self.preview_window.lift()
            self.preview_window.attributes('-topmost', True)  # Keep the window on top
            self.preview_window.focus_force()  # Add this line to force focus on the preview window
            self.preview_window.update()  # Add this line to ensure the preview window is updated before proceeding
        else:
            raise Exception("Preview window is not created.")

    def handle_article_approval(self, approved, article_html):
        if approved:
            # If the user approves, schedule post_article_to_wordpress to be called on the main thread
            self.article_content = article_html
            self.root.after(0, self.post_article_to_wordpress)
        else:
            # If the user disapproves, schedule collect_feedback to be called on the main thread
            self.root.after(0, self.collect_feedback)

    def post_article_to_wordpress(self):
        # Assume that the article data is available as instance variables of the MainWindow class
        post_data = {
            'title': self.article_title,
            'content': self.article_content,
            'status': 'draft'
        }
        # Call the create_draft_post function from wordpress_api.py with the post data
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        self.generated_url = loop.run_until_complete(create_draft_post(post_data))
        
        # Temporarily set the main window as topmost before showing the message box
        self.root.attributes('-topmost', True)
        self.root.update()
        
        # Create Success message window using tkinter.messagebox
        import tkinter.messagebox as msgbox
        
        # Create a success message window using tkinter.messagebox
        msgbox.showinfo("Success", "Article generation was successful!")
        
        # Revert the main window to its normal state (not topmost)
        self.root.attributes('-topmost', False)

        # After message is closed, close the preview window
        self.preview_window.destroy()

        # Update the URL label in the main window and make it clickable
        self.url_label.configure(text=f"Generated URL: {self.generated_url}", cursor="hand2")
        self.url_label.bind("<Button-1>", lambda e: webbrowser.open_new(self.generated_url))

    def collect_feedback(self):
        # Open a feedback window and collect feedback from the user
        feedback_window = ctk.CTkToplevel(self.root)
        feedback_window.title("Provide Feedback")
        feedback_window.geometry('400x200')  # Adjust the size as needed
        # Center the new window in front of the preview window
        center_window(feedback_window, 400, 200)

        # Add a Text widget to collect the feedback
        feedback_label = ctk.CTkLabel(feedback_window, text="Please provide feedback for the AI:", font=("Lato", 10))
        feedback_label.pack(pady=(10, 0))
        self.feedback_entry = ctk.CTkEntry(feedback_window, width=50, font=("Lato", 10))
        self.feedback_entry.pack(pady=(0, 10))

        # Add a Submit button to submit the feedback
        submit_button = ctk.CTkButton(feedback_window, text="Submit Feedback", command=lambda: self.submit_feedback(feedback_window), font=("Lato", 10))
        submit_button.pack(pady=10)

        # Set the feedback window to not be topmost when it is closed
        feedback_window.protocol("WM_DELETE_WINDOW", lambda: self.on_feedback_window_close(feedback_window))

        # Delay setting the feedback window as topmost to ensure it stays on top
        self.root.after(100, lambda: feedback_window.attributes('-topmost', True))

    def on_feedback_window_close(self, feedback_window):
        # Revert the feedback window to its normal state (not topmost)
        feedback_window.attributes('-topmost', False)
        feedback_window.destroy()

    def submit_feedback(self, feedback_window):
        # Get the feedback from the feedback_entry widget
        feedback = self.feedback_entry.get()
        # Close the feedback window
        self.on_feedback_window_close(feedback_window)
        # Show a message indicating that the revised article generation is in progress
        self.spinner_label.configure(text="Your feedback has been received. Generating a revised article...")
        # Call the on_submit function again with the feedback
        threading.Thread(target=self.on_submit_thread, args=(self.spinner_label, self.num_articles, 1, self.progress_label, feedback), kwargs={'tones': self.tone_vars, 'preview_enabled': self.settings.get('ARTICLE_PREVIEW', False)}).start()






    def show(self):
        center_window(self.root, 800, 700)  # Adjusted width from 600 to 800 and height from 600 to 700
        self.root.mainloop()
