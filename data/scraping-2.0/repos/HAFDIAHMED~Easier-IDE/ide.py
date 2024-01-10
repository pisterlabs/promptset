import os
from tkinter import Tk, Text, Scrollbar, Menu, filedialog, Listbox, END, SINGLE, messagebox
import openai
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from pygments import lex
from pygments.lexers import PythonLexer
from pygments.formatters import HtmlFormatter
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

class SimpleIDE:
    def __init__(self, root):
        self.root = root
        self.root.title("Easier IDE")

        # Set window icon
        icon_path = os.path.join(os.path.dirname(__file__), "assets", "easier.png")
        if os.path.exists(icon_path):
            self.root.iconbitmap(icon_path)

        self.documents_listbox = Listbox(root, selectmode=SINGLE)
        self.documents_listbox.pack(side="left", fill="y")

        self.documents_scroll = Scrollbar(root, orient="vertical", command=self.documents_listbox.yview)
        self.documents_scroll.pack(side="left", fill="y")
        self.documents_listbox.configure(yscrollcommand=self.documents_scroll.set)

        self.text = Text(root, wrap="word", undo=True)
        self.text.pack(expand="yes", fill="both")

        self.scroll = Scrollbar(root, orient="vertical", command=self.text.yview)
        self.scroll.pack(side="right", fill="y")
        self.text.configure(yscrollcommand=self.scroll.set)

        # Configure the text widget to use the scrollbar
        self.scroll.config(command=self.text.yview)

        self.menu = Menu(root)
        root.config(menu=self.menu)

        self.file_menu = Menu(self.menu, tearoff=False)
        self.menu.add_cascade(label="File", menu=self.file_menu)
        self.file_menu.add_command(label="New", command=self.new_file)
        self.file_menu.add_command(label="Open", command=self.open_file)
        self.file_menu.add_command(label="Save", command=self.save_file)
        self.file_menu.add_command(label="Save As", command=self.save_as_file)
        self.file_menu.add_separator()
        self.file_menu.add_command(label="Exit", command=self.exit_app)

        self.debug_menu = Menu(self.menu, tearoff=False)
        self.menu.add_cascade(label="Debug", menu=self.debug_menu)
        self.debug_menu.add_command(label="Debug with ChatGPT", command=self.debug_with_chatgpt)
        self.debug_menu.add_command(label="Debug with AI", command=self.debug_opened_file)

        self.help_menu = Menu(self.menu, tearoff=False)
        self.menu.add_cascade(label="Help", menu=self.help_menu)
        self.help_menu.add_command(label="About", command=self.show_about)

        self.open_files = []  # List to store open file paths
        self.current_file = None

        # Bind double-click event on the documents listbox to open the selected file
        self.documents_listbox.bind("<Double-1>", lambda event: self.open_selected_file())

        # Bind Ctrl+S to save_file
        self.root.bind('<Control-s>', lambda event: self.save_file())
        # Set your OpenAI API key
        openai.api_key = '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'

        # Initialize an empty model
        self.model = make_pipeline(TfidfVectorizer(), MultinomialNB())
        self.scenario_model = make_pipeline(TfidfVectorizer(), MultinomialNB())
        # Create an output area at the bottom
        self.output_area = Text(root, wrap="word", height=5, state="disabled")
        self.output_area.pack(side="bottom", fill="x")
        self.output_area_title = "Imagined Scenario"
        self.output_area.insert("insert", f"{self.output_area_title}\n\n")
        self.output_area.config(state="disabled")
        # Training data and labels
        self.training_data = []
        self.labels = []
        # Training Data (code snippets with errors)
        training_code_1 = "for i in range(10):\n    print(i)"
        training_code_2 = "print('Hello, World!'"
        training_code_3 = "x = 5\nif x > 0:\n    print('Positive')"

        # Labels (indicating whether there is a syntax error or not)
        label_1 = 0  # No syntax error in training_code_1
        label_2 = 1  # Syntax error in training_code_2
        label_3 = 0  # No syntax error in training_code_3

        # ... add more code snippets ...

        # Append training data and labels to the lists
        self.training_data.append(training_code_1)
        self.labels.append(label_1)

        self.training_data.append(training_code_2)
        self.labels.append(label_2)

        self.training_data.append(training_code_3)
        self.labels.append(label_3)

        # ... add more data ...

        # Configure tags for syntax highlighting
        self.configure_syntax_tags()


        # Configure tags for syntax highlighting
        self.configure_syntax_tags()

    def configure_syntax_tags(self):
    # Configure tags for different syntax elements
        formatter = HtmlFormatter(style="friendly")  # You can change the style as needed
        style_defs = formatter.get_style_defs()
        self.text.tag_configure("pygments", lmargin1=10, background="white", font=("Courier New", 10))

        # Apply styles to the text widget
        self.text.insert("1.0", " ", ("pygments",) + (style_defs,))

    def apply_syntax_highlighting(self):
        code_content = self.text.get(1.0, "end-1c")
        tokens = lex(code_content, PythonLexer())  # Change PythonLexer to the desired lexer
        for token, value in tokens:
            start_line, start_char = token.start
            end_line, end_char = token.end
            start_pos = f"{start_line}.{start_char}"
            end_pos = f"{end_line}.{end_char}"
            self.text.tag_add(str(token), start_pos, end_pos)

    def generate_scenario(self, code_content):
        # Fit the scenario model with training data before calling predict
        if self.training_data and self.labels:
            self.scenario_model.fit(self.training_data, self.labels)

            # Now you can call predict for scenario generation
            scenario_prediction = self.scenario_model.predict([code_content])

            # Map the label to a meaningful scenario description
            if scenario_prediction[0] == 1:
                scenario_output = "Scenario: Syntax error is present in the code."
            else:
                scenario_output = "Scenario: Code is error-free."

        else:
            scenario_output = "No training data available for scenario generation."

        # Insert the scenario output into the output area
        self.output_area.config(state="normal")  # Enable editing
        self.output_area.delete(1.0, "end")  # Clear existing content
        self.output_area.insert("insert", scenario_output)  # Insert new content
        self.output_area.config(state="disabled")  # Disable editing

    def debug_opened_file(self):
        if self.current_file:
            with open(self.current_file, "r") as file:
                file_content = file.read()
                self.generate_scenario(file_content)
        else:
            self.output_area.config(state="normal")
            self.output_area.delete(1.0, "end")
            self.output_area.insert("insert", "No file is currently open.")
            self.output_area.config(state="disabled")


            

    def debug_code(self):
        # Fit the model with training data before calling predict
        if self.training_data and self.labels:
            self.model.fit(self.training_data, self.labels)

            # Now you can call predict
            code_text = self.text.get(1.0, "end-1c")
            prediction = self.model.predict([code_text])
            messagebox.showinfo("Debugging Result", f"Is there a syntax error? {'Yes' if prediction[0] == 1 else 'No'}")
        else:
            messagebox.showinfo("Debugging Result", "No training data available.")

    def get_file_content(self, file_path):
        with open(file_path, "r") as file:
            return file.read()

    def train_model(self):
        # Check if there is training data
        if not self.training_data or not self.labels:
            messagebox.showinfo("Training Result", "No training data available.")
            return

        # Split the data into training and testing sets
        train_texts, test_texts, train_labels, test_labels = train_test_split(
            self.training_data, self.labels, test_size=0.2, random_state=42
        )

        # Fit the model on the training data
        self.model.fit(train_texts, train_labels)

        # Predict on the test set
        test_predictions = self.model.predict(test_texts)

        # Evaluate the model
        accuracy = accuracy_score(test_labels, test_predictions)
        messagebox.showinfo("Training Result", f"Model accuracy: {accuracy}")


    def debug_with_chatgpt(self):
        if self.current_file:
            with open(self.current_file, "r") as file:
                file_content = file.read()
                debug_output = self.call_chatgpt(file_content)

                # Display the ChatGPT debug output in a messagebox
                messagebox.showinfo("Debug Output", debug_output)

    def call_chatgpt(self, code):
        # Call OpenAI API for debugging
        response = openai.Completion.create(
            engine="text-davinci-002",  # You can use other engines
            prompt=code,
            max_tokens=100
        )

        # Extract the generated text from ChatGPT response
        debug_output = response.choices[0].text.strip()

        return debug_output

    def set_window_title(self, name=None):
        title = "Easier IDE"
        if name:
            title += f" - {name}"
        self.root.title(title)

    def new_file(self):
        self.text.delete(1.0, "end")
        self.current_file = None
        self.set_window_title()

    def open_file(self):
        file_path = filedialog.askopenfilename()
        if file_path:
            self.text.delete(1.0, "end")
            with open(file_path, "r") as file:
                self.text.insert("insert", file.read())
            self.current_file = file_path
            self.open_files.append(file_path)
            self.update_documents_list()

    def save_file(self, event=None):
        if self.current_file:
            with open(self.current_file, "w") as file:
                file.write(self.text.get(1.0, "end-1c"))
        else:
            self.save_as_file()

    def save_as_file(self):
        file_path = filedialog.asksaveasfilename()
        if file_path:
            with open(file_path, "w") as file:
                file.write(self.text.get(1.0, "end-1c"))
            self.current_file = file_path
            self.open_files.append(file_path)
            self.update_documents_list()

    def open_selected_file(self):
        selected_index = self.documents_listbox.curselection()
        if selected_index:
            selected_index = int(selected_index[0])
            selected_file = self.open_files[selected_index]
            if selected_file and os.path.exists(selected_file):
                with open(selected_file, "r") as file:
                    code_content = file.read()
                    self.text.delete(1.0, "end")
                    self.text.insert("insert", code_content)
                    # Apply syntax highlighting
                    self.apply_syntax_highlighting()

                self.current_file = selected_file
                self.set_window_title(os.path.basename(selected_file))

    def apply_syntax_highlighting(self):
        code_content = self.text.get(1.0, "end-1c")
        tokens = lex(code_content, PythonLexer())  # Change PythonLexer to the desired lexer
        for token, value in tokens:
            self.text.tag_add(str(token), f"{token.start[0]}.{token.start[1]}", f"{token.end[0]}.{token.end[1]}")


    def update_documents_list(self):
        self.documents_listbox.delete(0, END)
        for file_path in self.open_files:
            self.documents_listbox.insert(END, os.path.basename(file_path))

    def exit_app(self):
        self.root.destroy()

    def show_about(self):
        about_text = "Easier IDE\n\nA simple text editor with basic functionality.\nVersion: 1.0\n\nDeveloped by WebSolvus for debugging code with machine learning."
        messagebox.showinfo("About", about_text)

if __name__ == "__main__":
    root = Tk()
    app = SimpleIDE(root)
    # Train the model (add this as needed)
    app.train_model()
    root.mainloop()