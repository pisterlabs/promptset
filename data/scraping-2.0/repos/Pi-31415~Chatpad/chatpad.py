import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox
import openai
import os
import threading

# Set up the OpenAI API key
# openai.api_key = api_key
openai.api_key = "ADD KEY HERE"

class Notepad:
    def __init__(self, root):
        self.root = root
        self.root.title("Notepad")

        # Set the window icon (only works on Windows)
        self.root.iconbitmap('Notepad_23093.ico')  # Replace 'Notepad_23093.ico' with the path to your icon file

        self.textArea = tk.Text(self.root)
        self.textArea.pack(expand=True, fill='both')

        self.entry = tk.Entry(self.root)
        self.entry.pack(fill=tk.X)
        self.entry.bind("<Return>", self.send_message)

        self.menuBar = tk.Menu(self.root)

        # File Menu
        self.fileMenu = tk.Menu(self.menuBar, tearoff=0)
        self.fileMenu.add_command(label="New", command=self.newFile)
        self.fileMenu.add_command(label="Open", command=self.openFile)
        self.fileMenu.add_command(label="Save", command=self.saveFile)
        self.fileMenu.add_separator()
        self.fileMenu.add_command(label="Exit", command=self.quitApplication)
        self.menuBar.add_cascade(label="File", menu=self.fileMenu)

        # Edit Menu (No functionality yet)
        self.editMenu = tk.Menu(self.menuBar, tearoff=0)
        self.editMenu.add_command(label="Undo")
        self.editMenu.add_command(label="Cut")
        self.editMenu.add_command(label="Copy")
        self.editMenu.add_command(label="Paste")
        self.menuBar.add_cascade(label="Edit", menu=self.editMenu)

        # View Menu (No functionality yet)
        self.viewMenu = tk.Menu(self.menuBar, tearoff=0)
        self.viewMenu.add_command(label="Zoom")
        self.viewMenu.add_command(label="Status Bar")
        self.menuBar.add_cascade(label="View", menu=self.viewMenu)

        self.root.config(menu=self.menuBar)
        self.textArea.pack(expand=True, fill='both')

    def call_api(self, message):
        response = self.simulate_response(message)
        self.update_text_area(response)


    def update_text_area(self, response):
        # Ensure that GUI updates are done in the main thread
        if self.root:
            self.root.after(0, self.insert_response, response)

    def insert_response(self, response):
        self.textArea.insert(tk.END, f"{response}\n\n")
        self.textArea.see(tk.END)
        
    def send_message(self, event=None):
        message = self.entry.get()
        if message:
            if message.startswith("GAMS:"):
                # Remove 'GAMS:' from the message and add the specified prompt
                prompt = r"""In all the outputs, only output the GAMS code only, well indented and formatted. Do not output anything else.
                
                I will give you a problem to solve, solve it using GAMS and write GAMS code.
                
                As an example, below are two sample problems and GAMS solution.
                
                Problem 1 : 
                \begin{aligned}
                & \max 109 X_{\text {corn }}+90 X_{\text {wheat }}+115 X_{\text {cotton }} \\
                & \text { s.t. } X_{\text {corn }}+X_{\text {wheat }}+X_{\text {cotton }} \leq 100 \text { (land) } \\
                & 6 X_{\text {corn }}+4 X_{\text {wheat }}+8 X_{\text {cotton }} \leq 500 \text { (labor) } \\
                & X_{\text {corn }} \geq 0 \quad \text { (nonnegativity) } \\
                & X_{\text {wheat }} \geq 0 \quad \text { (nonnegativity) } \\
                & X_{\text {cotton }} \geq 0 \quad \text { (nonnegativity) } \\
                &
                \end{aligned}
                
                GAMS Code for Problem 1:
                
                Positive Variables    Xcorn, Xwheat, Xcotton;
                Variables             Z;

                Equations     obj, land, labor;

                obj..  Z =e= 109 * Xcorn + 90 * Xwheat + 115 * Xcotton;
                land..             Xcorn +      Xwheat +       Xcotton =l= 100;
                labor..        6 * Xcorn +  4 * Xwheat +   8 * Xcotton =l= 500;

                Model farmproblem / obj, land, labor /;

                solve farmproblem using LP maximizing Z;
                
                
                
                Problem 2:
                Table 1: Data for the transportation problem (adapted from Dantzig, 1963) illustrates Shipping Distances from Plants to Markets (1000 miles) as well as Market Demands and Plant Supplies.
                
                \begin{array}{|l|r|r|r|r|}
                \hline \text { Plants } \downarrow & \text { New York } & \text { Chicago } & \text { Topeka } & \leftarrow \text { Markets } \\
                \hline \text { Seattle } & 2.5 & 1.7 & 1.8 & 350 \\
                \hline \text { San Diego } & 2.5 & 1.8 & 1.4 & 600 \\
                \hline \text { Demands } \rightarrow & 325 & 300 & 275 & \text { Supplies } \uparrow \\
                \hline
                \end{array}
                
                As an instance of the transportation problem, suppose there are two canning plants and three markets, with the data given in table Table 1. Shipping distances are in thousands of miles, and shipping costs are assumed to be $90.00 per case per thousand miles. The GAMS representation of this problem is as follows:
                
                $title a transportation model
                Sets
                    i   canning plants   / seattle, san-diego /
                    j   markets          / new-york, chicago, topeka / ;

                Parameters

                    a(i)  capacity of plant i in cases
                    /    seattle     350
                            san-diego   600  /

                    b(j)  demand at market j in cases
                    /    new-york    325
                            chicago     300
                            topeka      275  / ;

                Table d(i,j)  distance in thousands of miles
                                new-york       chicago      topeka
                    seattle          2.5           1.7          1.8
                    san-diego        2.5           1.8          1.4  ;

                Scalar f  freight in dollars per case per thousand miles  /90/ ;

                Parameter c(i,j)  transport cost in thousands of dollars per case ;

                        c(i,j) = f * d(i,j) / 1000 ;

                Variables
                    x(i,j)  shipment quantities in cases
                    z       total transportation costs in thousands of dollars ;

                Positive Variable x ;

                Equations
                    cost        define objective function
                    supply(i)   observe supply limit at plant i
                    demand(j)   satisfy demand at market j ;

                cost ..        z  =e=  sum((i,j), c(i,j)*x(i,j)) ;

                supply(i) ..   sum(j, x(i,j))  =l=  a(i) ;

                demand(j) ..   sum(i, x(i,j))  =g=  b(j) ;

                Model transport /all/ ;

                Solve transport using lp minimizing z ;

                Display x.l, x.m ;
                
                ----------------------------------------
                The actual problem to solve is as follows:
                -----------------------------------"""
                modified_message = prompt + message[5:]  # Remove 'GAMS:' (first 5 characters)
            else:
                # Send the message without any additional prompt
                modified_message = message

            # Clear the message input
            self.entry.delete(0, tk.END)
            # Start a new thread for the API call
            threading.Thread(target=self.call_api, args=(modified_message,)).start()
            return "break"  # Prevents default Enter key behavior (newline)

    def simulate_response(self, message):
        """
        Get a response from OpenAI's GPT model based on the user's message.
        """
        try:
            response = openai.ChatCompletion.create(
                model="gpt-4",  # You can choose a different model as needed
                messages=[
                    {"role": "user", "content": message}
                ]
            )

            # Extract and return the response content
            return response.choices[0].message['content'].strip()
        except Exception as e:
            print(f"Error: {e}")
            return "There was an error in processing the request."
        
    def newFile(self):
        self.root.title("Untitled - Notepad")
        self.file = None
        self.textArea.delete(1.0, tk.END)

    def openFile(self):
        self.file = filedialog.askopenfilename(defaultextension=".txt",
                                               filetypes=[("All Files", "*.*"),
                                                          ("Text Documents", "*.txt")])
        if self.file == "":
            self.file = None
        else:
            self.root.title(f"{self.file} - Notepad")
            self.textArea.delete(1.0, tk.END)
            with open(self.file, "r") as f:
                self.textArea.insert(1.0, f.read())

    def saveFile(self):
        if self.file is None:
            self.file = filedialog.asksaveasfilename(initialfile='Untitled.txt',
                                                     defaultextension=".txt",
                                                     filetypes=[("All Files", "*.*"),
                                                                ("Text Documents", "*.txt")])
            if self.file == "":
                self.file = None
            else:
                with open(self.file, "w") as f:
                    f.write(self.textArea.get(1.0, tk.END))
                self.root.title(f"{self.file} - Notepad")
        else:
            with open(self.file, "w") as f:
                f.write(self.textArea.get(1.0, tk.END))

    def quitApplication(self):
        self.root.destroy()

if __name__ == '__main__':
    root = tk.Tk()
    notepad = Notepad(root)
    root.mainloop()
