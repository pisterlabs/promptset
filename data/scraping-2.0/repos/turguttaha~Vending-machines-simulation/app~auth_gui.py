import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore
import time
import tkinter as tk
from simulation_lab.vending_machines_simulations import sell_item_from_all_vm
from simulation_lab.vending_machines_simulations import reload_vm
from config.openai_key import *
from ai_module import openai_operations
import asyncio
import threading
import tracemalloc
from app.selection_gui import *

# Initialize Firebase
cred = credentials.Certificate('..\config\chatbot-2c28b-firebase-adminsdk-eoj2u-af1dbe56f8.json')
firebase_admin.initialize_app(cred)
db = firestore.client()


def simulate_vm_operations():

    while True:
        sell_item_from_all_vm()
        # Wait for 1 minutes
        time.sleep(60)


def reload_machines():
    while True:
        reload_vm()
        # Wait for 6 minutes
        time.sleep(12000)


# Enable tracemalloc
tracemalloc.start()

# Create a thread for the background loop
background_thread1 = threading.Thread(target=simulate_vm_operations)
background_thread2 = threading.Thread(target=reload_machines)


# Start the thread
background_thread1.start()
background_thread2.start()


class AuthGUI:
    def __init__(self, root):
        self.root = root
        self.root.title('Auth UI')

        self.current_username = None
        self.current_bedrijfname = None

        # Initialize Inlog UI
        self.init_login_ui()

    def init_login_ui(self):
        # Clear all components in the window
        for widget in self.root.winfo_children():
            widget.destroy()

        # Create Inlog UI
        tk.Label(self.root, text="Username:").grid(row=0, column=0, sticky="e")
        self.login_username = tk.Entry(self.root)
        self.login_username.grid(row=0, column=1, padx=5, pady=5)

        tk.Label(self.root, text="Password:").grid(row=1, column=0, sticky="e")
        self.login_password = tk.Entry(self.root, show="*")
        self.login_password.grid(row=1, column=1, padx=5, pady=5)

        tk.Button(self.root, text="Login", command=self.login).grid(row=2, column=1, sticky="ew", padx=5)
        tk.Button(self.root, text="Register", command=self.init_signup_ui).grid(row=3, column=1, sticky="ew", padx=5)
        tk.Button(self.root, text="Forgot Password", command=self.init_forgot_password_ui).grid(row=4, column=1,
                                                                                                sticky="ew", padx=5)

    def init_signup_ui(self):
        # Clear all components in the window
        for widget in self.root.winfo_children():
            widget.destroy()

        # Create Singnup UI
        tk.Label(self.root, text="Email:").grid(row=0, column=0, sticky="e")
        self.signup_email = tk.Entry(self.root)
        self.signup_email.grid(row=0, column=1, padx=5, pady=5)

        tk.Label(self.root, text="Username:").grid(row=1, column=0, sticky="e")
        self.signup_username = tk.Entry(self.root)
        self.signup_username.grid(row=1, column=1, padx=5, pady=5)

        tk.Label(self.root, text="Password:").grid(row=2, column=0, sticky="e")
        self.signup_password = tk.Entry(self.root, show="*")
        self.signup_password.grid(row=2, column=1, padx=5, pady=5)

        tk.Label(self.root, text="BedrijfName:").grid(row=3, column=0, sticky="e")
        self.signup_bedrijf_name = tk.Entry(self.root)
        self.signup_bedrijf_name.grid(row=3, column=1, padx=5, pady=5)

        tk.Button(self.root, text="Confirm Registration", command=self.signup).grid(row=4, column=1, sticky="ew",
                                                                                    padx=5)
        tk.Button(self.root, text="Back to Login", command=self.init_login_ui).grid(row=5, column=1, sticky="ew",
                                                                                    padx=5)

    def init_forgot_password_ui(self):
        # Clear all components in the window
        for widget in self.root.winfo_children():
            widget.destroy()

        # Create Forgot Password UI
        tk.Label(self.root, text="Email:").grid(row=0, column=0, sticky="e")
        self.forgot_password_email = tk.Entry(self.root)
        self.forgot_password_email.grid(row=0, column=1, padx=5, pady=5)

        tk.Button(self.root, text="Reset Password", command=self.reset_password).grid(row=1, column=1, sticky="ew",
                                                                                      padx=5)
        tk.Button(self.root, text="Back to Login", command=self.init_login_ui).grid(row=2, column=1, sticky="ew",
                                                                                    padx=5)

    def signup(self):
        email = self.signup_email.get()
        username = self.signup_username.get()
        password = self.signup_password.get()
        bedrijfname = self.signup_bedrijf_name.get()

        # Create user record in Firestore
        users_ref = db.collection(u'users')

        try:
            # First check if the username already exists
            query_ref = users_ref.where('username', '==', username).limit(1)
            docs = query_ref.stream()
            if next(docs, None):
                messagebox.showerror("Registration Failed", "Username already exists.")
            else:
                users_ref.add({
                    u'email': email,
                    u'username': username,
                    u'password': password,
                    u'bedrijfname': bedrijfname
                })
                messagebox.showinfo("Registration Successful", "You have been registered.")
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def login(self):
        username = self.login_username.get()
        password = self.login_password.get()

        # Query user records in Firestore
        users_ref = db.collection(u'users')
        query_ref = users_ref.where('username', '==', username).limit(1)

        try:
            docs = query_ref.stream()
            user_doc = next(docs, None)
            if user_doc:
                user_info = user_doc.to_dict()
                if user_info['password'] == password:
                    messagebox.showinfo("Login Successful", "You are now logged in.")
                    self.current_username = username
                    self.current_bedrijfname = self.get_bedrijfname(username)
                    self.open_selection_ui()
                else:
                    messagebox.showerror("Login Failed", "Incorrect password.")
            else:
                messagebox.showerror("Login Failed", "Username does not exist.")
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def get_bedrijfname(self, username):
        users_ref = db.collection(u'users')
        try:
            query_ref = users_ref.where('username', '==', username).limit(1)
            docs = query_ref.stream()
            doc = next(docs, None)
            if doc:
                # Assume that the document has a 'bedrijfname' field.
                return doc.to_dict().get('bedrijfname')
            else:
                # Handle the case where the user does not exist.
                messagebox.showerror("Error", "User does not exist.")
        except Exception as e:
            messagebox.showerror("Error", str(e))
            return None

    def reset_password(self):
        email = self.forgot_password_email.get()
        try:
            messagebox.showinfo("Reset Password", "Password reset email has been sent.")
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def open_selection_ui(self):
        selection_ui = SelectionGUI(self.root, self.current_username, self.current_bedrijfname)
        return selection_ui


if __name__ == "__main__":
    root = tk.Tk()
    app = AuthGUI(root)
    root.mainloop()
