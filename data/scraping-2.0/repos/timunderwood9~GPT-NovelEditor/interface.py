import tkinter as tk
from tkinter import ttk
import tkinter.font as tkFont
from tkinter import messagebox
from tkinter import filedialog
from functools import partial
import json
from utility_functions import token_length
from project_class import Project
import openai
import asyncio
from dotenv import load_dotenv
import os
import threading

#testing function
def load_project():
    script_dir = os.path.dirname(os.path.abspath(__file__))

    options = {
        'defaultextension': '.json',
        'filetypes': [('JSON files', '.json')],
        'initialdir': script_dir,  # Set initial directory to "saves" subdirectory
        'title': 'Open JSON File',
    }

    filename = filedialog.askopenfilename(**options)
    
    if filename:
        global PROJECT, main_interface, loading_page2
        PROJECT = Project.load(filename)
        print(PROJECT.api_key)
        print(PROJECT.chapters)
        destroy_widgets(root)
        loading_page2 = LoadingPage2(root)
        destroy_widgets(root)
        main_interface = MainInterface(root)
        main_interface.editing_frame.build_display_box()
        main_interface.notebook.select(main_interface.editing_frame)

def _go_to_current_position(event):
    load_dotenv()
    load_dotenv(r"C:\Users\Dell Latitude 7400\OneDrive\Documents\GitHub\timunderwood-private\python.env")
    PROJECT.api_key = os.getenv('OPENAI_API_KEY')


    PROJECT.title = "The Missing Prince"
    PROJECT.divided = False
    with open ('test_novel.txt', 'r', encoding='utf-8') as f:
        text = f.read()

#    PROJECT.project_text = text[1000:20000]
    PROJECT.project_text = text

def clear_rate_limit():
    global rate_limit_hit
    rate_limit_hit = False

def show_error(error_message):
    error_window = tk.Tk()
    error_window.withdraw()
    messagebox.showerror("Error", error_message)
    error_window.destroy()  # Destroy the main window

def asyncio_loop_cycle(loop):
    loop.stop()
    loop.run_forever()
    root.after(100, asyncio_loop_cycle, loop)

def wrap_with_font(font, text, max_width = 600):
    words = text.split()
    wrapped_lines = []
    line = ""
    for word in words:
        temp_line = line + " " + word if line else word

        if font.measure(temp_line) <= max_width:
            line = temp_line
        else:
            wrapped_lines.append(line)
            line=word
    wrapped_lines.append(line)
    return "\n".join(wrapped_lines)

def print_root_children():
    print(root.winfo_children())

def quit_program(event):
    root.quit()

def return_root_single_child():
    children = root.winfo_children()
    if len (children) == 0:
        print('There are no children')
    elif len (children) > 1:
        print('There is more than 1 child')
    else:
        return children[0]

def quit_program(event):
    root.quit()

def select_tab_by_label(label_text, notebook):
    for index, tab in enumerate(notebook.tabs()):
        if notebook.tab(tab, "text") == label_text:
            notebook.select(index)
            break

def destroy_widgets(master):
    for widget in master.winfo_children():
            widget.destroy()

def deactivate_window_scrollbar(event):
    root.unbind_all("<MouseWheel>")


class OnSubmit:
    
    def save_text(entered_text, data_label):
        json_data = {
            data_label: entered_text
        }

        with open('save_data.json', 'w') as file:
            json.dump(json_data, file)


    def title(frame, entry_box):
        title = entry_box.get()
        frame.pack_forget()
        return Project.create_project(title)

    def text_box(frame, text_box):
        entered_text = text_box.get("1.0", tk.END)
        text_token_length = token_length(entered_text)

        

        for widget in frame.winfo_children():
            widget.pack_forget()
        new_label = tk.Label(frame, text= "We've saved your text")
        new_label.pack()
        new_label2 = tk.Label(frame, text=f"The text is {text_token_length} tokens long")
        new_label2.pack()

        edit_button = tk.Button(frame, text='Edit', command=restore_frame)
        edit_button.pack()

def restore_frame(frame):
    pass

class AutoHideScrollbar(tk.Scrollbar):
    def __init__(self, master, **kwargs):
        tk.Scrollbar.__init__(self, master, **kwargs)
        self.pack_forget()

    def set(self, lo, hi):
        if float(lo) <= 0.0 and float(hi) >= 1.0:
            self.pack_forget()
        else:
            self.pack(side="right", fill="y")
        tk.Scrollbar.set(self, lo, hi)

class LoadingPage:
    def __init__(self, master):
        self.frame = tk.Frame(master)
        new_project_button = tk.Button(self.frame, bd=5, text="New Project", command = self.enter_project_title)
        new_project_button.pack()
        load_button = tk.Button(self.frame, bd = 5, text="Load", command=load_project)
        load_button.pack()
        exit_button = tk.Button(self.frame, bd = 5, text="Exit", command = self.exit_app)
        exit_button.pack()
        self.frame.pack()

    #This function takes the entered text and creates the new project
    def new_project(title, window, entry_box, event):
        global PROJECT
        title=entry_box.get()
        PROJECT = Project.create_project(title)
        PROJECT.divided = False
        window.destroy()
        global loading_page2
        loading_page2 = LoadingPage2(root)

    def enter_project_title(self):
        entry_window = tk.Toplevel(root)
        entry_window.focus_set()

        label = tk.Label(entry_window, text='Title:')
        label.pack(side=tk.LEFT)
        entry_box = tk.Entry(entry_window, width=100)
        entry_box.pack(side=tk.LEFT)
        entry_box.bind("<Return>", partial(self.new_project, entry_window, entry_box))
        entry_box.focus_set()

    def exit_app(self):
        root.quit()
        
class LoadingPage2:
    def __init__(self, master):
        destroy_widgets(master)
        
        self.frame = tk.Frame(master)
        self.title = PROJECT.title
        title_label = self.create_title_label(self.title)
        

        
        input_details_button = tk.Button(self.frame, text='Input Novel Details', command=self.load_input_page)
        editing_button = tk.Button(self.frame, text='Use editing and summarization prompts', command=self.load_editing_page)
        
        title_label.pack()
        input_details_button.pack()
        editing_button.pack()
        self.frame.pack()

    def create_title_label(self, text):
        bold_font = tkFont.Font(family="Helvetica", size=14, weight="bold")
        return tk.Label(self.frame, text=text, font=bold_font, relief='ridge', borderwidth=10)
    
    def load_input_page(self):
        destroy_widgets(root)
        global main_interface
        main_interface = MainInterface(root)
        main_interface.editing_frame.build_display_box()
        main_interface.notebook.select(main_interface.input_frame)

    def load_editing_page(self):
        destroy_widgets(root)
        global main_interface
        main_interface = MainInterface(root)
        main_interface.editing_frame.build_display_box()
        main_interface.notebook.select(main_interface.editing_frame)

#currently defined to only take 'textbox' and 'entrybox' as types.
class CustomTextBox:
    def __init__(self, master, property = "", object = None, label = "", widget_type = 'textbox', submitted_text = "Your entry has been saved:\n{}", default_text=None):
        self.frame = tk.Frame(master, borderwidth=5, relief=tk.RIDGE)
        self.frame.pack()
        self.object = object

        self.label_text = label
        self.property_name = property
        self.submitted_text = submitted_text
        self.widget_type = widget_type
        self.submit_button_text = "Save"
        self.default_text=default_text

        self.build_inner_widgets()

    def deactivate_window_scrollbar(self, event):
        root.unbind_all("<MouseWheel>")

    def submit_command(self):
        if self.widget_type == 'textbox':
            new_property_value = self.textbox.get("1.0", tk.END).strip() # Get text from Text widget
        elif self.widget_type == 'entrybox':
            new_property_value = self.entrybox.get()

        setattr(self.object, self.property_name, new_property_value)
        print(getattr(self.object, self.property_name))

        destroy_widgets(self.frame)
        #Limiting the length of the printed portion of the entry and adding a '...' for formatting reasons
        if len(new_property_value) > 200: new_property_value_string = new_property_value[0:200] + '...'
        else: new_property_value_string = new_property_value

        self.label = tk.Label(self.frame, height=5, text = self.submitted_text.format(new_property_value_string))
        self.label.pack(side=tk.LEFT)

        self.edit_button = tk.Button(self.frame, text = 'Edit', command=self.edit_command)
        self.edit_button.pack(side=tk.LEFT)
    
    def edit_command(self):
        destroy_widgets(self.frame)
        self.build_inner_widgets()

    def build_inner_widgets(self):
        self.label = tk.Label(self.frame, text=self.label_text)
        self.label.pack()

        if self.widget_type == 'textbox':
            self.textbox = tk.Text(self.frame, height=5, wrap=tk.WORD)
            self.textbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
            existing_text = getattr(PROJECT, self.property_name, None)
            if existing_text:
                self.textbox.insert(1.0, existing_text)

            self.textbox.bind("<FocusIn>", deactivate_window_scrollbar)

        elif self.widget_type == 'entrybox':
            self.entrybox = tk.Entry(self.frame, width=40)
            self.entrybox.pack(side=tk.LEFT)

        elif self.widget_type == 'Text':
            self.text = tk.Text (self.frame, height=5, wrap='word')
            self.text.insert('1.0', self.default_text)
            self.text.configure(state='disabled')
            self.text.pack(side=tk.LEFT)


        else: print('Widget type not defined')

        submit_button = tk.Button(self.frame, text=self.submit_button_text, command=self.submit_command)
        submit_button.pack()

    
    def change_submitted_text(self, new_text):
        self.submitted_text = new_text

class EditAndRestoreBox:
    def __init__ (self, master, text, label_text = None, property = '', object = None, width = None, height = 5):
        self.frame = tk.Frame(master)
        self.frame.pack()
        self.default_text = text
        self.current_text = text
        self.property = property
        self.object = object
        self.update_property(self.default_text)
        self.width = width
        self.height = height
        
        if label_text:
            self.label = tk.Label(self.frame, text = label_text)
            self.label.pack()

        self.generate_prompt_display(self.current_text)

    def update_property (self, new_value):
        setattr(self.object, self.property, new_value)
        #code to set in case it is api_key, I feel like better practice wouldn't have this update here, but I'm not sure how to switch it without lots of extra code otherwise
        if self.property == 'api_key':
            openai.api_key = PROJECT.api_key
    
    def generate_prompt_display(self, prompt_text):
        if self.width:
            self.prompt_display = tk.Text(self.frame, height = self.height, width=self.width, wrap = 'word')
        else:
            self.prompt_display = tk.Text(self.frame, height = self.height, wrap = 'word')
        self.prompt_display.pack()
        self.prompt_display.insert('1.0', prompt_text)
        self.prompt_display.configure(state='disabled')
        
        self.button_frame = tk.Frame(self.frame)
        self.button_frame.pack()

        self.edit_button = tk.Button(self.button_frame, text='Edit', command=self.on_edit_press)
        self.save_button = tk.Button(self.button_frame, text = 'Save', command=self.on_save)
        self.reset_button = tk.Button(self.button_frame, text = 'Reset', command=self.on_reset)
        self.edit_button.pack(side = tk.LEFT)
        

    def on_edit_press(self):
        self.clear_buttons()
        self.prompt_display.configure(state='normal')
        self.save_button.pack(side=tk.LEFT)    
        self.reset_button.pack(side=tk.LEFT)

    def on_save(self):
        print('run save')
        self.clear_buttons()
        self.prompt_display.configure(state='disabled')
        self.current_text = self.prompt_display.get('1.0', 'end-1c')
        self.update_property(self.current_text)

        self.edit_button.pack(side=tk.LEFT)
        if not (self.current_text == self.default_text):
            self.reset_button.pack(side=tk.LEFT)

    def on_reset(self):
        self.clear_buttons()
        self.update_property(self.default_text)
        self.edit_button.pack()
        self.prompt_display.configure(state='normal')
        self.prompt_display.delete('1.0', 'end')
        self.prompt_display.insert('1.0', self.default_text)
        self.prompt_display.configure(state='disabled')

    def clear_buttons(self):
        self.edit_button.pack_forget()
        self.save_button.pack_forget()
        self.reset_button.pack_forget()

    def get(self):
        return self.current_text
 
class AddFrame(ttk.Frame):
    def __init__(self, master):
        super().__init__(master)
        self.pack
        self.canvas = tk.Canvas(self)
        self.canvas.pack(side='left', fill = 'both', expand = True)

        self.scrollbar = tk.Scrollbar(self.canvas, orient='vertical', command=self.canvas.yview)
        self.scrollbar.pack(fill='y', side='right')
        self.canvas.configure(yscrollcommand=self.scrollbar.set)
        
        self.frame = tk.Frame(self.canvas)
        self.canvas_window = self.canvas.create_window((0,0), window=self.frame, anchor="nw")

        self.frame.bind("<Configure>", self.update_scrollregion)
        self.bind_all("<MouseWheel>", self.mouse_scroll)
        self.canvas.bind("<Configure>", self.update_window_size)
        self.frame.bind("<Button-1>", self.activate_window_scrollbar)  # Focus out when clicking on frame

        self.title = PROJECT.title
        self.style = ttk.Style()
        self.style.configure("Title.TLabel", font=("Helvetica", 24, "bold"), foreground="black")
        
        #Removed testing function
        # self.check_attribute_button = tk.Button(text='current_prompt', command= lambda : print(PROJECT.current_prompt))
        # self.check_attribute_button.pack()

    def mouse_scroll(self, event):
        self.canvas.yview_scroll(-1*(event.delta//120), "units")

    def update_scrollregion(self, event):
        self.canvas.config(scrollregion=self.canvas.bbox("all"))

    def update_window_size(self, event):
        self.canvas.itemconfig(self.canvas_window, width=event.width)

    def create_title(self, text):
        label = ttk.Label(self.frame, text=text, style="Title.TLabel")
        label.pack()

    #Note: While when I click on the textbox, the window stops moving, sometimes
    #while the window is selected the textbox also scrolls if the mouse cursor is over 
    #it. This should be corrected later, but doesn't seem like an essential part of an MVP
    #My best guess it that this driven by the scrollbar built into Tkinter's textbox class
    #and that would need to be turned off and on in the relevant functions. 
    def activate_window_scrollbar(self, event):
        self.canvas.bind_all("<MouseWheel>", self.mouse_scroll)
        for widget in self.frame.winfo_children():
            if isinstance(widget, tk.Text):
                widget.unbind("<MouseWheel>")
        self.canvas.focus_set()

    def deactivate_window_scrollbar(self, event):
        self.canvas.unbind_all("<MouseWheel>")

    def bind_activate_window_scrollbar_to_textbox_labels(self):
        for widget in self.frame.winfo_children():
            if isinstance(widget, tk.Frame):
                for sub_widget in widget.winfo_children():
                    if isinstance (sub_widget, tk.Label):
                        sub_widget.bind("<Button-1>", self.activate_window_scrollbar)
                


    #CHANGE/ WARNING: A clearly ugly solution to getting the font for the default label, that also
        #makes the code more rigid because it doesn't respond to changes in font
    def label_word_wrapper(self, text, max_width = 500):
        label = tk.Label(self)
        font = tkFont.Font(font=label.cget("font"))
        label.destroy()
        return wrap_with_font(font, text, max_width)

class ProjectDisplayLine:
    def __init__(self, parent, section):
        self.section = section
        self.parent = parent
        self.text_frame = None
        self.line_frame = tk.Frame(self.parent)
        self.label = tk.Label(self.line_frame, width=70, text=section.name, padx=10, borderwidth=1, relief='solid')
        self.view_text_button = tk.Button(self.line_frame, text='View Text', command=self.view_text, 
                                bg="#DDDDDD", relief="groove")
        self.generate_output_button = tk.Button(self.line_frame, text='Generate', width=10, command=self.generate_output, 
                                bg="#DDDDDD", relief="groove")
        self.view_output_button = tk.Button(self.line_frame, text='View Outputs', width = 10, command=self.view_output, 
                                bg="#DDDDDD", relief="groove")
        self.no_outputs_button = tk.Button(self.line_frame, text='No Outputs', width=10, command=None, 
                                   bg="#AAAAAA", relief="sunken")
        self.processing_button = tk.Button(self.line_frame, text="Processing", width=10, fg='yellow', bg='green', relief="solid")
        
        self.label.grid(row=0, column=0, sticky='nsew')
        self.view_text_button.grid(row=0, column=1, sticky='nsew')
        self.generate_output_button.grid(row=0, column=2, sticky='nsew')
        if self.section.llm_outputs:
            self.view_output_button.grid(row=0, column=3, sticky= 'nsew')
        else: self.no_outputs_button.grid(row=0, column=3, sticky='nsew')

        #have the scrollbar for the notebook tab activate when the labels are clicked
        self.label.bind('<Button-1>', main_interface.editing_frame.activate_window_scrollbar)

    def place_in_grid(self, row):
        self.row = row
        self.line_frame.grid(row=row, columnspan=4, sticky='nsew')

    def view_text(self):
        if self.text_frame:
            self.text_frame.destroy()
            self.text_frame = None
            return
        self.text_frame = tk.Frame(self.line_frame)
        self.text_frame.grid(row=1, columnspan=4, sticky='nsew')
        self.textbox = tk.Text(self.text_frame, wrap='word', width=60, height=10)
        self.textbox.pack(expand=True, fill='both')
        self.textbox.insert(tk.END, self.section.section_text) 
        self.textbox.bind("<FocusIn>", deactivate_window_scrollbar)
        
        self.close_button = tk.Button(self.text_frame, text="Close", command=self.close_text_frame)
        self.close_button.pack(side="right")

    def generate_output(self):
        self.generate_output_button.grid_forget()
        self.processing_button.grid(row=0, column=2, sticky='nsew')
        threading.Thread(target=self.run_asyncio_loop).start()

    def run_asyncio_loop(self):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(self.get_GPT_response())
    
    async def get_GPT_response(self):
        print ('about to start waiting')

        if PROJECT.gpt4_flag == 1:
            model = 'gpt-4'
        else: 
            model = 'gpt-3.5-turbo'

        try:
            output = await openai.ChatCompletion.acreate (
            model = model,
            messages = [{"role" : "system", "content" : PROJECT.current_prompt},
                      {"role" : "user", "content" : self.section.section_text}
                      ]
        )
            reply = output['choices'][0]['message']['content']
            self.section.llm_outputs += f'Prompt: {PROJECT.current_prompt} \n Output: {reply} \n [End of Output] \n'

        except openai.error.AuthenticationError as e:
            print('exception')
            show_error('You probably entered an invalid API key')
            self.processing_button.grid_forget()
            self.generate_output_button.grid(row=0, column=2, sticky='nsew')
        
        except openai.error.RateLimitError as e:
            global rate_limit_hit
            rate_limit_hit = True
            root.after(60000, clear_rate_limit)
            return
            
        except Exception as e:
            show_error(str(e))


        self.section.llm_outputs += PROJECT.current_prompt + '\n' + reply + '\n'
        self.processing_button.after(0, self.processing_button.grid_forget)
        self.generate_output_button.after(0, lambda : self.generate_output_button.grid(row=0, column=2))
        self.no_outputs_button.after(0, self.no_outputs_button.destroy)
        self.view_output_button.after(0, lambda : self.view_output_button.grid(row=0, column=3))
    
    def view_output(self):
        if self.text_frame:
            self.text_frame.destroy()
            self.text_frame = None
            return
        self.text_frame = tk.Frame(self.line_frame)
        self.text_frame.grid(row=1, columnspan=4, sticky='nsew')
        self.textbox = tk.Text(self.text_frame, wrap='word', width=60, height=10)
        self.textbox.pack(expand=True, fill='both')
        self.textbox.insert(tk.END, self.section.llm_outputs) 
        self.textbox.bind("<FocusIn>", deactivate_window_scrollbar)
        
        self.close_button = tk.Button(self.text_frame, text="Close", command=self.close_text_frame)
        self.close_button.pack(side="right")

    def close_text_frame(self):
        self.text_frame.destroy()
        self.text_frame=None

class ProjectDisplayBox:
    def __init__(self, master):
        self.frame = tk.Frame(master)
        self.frame.pack()

        self.lines = []
        i = 0
        for chapter in PROJECT.chapters:
            for section in chapter.sections:
                line = ProjectDisplayLine(self.frame, section)
                line.place_in_grid(i)
                i += 1
                self.lines.append(line)

    #maybe stick the chapter name in the section name at creation?
    def create_line(self, chapter, section):
        name = f'{chapter.name}:{section.name}'
        button_frame = tk.Frame(self.frame)

    def display_processing_symbol(self):
        pass

    def display_processing_message(self):
        pass
    

class EditorFrame(AddFrame):
    def __init__(self, master):
        super().__init__(master)
        global PROJECT
        self.divided = PROJECT.divided
        self.rate_limit_hit = False
        PROJECT.current_prompt = ""

        # if not api_key:
        #     self.api_entry_box = CustomTextBox(self.frame, 'api_key', label='Please enter your OpenAI api key here', widget_type='entrybox', submitted_text='We will use {} as the API key')
        if PROJECT.api_key:
            openai.api_key = PROJECT.api_key
            self.api_entry_box = EditAndRestoreBox(self.frame, PROJECT.api_key, height=1, width=40, label_text="The OpenAI API key is:", property='api_key', object=PROJECT)
        else:
            self.api_entry_box = EditAndRestoreBox(self.frame, "", height=1, width=40, label_text="Enter your OpenAI API key:", property='api_key', object=PROJECT)
        

        prompt_text = "You are a developmental editor with years of experience in helping writers create bestselling novels, you will rate the following scene and then provide concrete and specific advice on how to make it more emotionally powerful, compelling, and evocative."
        label_text='Your Current Prompt'
        
        self.prompt = EditAndRestoreBox(self.frame, prompt_text, label_text=label_text, property = "current_prompt", object=PROJECT)
        self.create_gpt4_toggle()
        
        self.bind_activate_window_scrollbar_to_textbox_labels()


    #ugly solution that I came up with at 9pm to allow me to bind in the ProjectDisplayLine the window scrollbar in
    #the issue is that it doesn't have a self that refers to the editing window, but it wants to use the editing windows scrollbar
    #but it is called in the code that creates the editing window, so any global variable that has the editing window as an attribute
    #will not yet have been created when the interpreter checks if the editing window exists so that the binding can work
    #I'm solving this right now by delaying building the display box until after the editing window has been initialized and defined
    #but I feel like this should be in the intiialization of the editing window... though I'm not actually sure now that I think about it
    def build_display_box(self):
        if not self.divided:
            self.break_into_sections_box()
        else: 
            self.display_current_project()

    def update_gpt4_flag(self):
        #why is it global button_exists? Who uses that, and can we get this different?
        #Sure, use globals for things like PROJECT and main_interface, but this?
        global button_exists
        PROJECT.gpt4_flag = 0
        PROJECT.gpt4_flag = self.gpt4_flag.get()
        print(PROJECT.gpt4_flag)

        if PROJECT.gpt4_flag and not self.button_exists == True:
            text="Note: GPT-4 costs 20 times as much per token. Experiment to make sure you like the results before using it extensively"
            customFont = tkFont.Font(family="Helvetica", size=10, weight="bold")
            wrapped_text = self.label_word_wrapper(text)

            self.gpt_buttons_label = tk.Label(self.toggle_frame, text = wrapped_text,
                    font=customFont,
                    fg="red")
            self.gpt_buttons_label.pack(side=tk.BOTTOM)
            self.button_exists = True

    def create_gpt4_toggle(self):
        self.toggle_frame = tk.Frame(self.frame)
        self.toggle_frame.pack()
        self.inner_toggle_frame = tk.Frame(self.toggle_frame)
        self.inner_toggle_frame.pack()

        self.gpt4_flag = tk.IntVar()
        self.gpt4_flag.set(0)
        self.button_exists = False


        self.gpt_turbo_button = tk.Radiobutton(self.inner_toggle_frame, variable = self.gpt4_flag, command= self.update_gpt4_flag, text='GPT-3.5', value=0)
        self.gpt_4_button = tk.Radiobutton(self.inner_toggle_frame, variable=self.gpt4_flag, command=self.update_gpt4_flag, text= 'GPT-4', value=1)
        self.gpt_turbo_button.pack(side=tk.LEFT)
        self.gpt_4_button.pack(side=tk.LEFT)

    def break_into_sections_box(self):
        # Check if the divide frame already exists
        print('starts break into sections box')
        if not hasattr(self, 'divide_frame'):
            self.divide_frame = tk.Frame(self.frame)
            self.divide_frame.pack()

        # Create or update label
        if hasattr(self, 'label'):
            print(PROJECT.project_text)
            if PROJECT.project_text:
                print('if project statement fires')
                self.label.config(text=self.label_word_wrapper("..."))
            else:
                print('if project statement does not fire')
                self.label.config(fg='red', text=self.label_word_wrapper("You need to enter text..."))
            print(PROJECT.project_text)
        else:
            if PROJECT.project_text:
                self.label = tk.Label(self.divide_frame, text=self.label_word_wrapper("We still need to break this project into sections small enough to be sent to GPT. You can break the text up at each capitalized 'Chapter'. Otherwise everything will be divided into equally sized sections of up to about 1500 words (2000 tokens) with an overlap of about forty words. This is how chapters will be divided also."))
            else:
                self.label = tk.Label(self.divide_frame, fg='red', text=self.label_word_wrapper("You need to enter text..."))
            self.label.pack()

        # Check if the divide text buttons frame already exists
        if not hasattr(self, 'divide_text_buttons_frame'):
            self.divide_text_buttons_frame = tk.Frame(self.divide_frame)
            self.divide_text_buttons_frame.pack()

        # Check if the check button already exists
        if not hasattr(self, 'check_button'):
            self.chapter_divider_flag = tk.BooleanVar()
            self.chapter_divider_flag.set(False)
            self.check_button = tk.Checkbutton(self.divide_text_buttons_frame, text="'Chapter' is the chapter divider", variable=self.chapter_divider_flag, command=lambda: print(f'{self.chapter_divider_flag.get()}'))
            self.check_button.pack(side=tk.LEFT)

        # Check if the submit button already exists
        if not hasattr(self, 'submit_button'):
            self.submit_button = tk.Button(self.divide_text_buttons_frame, text='Split your text into sections', command=self.submit_break_into_sections)
            self.submit_button.pack(side=tk.LEFT)


    def submit_break_into_sections(self, flag = None):
        #Something stuck in for a test, should be removed later
        if isinstance(flag, bool):
            PROJECT.create_sections_and_chapters_from_text(flag)
            return
        PROJECT.create_sections_and_chapters_from_text(self.chapter_divider_flag.get())
        PROJECT.divided = True

        self.divide_frame.destroy()
        self.display_current_project()
        
        

    def display_current_project(self):
        self.outer_project_frame = tk.Frame(self.frame)
        self.outer_project_frame.pack()
        title = tk.Label(self.outer_project_frame, text=f'Your Current Project: {PROJECT.title}')
        title.pack()
        self.create_project_buttons(self.outer_project_frame)
        self.display = ProjectDisplayBox(self.outer_project_frame)
        self.create_project_buttons(self.outer_project_frame)
        self.bind_all("<MouseWheel>", self.mouse_scroll)



    def create_project_buttons(self,master):
        button_frame = tk.Frame(master)
        button_frame.pack()
        self.run_all_button = tk.Button(button_frame, text = 'Run on all sections', command=self.run_all)
        self.download_responses_to_txt_button = tk.Button(button_frame, text = 'Get text of GPT responses', command=self.save_outputs)
        self.save_project_button = tk.Button(button_frame, text = 'Save Project', command=self.save_project)
        
        self.run_all_button.pack(side=tk.LEFT)
        self.download_responses_to_txt_button.pack(side=tk.LEFT)
        self.save_project_button.pack(side=tk.LEFT)

        run_all_tooltip = ToolTip(self.run_all_button, 'To avoid hitting openai rate limits this will wait for sixty seconds after starting each batch of thirty')
        save_project_tooltip = ToolTip(self.save_project_button, 'The file is saved as a .json which is a human readable format. You can open it with notepad and copy paste any data out.')

    def check_rate_limit(self):
        if rate_limit_hit == True:
            root.after(1000, self.check_rate_limit)
        else: 
            self.run_all()

    def run_all (self):
        delay = 0
        counter = 0
        self.run_all_button.configure(command=None, relief='sunken')
        for line in self.display.lines:
            counter += 1
            if not line.section.llm_outputs:
                if counter >= 30:
                    delay += 60000
                    counter = 0
            root.after (delay, lambda line=line : line.generate_output())
            delay += 400

    def text_for_download(self):
        text = ""
        for chapter in PROJECT.chapters:
            for section in chapter.sections:
                text += section.name + ':\n' + section.llm_outputs + '\n'
        return text
    
    def save_outputs(self):
        text = self.text_for_download()
        file_path = filedialog.asksaveasfilename(defaultextension=".txt",
                                             filetypes=[("Text files", "*.txt"),
                                                        ("Python files", "*.py"),
                                                        ("All files", "*.*")])
        if file_path:
            with open(file_path, 'w') as f:
                f.write(text)
                
    def save_project(self):
        file_path = filedialog.asksaveasfilename(defaultextension=".json",
                                             filetypes=[("Json", "*.json")],
                                                        )
        PROJECT.save(file_path)

class InputFrame(AddFrame):
    def __init__ (self, master):
        super().__init__(master)
        self.create_title(text = f'Enter the details of {self.title}')

        label_texts = self.fetch_label_texts()
        self.project_text = CustomTextBox(self.frame, property = 'project_text', object = PROJECT, label = label_texts['project_text'])

        #Section headings that will be used for later features but I've dropped from my
        #minimum viable product version
        # self.key_information = CustomTextBox(self.frame, 'key_information', label_texts['key_information'])
        # self.key_information.change_submitted_text('Your key information has been saved.')
        
        # self.reviews = CustomTextBox(self.frame, 'reviews', label_texts['reviews'])
        # self.reviews.change_submitted_text('Your example reviews have been saved.')

        # self.blurbs = CustomTextBox(self.frame, 'sample_blurbs', label_texts['sample_blurbs'])
        # self.blurbs.change_submitted_text('Your example blurbs have been saved.')
        
        self.go_to_editing_page_button()

        self.bind_activate_window_scrollbar_to_textbox_labels()

    def fetch_label_texts(self):
        label_texts = {}
        label_texts['project_text'] = 'Enter the text for your project here'
        label_texts['key_information'] = self.label_word_wrapper('OPTIONAL: Put important information about the novel here. You likely will only realize something needs to be here after seeing what chatGPT writes about your novel. Experiment to see what works best.')
        label_texts['reviews'] = self.label_word_wrapper('OPTIONAL: Put reviews of novels in your genre here. ChatGPT will later use it to identify key features of books in it that readers like. This can be used to improve the editing suggestions and optimize the blurb.')
        label_texts['sample_blurbs'] = self.label_word_wrapper('OPTIONAL: Put examples of blurbs from books in your genre so that chatGPT can use their example to generate a better blurb for you. This only is used for blurb writing.')
        return label_texts
    
    def go_to_editing_page_button(self):
        button_frame = tk.Frame(self.frame)
        button_frame.pack()
        to_editing_page = tk.Button(button_frame, text='Go to Editing Page', command=loading_page2.load_editing_page)
        to_editing_page.pack()

class ToolTip:
    def __init__(self, widget, text):
        self.widget = widget
        self.text = text
        self.tooltip_window = None
        self.widget.bind("<Enter>", self.show_tooltip)
        self.widget.bind("<Leave>", self.hide_tooltip)

    def show_tooltip(self, event=None):
        x, y, _, _ = self.widget.bbox("insert")
        x += self.widget.winfo_rootx() + 25
        y += self.widget.winfo_rooty() + 25

        self.tooltip_window = tk.Toplevel(self.widget)
        self.tooltip_window.wm_overrideredirect(True)  # Removes the window decorations
        self.tooltip_window.wm_geometry(f"+{x}+{y}")

        label = tk.Label(self.tooltip_window, text=self.text, bg="yellow", relief="solid", borderwidth=1)
        label.pack()

    def hide_tooltip(self, event=None):
        if self.tooltip_window:
            self.tooltip_window.destroy()
            self.tooltip_window = None

class MainInterface:
    def __init__(self, master) -> None:
        root.notebook = ttk.Notebook(master)
        root.notebook.pack(expand=True, fill="both")

        self.notebook = root.notebook

        self.editing_frame = EditorFrame(self.notebook)

        self.input_frame = InputFrame(self.notebook)

        self.notebook.add(self.input_frame, text="Input Novel")
        self.notebook.add(self.editing_frame, text="Editing and Summarization")

def start_program():
    global PROJECT, rate_limit_hit, root, async_loop, loading_page, loading_page2, api_key, WARNING_FONT, main_interface
    PROJECT = Project
    rate_limit_hit = False
    api_key = None
    root = tk.Tk()
    root.title('GPT Writing Tools')
    loading_page = LoadingPage(root)
    root.geometry("800x600+100+50")
    root.bind_all('<Control-q>', quit_program)
    root.bind_all('<Control-m>', _go_to_current_position)

    
    async_loop = asyncio.get_event_loop()
    root.after(100, asyncio_loop_cycle, async_loop)
    root.mainloop()

if __name__ == '__main__':
    start_program()