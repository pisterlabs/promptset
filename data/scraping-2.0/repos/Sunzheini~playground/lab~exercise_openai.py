from tkinter import *
import openai


class DirWalkerGui:
    def __init__(self):
        self.window = Tk()
        self.window.geometry("300x300")
        self.window.title("DZ AI Prompt Generator")
        self.window.config(background='#2b2828')

        self.api_key = None

# Section 1 -----------------------------------------------------------
        self.label1 = Label(
            self.window, text='постави api ключ', width=30, height=1,
            bg='#2b2828', borderwidth=0, relief="ridge", fg='white'
        )
        self.label1.pack()

        self.entry1 = Entry(
            self.window,
            font=("Arial", 8),
            fg='white',
            bg='black',
        )
        self.entry1.insert(
            0,                  # from th beginning position
            '',     # default text
        )
        self.entry1.pack()

        self.button1 = Button(
            self.window, text='запази ключа', width=15, height=1,
            command=self.store_key
        )
        self.button1.pack()

        canvas = Canvas(self.window, width=300, height=1, bg='#2b2828', borderwidth=0)
        canvas.pack()


# Section 2 -----------------------------------------------------------
        self.label2 = Label(
            self.window, text='задай въпрос', width=30, height=1,
            bg='#2b2828', borderwidth=0, relief="ridge", fg='white'
        )
        self.label2.pack()

        self.entry2 = Entry(
            self.window,
            font=("Arial", 8),
            fg='white',
            bg='black',
        )
        self.entry2.insert(
            0,  # from th beginning position
            '',  # default text
        )
        self.entry2.pack()

        self.button2 = Button(
            self.window, text='изпрати', width=15, height=1,
            command=self.send_query
        )
        self.button2.pack()

        canvas2 = Canvas(self.window, width=300, height=1, bg='#2b2828', borderwidth=0)
        canvas2.pack()

# Section 3 -----------------------------------------------------------

        self.text_widget = Text(self.window, height=8, width=33)
        self.text_widget.pack()

# ------------------------------------------------------------------

    def store_key(self):
        new_input = self.entry1.get()
        self.api_key = new_input
        self.entry1.config(state=DISABLED)  # disable after submitting

    def send_query(self):
        if self.api_key is None:
            self.print_text('Моля, поставете API ключа!')
            return

        # openai.api_key = key
        openai.api_key = self.api_key
        total_tokens = 0
        prompt = self.entry2.get()

        response = openai.Completion.create(
          model="text-davinci-003",
          prompt=prompt,
          temperature=0.9,
          max_tokens=150,
          top_p=1,
          frequency_penalty=0.0,
          presence_penalty=0.6,
          stop=[" Human:", " AI:"]
        )

        self.print_text(response.choices[0].text)

    def print_text(self, text):
        self.text_widget.delete('1.0', 'end')  # clear any previous text

        # formatted_text = '\n'.join([f'AI: {line.strip()}' for line in text.split('\n')])
        formatted_text = '\n'.join([line.strip() for line in text.split('\n')])

        self.text_widget.insert('1.0', formatted_text)  # insert new text

    def run(self):
        self.window.mainloop()


new_gui = DirWalkerGui()
new_gui.run()
