from typing import Literal, Iterable
import dotenv
import os
import openai


HTML_TAGS_DESC = '''
    <h1> to <h6>: Headings
    Used to define headings of different levels. <h1> being the highest level and <h6> the lowest.
    <p>: Paragraph
    Used to define a paragraph of text.
    <a>: Anchor
    Used to create a hyperlink to another web page or a specific location within the same page.
    <img>: Image
    Used to embed an image into an HTML document.
    <ul>: Unordered List
    Used to create an unordered (bulleted) list of items.
    <ol>: Ordered List
    Used to create an ordered (numbered) list of items.
    <li>: List Item
    Used to define an item in a list (<ul> or <ol>).
    <div>: Division
    Used as a container to group and style HTML elements together.
    <span>: Inline Container
    Used as an inline container to group and style a specific section of text or inline elements.
'''

answer_example_for_loop = '''
 <h1>Using a for loop in Python</h1>
  <h2>Step 1: Syntax</h2>
  <p>In Python, the syntax for a for loop is as follows:</p>
  <pre><code>for variable in iterable:
    # Code block to be executed</code></pre>
  <p>The <code>variable</code> represents the current item in the iteration, and the <code>iterable</code> is a collection of items.</p>
  <h2>Step 2: Example</h2>
  <p>Let's consider an example where we have a list of numbers and we want to print each number using a for loop:</p>
  <pre><code>numbers = [1, 2, 3, 4, 5]
    for num in numbers:
        print(num)</code></pre>
  <p>This code will iterate over each number in the <code>numbers</code> list and print it.</p>
  <h2>Step 3: Output</h2>
  <p>When you run the code, the output will be:</p>
  <pre>
  <code>
    1
    2
    3
    4
    5
  </code></pre>
  <h2>Step 4: Additional Operations</h2>
  <p>A for loop can be used to perform various operations on the elements of an iterable. For example, you can calculate the sum of a list of numbers:</p>
  <pre><code>numbers = [1, 2, 3, 4, 5]
sum = 0
for num in numbers:
    sum += num
print("Sum:", sum)</code></pre>
  <p>The output will be:</p>
  <pre><code>Sum: 15</code></pre>
'''
example_car_class = '''

<h1>Defining a Car Class in Python</h1>
  <h2>Step 1: Creating the Class</h2>
  <p>A class in Python can be created using the <code>class</code> keyword followed by the name of the class:</p>
  <pre><code>class Car:</code></pre>
  <p>This creates an empty class called <code>Car</code>.</p>
  <h2>Step 2: Defining Attributes and Methods</h2>
  <p>Attributes are variables that belong to the class, while methods are functions that belong to the class. We can define them inside the class:</p>
  <pre><code>class Car:
    def __init__(self, make, model, year):
        self.make = make
        self.model = model
        self.year = year
        self.odometer_reading = 0

    def get_description(self):
        return f"{self.year} {self.make} {self.model}"

    def read_odometer(self):
        return f"This car has {self.odometer_reading} miles on it."

    def update_odometer(self, mileage):
        if mileage >= self.odometer_reading:
            self.odometer_reading = mileage
        else:
            return "You can't roll back an odometer!"</code></pre>
  <p>The <code>__init__()</code> method is a special method that is called when an instance of the class is created. It sets the initial values of the attributes.<br>
    The <code>get_description()</code> method returns a formatted description of the car's make, model, and year.<br>
    The <code>read_odometer()</code> method returns the current odometer reading.<br>
    The <code>update_odometer()</code> method updates the odometer reading if the new value is equal to or greater than the current value.</p>
  <h2>Step 3: Creating an Instance of the Class</h2>
  <p>To create an instance of the class, we need to call the class name and pass the required arguments to the <code>__init__()</code> method.</p>
  <pre><code>my_car = Car("Ford", "Mustang", 2021)</code></pre>
  <h2>Step 4: Accessing Attributes and Calling Methods</h2>
  <p>Once an instance of the class has been created, we can access its attributes and call its methods using dot notation:</p>
  <pre><code>
    print(my_car.get_description())
    print(my_car.read_odometer())
    my_car.update_odometer(10000)
    print(my_car.read_odometer())</code></pre>
  <p>This will output:</p>
  <pre><code>
    2021 Ford Mustang
    This car has 0 miles on it.
    This car has 10000 miles on it.
    </code></pre>
'''
nonsense_example = '''
<p style="color: white">I'm sorry, I'm not sure what you mean by "something not make sense". Could you please provide more context or clarify your question?</p>
'''

Role =  Literal["system", "user", "assistant"]

class CodeAdvisor:
    PROMPT = f"""
    You are performing the role of a friendly and nice code advisor. 
    You will receive a question at a time.
    Yor job is answer the question from the perspective of software development.
    Please strictly follow the following rules of answering the question
        - MOST IMPORTANT Since your response will be rendered on a frontend page, 
          foramt your response and anwer into HTML even if you don't know what user is asking.
        - Answer your question in a step-by-step manner with a clear explaination, provide example code if needed.
        - Don't put class in your HTML response, which may waste the available tokens you can use.
        - Don't make up an answer if you don't know what user is asking.
        - Indent the python code block while needed, since indentation matters in python. 
        - You are strongly encouraged to use a wide range of HTML tags as below:
            {HTML_TAGS_DESC}
         
    """
    def __init__(self) -> None:
        self.messages = self._get_init_messages()
        self._init_messages_len = len(self.messages)
        self.__ask_api_key()

    def _get_init_messages(self) -> list[dict[str, str]]:
        return [
            {"role": "system", "content": self.PROMPT},
            {"role": "user", "content": "how to use for loop in python?"},
            {"role": "assistant", "content": answer_example_for_loop},
            {"role": "user", "content": "thank you! that's very helpful"},
            {"role": "assistant", "content": "You are welcome, happy to help!"},
            {"role": "user", "content": "Write me a example car class in python."},
            {"role": "assistant", "content": example_car_class},
            {"role": "user", "content": "something not make sense"},
            {"role": "assistant", "content": nonsense_example},
        ]

    def _add_message(self, role: Role, content: str) -> None:
        msg = {"role": role, "content": content}
        self.messages.append(msg)
    
    def __ask_api_key(self) -> None:
        dotenv.load_dotenv()
        key =  os.getenv("OPENAI_API_KEY")
        if key is None:
            raise ValueError("'OPENAI_API_KEY' not found")
            
        openai.api_key = key

    def clear_session(self) -> None:
        print("Cleaning up message queue...")
        self.messages = self._get_init_messages()

    def _update_messages(self, msg: str) -> None:
        self.messages[-1]["content"] += msg

    @property
    def message_session(self) -> str:
        return self.messages[self._init_messages_len:]

        
    def answer(self, question: str) -> Iterable[str]:
        try:
            self._add_message("user", f"{question}\n")
            generator = openai.ChatCompletion.create(
                model= "gpt-3.5-turbo",
                stream= True,
                messages= self.messages
            )
            self._add_message("assistant", "")
            yield self.message_session
            for response in generator:
                choice = response['choices'][0]["delta"]
                if "content" in choice:
                    content = choice["content"]
                    self._update_messages(content)
                    yield self.message_session
            self._update_messages("\n")

            yield self.message_session
        except openai.error.InvalidRequestError as error:
            self.clear_session()
            return self.answer(question)