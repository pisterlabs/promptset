import subprocess
from openai import OpenAI

try:
    client = OpenAI(
        api_key="<Enter your API Key HERE **REMOVED FOR SECURITY PURPOSES**>")
    user_query = input("What would you like to ask? ")
    completion = client.chat.completions.create(
        model="gpt-4-1106-preview",
        messages=[
            {
                "role": "system",
                "content": """
                # Manim Script Instructions


## Introduction
Instruct ChatGPT to create a lengthy, well laid out, extensive Manim script that elegantly demonstrates a mathematical concept or problem. keep in mind the dimensions of the window.
IMPORTANT - it is Create() not ShowCreation()
## Overall Structure
Specify the general structure of the script.

1. **Set Up the Scene**:
   - Define the scene and its properties.
   - Create any necessary objects.
   - Ensure all objects are instances of `Mobject` or its subclasses.
   - Position objects to avoid overlap.Keep space between words. Load all symbols properly. Global text size is small but legible in such a way that the text does not leave the border. Have enough checks to ensure nothing is going out of boundary.

2. **Display Mathematical Content**:
   - Include relevant equations or mathematical expressions using `MathTex`.
   - Ensure proper formatting, including LaTeX when applicable. Avoid LaTeX compilation errors in all cases. No "Undefined control sequence." Avoid in all cases please.
   - Animate or display mathematical content as needed.
   
3. **Provide Explanations**:
   - Include step-by-step explanations with `MathTex` formatting.
   - Use text boxes, labels, or annotations to clarify concepts.
   - Position text to avoid overlap with other elements.

4. **Transition Effects**:
   - Add transition effects between scenes or elements.
   - Specify the timing and duration of animations.
   - Ensure that animation targets are valid `Mobject` instances.

5. **Customization** (optional):
   - Allow for any customizations or variations based on specific examples or preferences.

6. **Quality and Layout**:
   - Ensure a clear layout with appropriate spacing.
   - Adjust the rendering quality for optimal viewing.
   - Check for compatibility with your chosen Manim version.

7. **Debugging and Troubleshooting**:
   - Include instructions on handling common errors, such as LaTeX-related issues.
   - Verify that all objects used for animations are of the correct type (`Mobject` or subclasses).
   - Double-check the inheritance of custom objects to ensure compatibility.

## Example Script
Provide a complete example script based on the instructions above. Keep everything inside the window. Do your best.

```python

from manim import *
# Increase the resolution and window size MANDATORY
config.pixel_height = 1080
config.pixel_width = 1920
config.frame_height = 6.0
config.frame_width = 6.0


class MyMathAnimation(Scene):
    def construct(self):
        # Set Up the Scene
        # Define the scene properties and create objects.

        # Example: Create a triangle
        triangle = Polygon(
            ORIGIN, RIGHT * 2, UP * 2,
        )
# Incorrect (causes the error):
my_object.color = RED

# Correct:
my_object.set_color(RED)

# Incorrect
ShowCreation() # is not defined

#Correct
Create()

        # Position the triangle
        triangle.move_to(LEFT * 2)

        # Display Mathematical Content
        # Include equations or expressions using `MathTex`.

        # Example: Add a mathematical expression
        expression = MathTex("E = \\frac{m \\cdot c^2}{\\quad\\quad}")


        # Position the equation below the triangle
        equation.next_to(triangle, DOWN)

        # Provide Explanations
        # Add step-by-step explanations with `MathTex` formatting.

        # Example: Add an explanation
        explanation = MathTex("This equation is known as Euler's identity.")

        # Position the explanation below the equation
        explanation.next_to(equation, DOWN)

        # Transition Effects
        # Specify animations and transitions between scenes.

        # Example: Animate the creation of objects
        self.play(Create(triangle))

        # Example: Animate the appearance of equations
        self.play(Write(equation))

        # Example: Animate explanations
        self.play(Write(explanation))

        # Customization (optional)
        # Allow for any customizations or variations.

        # Quality and Layout
        # Ensure clear layout and adjust rendering quality.

        # Debugging and Troubleshooting
        # Include instructions for handling errors.
        # Correct usage with two arguments (real and imag)
my_complex_number = complex(2.0, 3.0)

# Correct usage with one argument (real, imag defaults to 0)
my_complex_number = complex(2.0)

# Incorrect usage with more than two arguments (causes the error)
my_complex_number = complex(2.0, 3.0, 4.0)  # This will raise "complex() takes at most 2 arguments (3 given)"
# if text length more than 30 characters, break it and start at new line. do not exceed boundaries.

                """
            },
            {
                "role": "user",
                "content": f"Just give code as plain text no '```' or '```python' in the output. I don't need explanations. use MathTex and keep in mind the size of the window. adjust size of text accordingly. DO NOT exceed boundaries.If there's more content, erase stuff first and then write on it.Go slow, increase wait time. And keep it simple unless programming wise. required. Query: {user_query}"
            }
        ]


    )
    response_content = completion.choices[0].message.content
except Exception as e:
    print(f"An error occurred: {e}")

manim_script_filename = "generated_manim_script.py"
# Split the response content into lines
lines = response_content.split('\n')

# Remove the first and last lines
if len(lines) >= 2:
    modified_lines = lines
else:
    # Handle the case where there are less than 2 lines
    modified_lines = []

# Join the remaining lines back into a single string
modified_content = '\n'.join(modified_lines)

# Write the modified content to the file
with open(manim_script_filename, "w") as file:
    file.write(modified_content)
manim_command = ["manim", "-pql", manim_script_filename]
try:
    subprocess.run(manim_command, check=True)
except subprocess.CalledProcessError as e:
    print(f"An error occurred while running the Manim script: {e}")
except FileNotFoundError:
    print("Manim command not found. Ensure that Manim is installed and accessible.")
