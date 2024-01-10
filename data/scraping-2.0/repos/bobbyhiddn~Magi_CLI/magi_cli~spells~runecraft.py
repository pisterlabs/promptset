import os
import requests
from PIL import ImageDraw, Image, ImageOps
import subprocess
from io import BytesIO
from openai import OpenAI
import click
import pkg_resources
from magi_cli.spells import SANCTUM_PATH 

DEFAULT_IMAGE_PATH = pkg_resources.resource_filename('magi_cli.artifacts', 'Rune.png')

# Conditional instantiation of the OpenAI client
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if OPENAI_API_KEY:
    client = OpenAI(api_key=OPENAI_API_KEY)
else:
    client = None

# Function to generate an image using DALL-E API
def generate_image(prompt):
    if client:        
        response = client.images.generate(
            model="dall-e-3",
            prompt=prompt,
            n=1,
            size="1024x1024",
            response_format="url"
        )
        # Extract the image URL using dot notation
        image_url = response.data[0].url
        image_data = requests.get(image_url).content
        image = Image.open(BytesIO(image_data))
    else:
        # Fallback to a default image if the client is not available
        image = Image.open(DEFAULT_IMAGE_PATH)

    return image

# Function to create a circular mask
def create_circular_mask(image):
    width, height = image.size
    mask = Image.new('L', (width, height), 0)
    draw = ImageDraw.Draw(mask)
    draw.ellipse((0, 0, width, height), fill=255)
    return mask

@click.command()
@click.argument('file_paths', nargs=-1, required=True)  # Accepts multiple file paths
def runecraft(file_paths):
    ''' 'rc' - Generate a GUI for a Bash script in the form of an enchanted rune.'''

    # Correctly handle the first file path from the list
    if file_paths:
        file_path = file_paths[0]
        base_filename = os.path.basename(file_path)
    else:
        # Handle the case where no file path is provided
        click.echo("Error: No file path provided.")
        return

    print("Gathering the mana...")
    print("Applying the enchantment...")
    print("Engaging the arcane energies...")
    print("Engraving the rune from aether to stone...")
    print("(This may take up to 15 seconds)")

    # Example usage
    prompt = "Rune magic, arcane symbol, runework, gemstone, central sigil, ancient arcane language, modern pixel video game style icon, engraved in the aether and onto stone, magical energy"
    generated_image = generate_image(prompt)

    # Apply circular mask
    mask = create_circular_mask(generated_image)
    circular_image = ImageOps.fit(generated_image, mask.size, centering=(0.5, 0.5))
    circular_image.putalpha(mask)

    # Convert the image to RGBA mode
    circular_image = circular_image.convert('RGBA')

    # Get the image size to set the window size
    image_width, image_height = circular_image.size
    file_extension = os.path.splitext(base_filename)[1]

    # Use SANCTUM_PATH for .runes directory
    runes_dir = os.path.join(SANCTUM_PATH, '.runes')
    rune_subdir = base_filename.rsplit('.', 1)[0]
    rune_dir = os.path.join(runes_dir, rune_subdir)
    os.makedirs(rune_dir, exist_ok=True)

    # Save the generated image as a PNG
    image_file = base_filename.rsplit('.', 1)[0] + '_image.png'
    image_full_path = os.path.join(rune_dir, image_file)
    circular_image.save(image_full_path, format="PNG")


    # Define a dictionary that maps file extensions to commands
    extension_to_command = {
        '.sh': 'bash',
        '.py': 'python',
        '.spell': 'cast'
    }

    # Get the command for the input file's extension
    command = extension_to_command.get(file_extension, '')

    # Handle the case where the file extension is not supported
    if not command:
        print(f"File extension '{file_extension}' is not supported.")
        return

    gui_file_name = base_filename.rsplit('.', 1)[0] + '_gui.py'
    image_file = base_filename.rsplit('.', 1)[0] + '_image.png'


    # Save the generated image as a PNG in the rune directory
    circular_image.save(os.path.join(rune_dir, image_file), format="PNG")

    code = f'''
import sys
import subprocess
import signal
from PyQt5.QtGui import QPixmap, QColor
from PyQt5.QtWidgets import QGraphicsDropShadowEffect, QApplication, QLabel, QVBoxLayout, QWidget, QPushButton
from PyQt5.QtCore import Qt

# Ignore SIGINT
signal.signal(signal.SIGINT, signal.SIG_IGN)

class ImageButton(QLabel):
    def __init__(self, image_path, command, path, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pixmap = QPixmap(image_path)
        self.command = command
        self.path = path
        self.moved = False

        # Create a QGraphicsDropShadowEffect
        shadow = QGraphicsDropShadowEffect()
        shadow.setBlurRadius(15)  # Adjust the blur radius
        shadow.setXOffset(5)  # Adjust the horizontal offset
        shadow.setYOffset(5)  # Adjust the vertical offset
        shadow.setColor(QColor("black"))

        # Apply the shadow effect to the image_button
        self.setGraphicsEffect(shadow)

    def mousePressEvent(self, event):
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        self.moved = True
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton and not self.moved:
            subprocess.run([self.command, self.path])
        self.moved = False
        super().mouseReleaseEvent(event)

class DraggableWindow(QWidget):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mpos = None

    def mousePressEvent(self, event):
        self.mpos = event.pos()

    def mouseMoveEvent(self, event):
        if self.mpos:
            diff = event.pos() - self.mpos
            new_pos = self.pos() + diff
            self.move(new_pos)

    def mouseReleaseEvent(self, event):
        self.mpos = None
        self.findChild(ImageButton).moved = False

app = QApplication(sys.argv)

window = DraggableWindow()
window.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint)
window.setAttribute(Qt.WA_TranslucentBackground)

layout = QVBoxLayout(window)
layout.setAlignment(Qt.AlignCenter) # Center alignment

image_button = ImageButton(r"{image_full_path}", "{command}", r"{file_path}")
layout.addWidget(image_button)

# Add a close button
close_button = QPushButton("X", window)
close_button.clicked.connect(app.quit)
close_button.setStyleSheet(\"""
    QPushButton {{
        color: teal; 
        background-color: black; 
        border-radius: 10px; 
        font-size: 12px; 
        padding: 2px;
    }}
    QPushButton:hover {{
        background-color: #ff7f7f;
    }}
\""")
# Set button size
close_button.setFixedSize(20, 20)

# Calculate the new size for the window and image by halving the current size
new_size = window.size() * 0.25

# Scale the image to the new size while maintaining aspect ratio
scaled_pixmap = image_button.pixmap.scaled(new_size, Qt.KeepAspectRatio, Qt.SmoothTransformation)

# Set the scaled pixmap as the image button's pixmap
image_button.setPixmap(scaled_pixmap)

# Resize the window to the new size
window.resize(new_size)

# Calculate the position to center the image within the window
image_pos = window.rect().center() - image_button.rect().center()
image_button.move(image_pos)

# Move the close button to the top left
close_button.move(110, 0)

window.show()

sys.exit(app.exec_())

'''

    # Write the PyQt script to a file in the rune directory
    with open(os.path.join(rune_dir, gui_file_name), 'w') as f:
        f.write(code)

    print("The rune is complete. You may now cast it.")

    # Now run the new file
    subprocess.Popen(["python", os.path.join(rune_dir, gui_file_name)], start_new_session=True)

alias = "rc"

if __name__ == '__main__':
    runecraft()
