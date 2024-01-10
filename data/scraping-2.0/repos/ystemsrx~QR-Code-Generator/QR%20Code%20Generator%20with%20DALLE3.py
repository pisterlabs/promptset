import tkinter as tk
from tkinter import messagebox, colorchooser, filedialog
import qrcode
from PIL import Image, ImageTk
from openai import OpenAI
import requests

client = OpenAI()

def add_logo_to_qr(qr_code, logo_path, logo_size=(50, 50)):
    # Load logo and resize
    logo = Image.open(logo_path)
    logo = logo.resize(logo_size, Image.Resampling.LANCZOS)

    # Calculate position to place logo
    qr_size = qr_code.size
    logo_position = ((qr_size[0] - logo_size[0]) // 2, (qr_size[1] - logo_size[1]) // 2)

    # Embed logo into QR code
    qr_with_logo = qr_code.copy()
    qr_with_logo.paste(logo, logo_position, logo.convert('RGBA'))

    return qr_with_logo

# Main window class
class QRCodeGenerator(tk.Tk):
    def __init__(self):
        super().__init__()
        self.file_name = None

        self.english_text = {
            "language_label": "Language:",
            "url_label": "URL:",
            "bg_color_label": "Background Color",
            "qr_color_label": "QR Code Color",
            "preview_label": "Preview",
            "save_button": "Save",
            "edit_color_button": "Edit Color",
            "cancel_button": "Cancel",
            "chinese_rb": "Chinese (simplified)",
            "english_rb": "English",
            "quantity_label": "Quantity",
            "logo_path": "Logo Path",
            "select_logo": "Select Logo(Preview)",
            "background_label": "AI Background (Preview)",
            "generate_button": "Generate",
            "remove_button": "Remove"
        }

        self.chinese_text = {
            "language_label": "语言:",
            "url_label": "网址:",
            "bg_color_label": "背景颜色",
            "qr_color_label": "二维码颜色",
            "preview_label": "预览",
            "save_button": "保存",
            "edit_color_button": "编辑颜色",
            "cancel_button": "取消",
            "chinese_rb": "简体中文",
            "english_rb": "英文",
            "quantity_label": "数量",
            "logo_path": "Logo路径",
            "select_logo": "选择Logo(预览)",
            "background_label": "AI背景 (预览)",
            "generate_button": "生成",
            "remove_button": "移除"
        }

        self.title('QR Code Generator')
        # self.geometry('550x550')

        # Set initial language
        self.current_language = self.english_text

        # Language Selection
        self.language_var = tk.StringVar(value="English")

        self.language_label = tk.Label(self, text=self.current_language["language_label"])
        self.language_label.grid(row=0, column=0, sticky="w")

        self.chinese_rb = tk.Radiobutton(self, text=self.current_language["chinese_rb"],
                                         variable=self.language_var, value="Chinese",
                                         command=self.switch_language)
        self.chinese_rb.grid(row=0, column=1)

        self.english_rb = tk.Radiobutton(self, text=self.current_language["english_rb"],
                                         variable=self.language_var, value="English",
                                         command=self.switch_language)
        self.english_rb.grid(row=0, column=2)

        # URL Entry
        self.url_label = tk.Label(self, text=self.current_language["url_label"])
        self.url_label.grid(row=1, column=0, sticky="w")

        self.url_entry = tk.Entry(self)
        self.url_entry.grid(row=1, column=1, columnspan=2, sticky="we")
        self.url_entry.bind("<KeyRelease>", self.generate_preview)

        # Color Selection
        self.bg_color_var = tk.StringVar(value="white")
        self.qr_color_var = tk.StringVar(value="black")

        # Background Color Button
        self.bg_color_label = tk.Label(self, text=self.current_language["bg_color_label"])
        self.bg_color_label.grid(row=2, column=0, sticky="w")

        self.bg_color_button = tk.Button(self, bg=self.bg_color_var.get(), width=2,
                                         command=lambda: self.choose_color(self.bg_color_var))
        self.bg_color_button.grid(row=2, column=1, sticky="we")

        # QR Code Color Button
        self.qr_color_label = tk.Label(self, text=self.current_language["qr_color_label"])
        self.qr_color_label.grid(row=3, column=0, sticky="w")

        self.qr_color_button = tk.Button(self, bg=self.qr_color_var.get(), width=2,
                                         command=lambda: self.choose_color(self.qr_color_var))
        self.qr_color_button.grid(row=3, column=1, sticky="we")

        # Preview Label (placeholder for the QR code image)
        self.preview_label = tk.Label(self, text=self.current_language["preview_label"])
        self.preview_label.grid(row=4, column=0, columnspan=3)

        # Save Button
        self.save_button = tk.Button(self, text=self.current_language["save_button"], command=self.save_qr_code)
        self.save_button.grid(row=5, column=1, sticky="we")

        # Quantity Entry
        self.quantity_label = tk.Label(self, text=self.current_language["quantity_label"])
        self.quantity_label.grid(row=7, column=0, sticky="w")

        self.quantity_entry = tk.Entry(self)
        self.quantity_entry.grid(row=7, column=1, columnspan=2, sticky="we")
        self.quantity_entry.insert(0, "1")  # Default quantity

        # Logo File Selection
        self.logo_label = tk.Label(self, text=self.current_language["logo_path"])
        self.logo_label.grid(row=6, column=0, sticky="w")  # Adjust the row and column accordingly

        self.logo_entry = tk.Entry(self)
        self.logo_entry.grid(row=6, column=1)

        self.logo_button = tk.Button(self, text=self.current_language["select_logo"], command=self.choose_logo)
        self.logo_button.grid(row=6, column=2)

        self.background_label = tk.Label(self, text=self.current_language["background_label"])
        self.background_label.grid(row=8, column=0, sticky="w")  # Adjust the row and column accordingly

        self.background_entry = tk.Entry(self)
        self.background_entry.grid(row=8, column=1)

        self.generate_button = tk.Button(self, text=self.current_language["generate_button"],command=self.generate_button_clicked)
        self.generate_button.grid(row=8, column=2)

        self.remove_button = tk.Button(self, text=self.current_language["remove_button"],command=self.remove_button_clicked)
        self.remove_button.grid(row=8, column=3)

    def remove_button_clicked(self):
    # 清除背景图像设置
        self.file_name = None  # 将背景文件名设置为空
        self.generate_preview()  # 重新生成预览

    def generate_button_clicked(self):
        
        self.background_content = self.background_entry.get()

        self.generate_button.config(state='disabled', text='Generating')

        response = client.images.generate(
        model="dall-e-3",
        prompt=f"Generate a background image with low color concentration, the content is: {self.background_content}",
        size="1024x1024",
        quality="standard",
        n=1,
        )
        image_url = response.data[0].url
        self.file_name = f"{self.background_content}.png"

        response = requests.get(image_url)
        if response.status_code == 200:
            with open(self.file_name, 'wb') as file:
                file.write(response.content)
            self.generate_button.config(state='normal', text='Generate')
            self.generate_preview()  # 调用预览函数
            messagebox.showinfo("Success", "Successfully create the background.")
        else:
            print("Error: Unable to download the image.")
            messagebox.showinfo("Error", "Unable to download the image.")
    
    def switch_language(self):
        # Switch the language text and update labels/buttons
        language = self.language_var.get()
        self.current_language = self.chinese_text if language == "Chinese" else self.english_text
        self.language_label.config(text=self.current_language["language_label"])
        self.url_label.config(text=self.current_language["url_label"])
        self.bg_color_label.config(text=self.current_language["bg_color_label"])
        self.qr_color_label.config(text=self.current_language["qr_color_label"])
        self.save_button.config(text=self.current_language["save_button"])
        self.preview_label.config(text=self.current_language["preview_label"])
        self.chinese_rb.config(text=self.current_language["chinese_rb"])
        self.english_rb.config(text=self.current_language["english_rb"])
        self.quantity_label.config(text=self.current_language["quantity_label"])
        self.logo_label.config(text=self.current_language["logo_path"])
        self.logo_button.config(text=self.current_language["select_logo"])
        self.background_label.config(text=self.current_language["background_label"])
        self.generate_button.config(text=self.current_language["generate_button"])
        self.remove_button.config(text=self.current_language["remove_button"])
        self.generate_preview()

    def generate_preview(self, event=None):
        data = self.url_entry.get()
        if data:
            try:
                # 生成二维码
                qr = qrcode.QRCode(
                    version=1,
                    error_correction=qrcode.constants.ERROR_CORRECT_H,
                    box_size=10,
                    border=4,
                )
                qr.add_data(data)
                qr.make(fit=True)

                if self.file_name:
                    # 有背景图像的情况
                    bg_img = Image.open(self.file_name).convert("RGBA")
                    bg_img = bg_img.resize((300, 300), Image.Resampling.LANCZOS)
                    
                    qr_code_img = qr.make_image(fill_color=self.qr_color_var.get(), back_color="transparent").convert("RGBA")
                    position = ((bg_img.width - qr_code_img.width) // 2, (bg_img.height - qr_code_img.height) // 2)
                    bg_img.paste(qr_code_img, position, qr_code_img)
                    final_img = bg_img
                else:
                    # 无背景图像，使用纯色背景的情况
                    qr_code_img = qr.make_image(fill_color=self.qr_color_var.get(), back_color=self.bg_color_var.get())
                    final_img = qr_code_img

                # 显示在 GUI 上
                self.qr_img = ImageTk.PhotoImage(final_img)
                self.preview_label.config(image=self.qr_img, text="")

            except IOError as e:
                print("Error:", e)
                self.preview_label.config(image='', text="Error in generating QR code.")
        else:
            self.preview_label.config(image='', text="No data for QR code.")

    def choose_color(self, color_var):
        # Open a color dialog and set the selected color to the button and variable
        color = colorchooser.askcolor()[1]
        if color:
            color_var.set(color)
            if color_var == self.bg_color_var:
                self.bg_color_button.config(bg=color)
            elif color_var == self.qr_color_var:
                self.qr_color_button.config(bg=color)

            # Generate a new preview with the updated colors
            self.generate_preview()

    def save_qr_code(self):
        base_data = self.url_entry.get()
        logo_path = self.logo_entry.get()  # Get the logo path from the entry widget

        if base_data:
            try:
                quantity = int(self.quantity_entry.get())
            except ValueError:
                messagebox.showerror("Error", "Invalid quantity.")
                return

            base_file_path = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG files", "*.png")])
            if not base_file_path:
                # User canceled the save operation
                return

            for i in range(quantity):
                unique_data = base_data + " " * i

                qr = qrcode.QRCode(
                    version=1,
                    error_correction=qrcode.constants.ERROR_CORRECT_H,
                    box_size=10,
                    border=4,
                )
                qr.add_data(unique_data)
                qr.make(fit=True)
                qr_code_img = qr.make_image(fill_color=self.qr_color_var.get(), back_color="transparent").convert("RGBA")

                if self.file_name:
                    # 如果存在背景图像
                    try:
                        bg_img = Image.open(self.file_name).convert("RGBA")
                        bg_img = bg_img.resize((300, 300), Image.Resampling.LANCZOS)

                        position = ((bg_img.width - qr_code_img.width) // 2, (bg_img.height - qr_code_img.height) // 2)
                        bg_img.paste(qr_code_img, position, qr_code_img)
                        final_img = bg_img
                    except IOError:
                        messagebox.showerror("Error", "Unable to load the background image.")
                        return
                else:
                    # 无背景图像，使用二维码图像
                    final_img = qr_code_img

                if logo_path:
                    # 添加 Logo
                    final_img_with_logo = add_logo_to_qr(final_img, logo_path)
                    file_path = f"{base_file_path}_{i + 1}.png"
                    final_img_with_logo.save(file_path)
                else:
                    # 保存二维码图像
                    file_path = f"{base_file_path}_{i + 1}.png"
                    final_img.save(file_path)

            messagebox.showinfo("Success", f"{quantity} QR Codes saved successfully.")
        else:
            messagebox.showerror("Error", "No base data to encode.")

    def choose_logo(self):
        logo_path = filedialog.askopenfilename(filetypes=[("Image files", "*.png;*.jpg;*.jpeg")])
        if logo_path:
            self.logo_entry.delete(0, tk.END)
            self.logo_entry.insert(0, logo_path)

# Run the application
if __name__ == "__main__":
    app = QRCodeGenerator()
    app.mainloop()
