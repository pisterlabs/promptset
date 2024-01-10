import openai
import time
import os

openai.api_key = "API_KEY_ANDA"
openai.api_base = "ALAMAT_END_POINT"

messages = [{"role": "assistant", "content": "Anda adalah asisten cerdas."}]

def print_red(text):
    print("\033[91m" + text + "\033[0m")  # ANSI escape code untuk teks merah

def print_green(text):
    print("\033[92m" + text + "\033[0m")  # ANSI escape code untuk teks hijau

def clear_screen():
    """Membersihkan layar CMD."""
    # Untuk Windows
    if os.name == "nt":
        os.system("cls")
    # Untuk macOS atau Linux
    else:
        os.system("clear")

while True:
    message = input("\033[91mHaikal : \033[0m")  # Input pengguna dalam warna merah
    if message.lower() == "stop":
        print("Program berhenti.")
        break

    # Menambahkan kondisi untuk membersihkan layar
    if message.lower() == "clear":
        clear_screen()
        continue  # Melanjutkan ke iterasi berikutnya tanpa menjalankan logika chat

    # Menambahkan kondisi untuk menanggapi pertanyaan khusus "siapa saya?"
    if message.lower() in ["siapa saya?", "siapa saya"]:
        reply = "Anda adalah Haikal, Orang Paling Ganteng"
        print_green(f"\nEntitas-Pintar: \n\n{reply}\n")  # Menampilkan jawaban khusus
        messages.append({"role": "assistant", "content": reply})
        continue  # Melanjutkan ke iterasi berikutnya tanpa menjalankan logika chat

    if message.lower() == "ping":
        # Menjalankan perintah ping untuk memeriksa koneksi internet
        response = os.system("ping google.com")
        if response == 0:
            reply = "Koneksi internet terhubung."
        else:
            reply = "Koneksi internet terputus."
        print_green(f"\nEntitas-Pintar: \n\n{reply}\n")  # Menampilkan jawaban khusus
        messages.append({"role": "assistant", "content": reply})
        continue  # Melanjutkan ke iterasi berikutnya tanpa menjalankan logika chat

    if message:
        messages.append({"role": "user", "content": message})

        # Menambahkan penundaan beberapa detik di antara permintaan
        time.sleep(3)

        # Membuat panggilan API ke OpenAI untuk chat completion
        chat = openai.ChatCompletion.create(
            model="gpt-3.5-turbo", 
            messages=messages, 
            # Tambahkan parameter untuk bahasa Indonesia
            model_parameters={"temperature": 1.0, "n": 1, "stop": None, "role": "system", "language": "indonesian"}
        )
        reply = chat.choices[0].message.content

        # Periksa apakah balasan adalah blok kode
        if "```" in reply:
            # Membersihkan layar CMD
            clear_screen()
            print_green(f"\nEntitas-Pintar (Kode): {reply}")  # Balasan asisten dalam warna hijau
        else:
            print_green(f"\nEntitas-Pintar: \n\n{reply}\n")  # Balasan asisten dalam warna hijau

        messages.append({"role": "assistant", "content": reply})
