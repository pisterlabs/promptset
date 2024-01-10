import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
from concurrent.futures import ThreadPoolExecutor
import requests
import re
import json
import io
import base64
from ast import literal_eval
from itertools import cycle
from urllib.parse import quote_plus
import ast
import random
import schedule

import time
import glob
import datetime
from tqdm import tqdm

from selenium import webdriver
from bs4 import BeautifulSoup
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
import openai
from retry import retry


ChromeDriverManager().install()

url = 'https://aibusiness.com/'
key = random.choice(['ai','chatbot'])
categories = {'ai':70,'chatbot':57}
page=1
all_artikel = []
new_urls = []
scrapped_url = []

api_key = "<ganti_dengan_api_key_open_ai_kamu>"

def clean_hashtag(text):
    """
    Menghapus hashtag (#) dari teks.

    Parameters:
    text (str): Teks yang akan dihapus hashtag-nya.

    Returns:
    str: Teks hasil penghapusan hashtag.
    """
    text = re.sub(r'\#\S+', '', text)
    return text

def clean_url_from_text(text):
    """
    Menghapus URL dari teks.

    Parameters:
    text (str): Teks yang akan dihapus URL-nya.

    Returns:
    str: Teks hasil penghapusan URL.
    """
    text = re.sub(r'(https?:\/\/)(\s)*(www\.)?(\s)*((\w|\s)+\.)*([\w\-\s]+\/)*([\w\-]+)((\?)?[\w\s]*=\s*[\w\%&]*)*', '', text)
    text = re.sub(r'\S+\.\S+', '', text)
    return text

def clean_multiple_spaces(text):
    """
    Menghapus spasi berlebih dari teks.

    Parameters:
    text (str): Teks yang akan dihapus spasi berlebih.

    Returns:
    str: Teks hasil penghapusan spasi berlebih.
    """
    text = text.replace("\n", " ")
    text = text.replace("\r", " ")
    text = re.sub(' +', ' ', text)
    return text

def remove_non_ascii(text):
    """
    Menghapus karakter non-ASCII dari teks.

    Parameters:
    text (str): Teks yang akan dihapus karakter non-ASCII.

    Returns:
    str: Teks hasil penghapusan karakter non-ASCII.
    """
    return re.sub(r'[^\x00-\x7F]', ' ', text)


new_urls = []

def get_all_link(url):
    """
    Mendapatkan semua link artikel dari URL yang diberikan.

    Parameters:
    url (str): URL halaman yang berisi daftar artikel.

    Returns:
    BeautifulSoup: Objek BeautifulSoup yang berisi HTML dari halaman web.
    """
    driver = webdriver.Chrome()
    driver.get(url)
    elem = driver.find_element(By.TAG_NAME, "body")
    inner_html = elem.get_attribute('innerHTML')

    soup = BeautifulSoup(inner_html, 'html.parser')
    return soup

def get_link(soup):
    """
    Mendapatkan daftar link artikel dari objek BeautifulSoup.

    Parameters:
    soup (BeautifulSoup): Objek BeautifulSoup yang berisi HTML halaman web.

    Returns:
    list: Daftar link artikel.
    """
    link_art = []
    links = soup.findAll('a', attrs={'class': 'ListPreview-Title'})
    for link in links:
        href = url[:-1] + link.get('href')
        if href:
            link_art.append(href)
    fin_link = [i for n, i in enumerate(link_art) if i not in link_art[:n]]
    return fin_link

def get_artikel(url):
    """
    Mendapatkan konten artikel dari URL yang diberikan.

    Parameters:
    url (str): URL artikel.

    Returns:
    BeautifulSoup: Objek BeautifulSoup yang berisi HTML dari halaman artikel.
    """
    driver = webdriver.Chrome()
    driver.get(url)

    body = driver.find_element(By.TAG_NAME, "body")
    inner_html = body.get_attribute('innerHTML')
    soup = BeautifulSoup(inner_html, 'html.parser')
    return soup

def res_artikel(soup, url, scrapped_url):
    """
    Mengekstrak informasi artikel dari objek BeautifulSoup dan mengembalikannya dalam bentuk kamus.

    Parameters:
    soup (BeautifulSoup): Objek BeautifulSoup yang berisi HTML halaman artikel.
    url (str): URL artikel.
    scrapped_url (list): Daftar URL yang sudah di-scrap sebelumnya.

    Returns:
    dict: Kamus yang berisi informasi artikel seperti judul, tanggal, URL, sumber, dan konten.
    """
    title = soup.find('span', attrs={"class": "ArticleBase-LargeTitle"}).text
    title = clean_multiple_spaces(remove_non_ascii(title))
    date = soup.find('p', attrs={"class": 'Contributors-Date'}).text
    date = clean_multiple_spaces(remove_non_ascii(date))

    if not date:
        date = pd.NaT
    artikel = ''
    header_summary = soup.find('p', attrs={"class": "ArticleBase-HeaderSummary"})
    if header_summary:
        artikel += header_summary.text + '.\n'

    content_div = soup.find('div', attrs={"data-module": "content"})
    if content_div:
        paragraphs = content_div.findAll('span')
        for paragraph in paragraphs:
            artikel += paragraph.text + '\n'

    artikel = clean_multiple_spaces(remove_non_ascii(clean_url_from_text(clean_hashtag(artikel)))).replace("''", '""')

    article_dict = {
        "title": title,
        "date": date,
        "url": url,
        "source": "aibusiness.com",
        "content": artikel
    }

    if url not in scrapped_url:
        return article_dict
    else:
        return None

def get_list_links(results_urls):
    """
    Fungsi ini menerima daftar URL hasil pencarian dan mengembalikan daftar link artikel.
    """
    article_link = []
    for i in range(len(results_urls)):
        link = get_all_link(results_urls[i])
        mylinks = get_link(link)
        article_link.append(mylinks)
    article_link = np.array(article_link).flatten().tolist()
    return article_link

def find_artikel(urls, scrapped_url):
    """
    Fungsi ini menerima daftar URL artikel dan mencari artikelnya.

    Parameters:
    urls (list): Daftar URL artikel.
    scrapped_url (list): Daftar URL yang sudah di-scrap sebelumnya.

    Returns:
    list: Daftar artikel yang ditemukan.
    """
    global all_artikel, new_urls

    for i, url in enumerate(urls):
        a_url = get_artikel(url)
        a_soup = res_artikel(a_url, url, scrapped_url)
        if a_soup is None:
            break  # Menghentikan perulangan jika artikel yang ditemukan memenuhi kondisi
        all_artikel.append(a_soup)
        new_urls.append(url)
        print(f"Processed article {i+1}: {url}")
        break  # Menghentikan perulangan setelah menemukan satu artikel


def search_url():
    """
    Fungsi ini menghasilkan daftar URL hasil pencarian berdasarkan query dan jumlah halaman.
    """
    results_urls = []
    query = quote_plus(key)
    base_url = url + 'search?q=' + query+'&sort=newest'
    for i in range(1, page+1):
        if i == 1:
            results_urls.append(base_url)
        else:
            page_url = base_url + '&page=' + str(i)
            results_urls.append(page_url)
    return results_urls

def save_list_links_to_txt(list_link):
    """
    Menyimpan seluruh daftar link ke dalam file txt dengan nama 'aibusiness_links.txt'.

    Parameters:
    list_link (list): Daftar link yang akan disimpan.

    Returns:
    str: Jalur file yang disimpan.
    """
    file_path = 'aibusiness_links.txt'

    # Membaca link-link yang sudah ada dari file
    existing_links = []
    try:
        with open(file_path, 'r') as file:
            existing_links = [line.strip() for line in file]
    except FileNotFoundError:
        pass

    # Menggabungkan link-link yang sudah ada dan link-link baru dengan list_link di awal
    all_links = [list_link[-1]] + existing_links

    # Menulis semua link unik ke dalam file dengan urutan yang sama
    with open(file_path, 'w') as file:
        for link in all_links:
            file.write(link + '\n')

    return file_path
    


def read_list_links(file_path):
    """
    Membaca file teks yang berisi daftar URL dan mengembalikan daftar URL.

    Parameters:
    file_path (str): Jalur file teks yang berisi daftar URL.

    Returns:
    list: Daftar URL.
    """
    scrapped_url = []
    try:
        with open(file_path, 'r') as file:
            for line in file:
                url = line.strip()
                scrapped_url.append(url)
    except FileNotFoundError:
        return None
    return scrapped_url


def scrap_aibusiness(url='https://aibusiness.com/', key='ai', page=1):
    """
    Fungsi ini melakukan scraping pada situs aibusiness.com berdasarkan URL, kata kunci, dan jumlah halaman yang ditentukan.

    Parameters:
    url (str): URL halaman yang berisi daftar artikel. Default: 'https://aibusiness.com/'.
    key (str): Kata kunci pencarian artikel. Default: 'ai'.
    page (int): Jumlah halaman pencarian yang akan di-scrap. Default: 1.

    Returns:
    list: Daftar artikel yang ditemukan.
    """
    global all_artikel, new_urls, scrapped_url
    all_artikel = []

    results_urls = search_url()
    scrapped_url = read_list_links('aibusiness_links.txt')

    if scrapped_url is None:
        scrapped_url = []

    all_links = get_list_links(results_urls)
    new_links = [link for link in all_links if link not in scrapped_url]
    print(all_links)
    if not new_links:
        print("Tidak ada artikel terbaru yang ditemukan...")
        return None

    find_artikel(new_links, scrapped_url)

    if all_artikel:
        return all_artikel[0]
    else:
        print("Tidak ada artikel yang ditemukan...")
        return None


@retry(openai.error.OpenAIError, tries=20, delay=15)
def rewrite(teks):
    """
    Fungsi untuk melakukan penulisan ulang (rewrite) artikel dalam bahasa Inggris menjadi artikel dalam bahasa Indonesia dengan teknik SEO Optimized menggunakan model gpt-3.5-turbo dari OpenAI.

    Parameters:
        teks (str): Teks artikel dalam bahasa Inggris yang akan ditulis ulang.

    Returns:
        str: Judul artikel yang telah diterjemahkan.
        str: Konten artikel yang telah ditulis ulang dan SEO Optimized.
        str: Headline dari isi artikel.
        str: Tags yang dipilih berdasarkan artikel.

    Raises:
        openai.error.OpenAIError: Error saat melakukan permintaan ke API OpenAI.
        json.JSONDecodeError: Error saat melakukan parsing JSON.

    """
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "system",
                    "content": "Kamu adalah mesin penerjemah bahasa Inggris ke bahasa Indonesia yang handal, kamu juga mampu menulis ulang artikel sekaligus melakukan SEO Optimized dengan luar biasa. jika artikel yang diberikan lebih dari 5000 kata maka kamu harus membuat artikelnya menjadi lebih padat dengan minimal output artikel 3000 kata dan maksimal 5000 kata sehingga lebih padat dan jelas!. Generate a JSON that can be parsed using the json.loads() function in Python, This json must escaped escape characters like \n and \", YOU MUST ADD triple backslash (\\\") for any word that have double quotes("") you found"
                },
                {
                    "role": "user",
                    "content": """OUTPUT YANG KAMU BERI TIDAK BOLEH KURANG DARI PANJANG ARTIKEL AWAL, Lakukan SEO Optimized dan terjemahkan ke dalam bahasa Indonesia.
                    Berikut artikel yang harus kamu eksekusi: """ + teks + """
                    Gunakan format dibawah ini dengan hanya menampilkan output berupa format JSON yang benar, Generate a JSON that can be parsed using the json.loads() function in Python, This json must escaped escape characters like \n and \", YOU MUST ADD triple backslash (\\\") for any word that have double quotes("") you found in konten :
                    {
                        "Judul": "Judul yang telah diterjemahkan (Judul adalah kalimat pertama dalam teks)",
                        "Headline": "<h1> Headline dari isi artikel(buatlah 1 kalimat topik dari artikel yang isinya berbeda dengan judul)</h1>",
                        "Konten": "Konten hasil rewrite yang telah SEO Optimized dan terlihat penulis professional, Generate a JSON that can be parsed using the json.loads() function in Python, This json must escaped escape characters like \n and \", YOU MUST ADD triple backslash (\\\") for any word that have double quotes("") you found in konten",
                        "Tags": "selected tags from this list based on corresponding article: AI, iot, Chatbot, bot whatsapp, ecommerce, omni, bot penjualan. if AI or iot please convert output to [10, 11], if chatbot or bor whatsapp, or bot penjualan convert output to [13,42], else convert to []"
                    }
                    Ensure the response can be parsed by Python json.loads with correct JSON syntax anytime. KAMU DILARANG MENGURANGI PANJANG DARI TEKS AWAL YANG DIBERIKAN"""
                }
            ],
            temperature=0,
            max_tokens=None,  
            n=1,
            stop=None,
            api_key=api_key
        )

        try:
            response_content = response.choices[0].message.content.strip()
            hasil_rewrite = json.loads(response_content, strict=False)
            judul = hasil_rewrite['Judul']
            fin_konten = hasil_rewrite['Konten']
            headline = hasil_rewrite['Headline']
            tags = hasil_rewrite['Tags']
        except json.JSONDecodeError:
            print('kesalahan json decode')
            return '', '', '', ''

        while response.choices[0].finish_reason != 'stop':
            print('tidak berhasil berhenti di looping pertama pada fungsi rewrite')
            continuation_token = response.choices[0].get('model_continuation', {}).get('index')
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {
                        "role": "system",
                        "content": "Kamu adalah mesin penerjemah bahasa Inggris ke bahasa Indonesia yang handal, kamu juga mampu menulis ulang artikel sekaligus melakukan SEO Optimized dengan luar biasa, KAMU DILARANG mengurangi panjang dan mengubah struktur artikel yang diberikan. dan Yang terpenting Terapkan Semua Perintah Saya Pada Semua Artikel Yang Diberikan!!!"
                    },
                    {
                        "role": "user",
                        "content": teks,
                    }
                ],
                temperature=0,
                max_tokens=None,  
                n=1,
                stop=None,
                api_key=api_key,
                model_continuation={'index': continuation_token}
            )

            response_content = response.choices[0].message.content.strip()
            try:
                hasil_rewrite = json.loads(response_content, strict=False)
                fin_konten += hasil_rewrite['Konten']
            except json.JSONDecodeError:
                print('kesalahan json decode')
                return '', '', '', ''

        return judul, fin_konten, headline, tags
    except openai.error.OpenAIError as e:
        return '', '', '', ''

    
@retry(openai.error.OpenAIError, tries=10, delay=10)
def format_html(teks):
    """
    Fungsi untuk melakukan penyuntingan teks dalam format HTML dengan menggunakan model gpt-3.5-turbo dari OpenAI.

    Parameters:
        teks (str): Teks yang akan disunting dalam format HTML.

    Returns:
        str: Teks yang telah disunting dalam format HTML.

    Raises:
        openai.error.OpenAIError: Error saat melakukan permintaan ke API OpenAI.

    """
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "system",
                    "content": "kamu adalah seorang penulis artikel professional. Kamu harus perhatikan dan mematuhi detail instruksi yang diberikan oleh user. Kamu Dilarang mengurangi atau bahkan menambahkan konten teks apapun ke dalam artikel."
                },
                {
                    "role": "user",
                    "content": "Lakukan penyuntingan pada artikel yang saya berikan:\n" + teks + "\nEdit artikel di atas dengan menambahkan penandaan terhadap kata-kata yang mengandung keyword ai, omnichannel, dan chatbot untuk diformat menjadi link pada struktur html dengan ketentuan sebagai berikut:\n- Jika 'ai', maka link akan terhubung ke https://botika.online/\n- Jika 'chatbot', link akan terhubung ke https://botika.online/chatbot-gpt/index.php\n- Jika 'omnichannel', link terhubung ke https://omni.botika.online/\nFormatnya harus seperti ini: <a href=\"{link}\">{keyword}</a>\nYOU MUST Do this FORMAT for the first 3 keywords that appear and MUST be on different keywords IF a keyword appears two or more times then it is ignored DO NOT NEED TO BE FORMATTED.\nDo not write any explanation and any pleasantries. Respond only with the new formatted article."
                }
            ],
            temperature=0,
            max_tokens=None,
            n=1,
            stop=None,
            api_key=api_key
        )

        hasil_format = response.choices[0].message.content.strip()

        while response.choices[0].finish_reason != 'stop':
            print('tidak berhasil berhenti di looping pertama pada fungsi format_html')
            continuation_token = response.choices[0].get('model_continuation', {}).get('index')
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {
                        "role": "system",
                        "content": "kamu adalah seorang penulis artikel professional. Kamu harus perhatikan dan mematuhi detail instruksi yang diberikan oleh user. Kamu Dilarang mengurangi atau bahkan menambahkan konten teks apapun ke dalam artikel."
                    },
                    {
                        "role": "user",
                        "content": teks,
                    }
                ],
                temperature=0,
                max_tokens=None,
                n=1,
                stop=None,
                api_key=api_key,
                model_continuation={'index': continuation_token}
            )

            hasil_format += response.choices[0].message.content.strip()

        return hasil_format
    except openai.error.OpenAIError as e:
        return 'server sibuk'



def bold_underline(teks):
    """
    Sebuah fungsi yang melakukan penyuntingan pada artikel dengan menambahkan tag HTML bold (menebalkan kata) dan underline (garis bawah) pada istilah-istilah atau kata asing dalam bahasa asing (di luar bahasa Indonesia) yang ditemukan.

    Parameter:
    - teks (string): Artikel yang akan disunting.

    Output:
    - hasil_format (string): Artikel yang telah diformat dengan penambahan tag HTML bold dan underline pada istilah-istilah atau kata asing yang ditemukan. Artikel ini merupakan hasil penyuntingan dari teks yang diberikan.

    Contoh Penggunaan:
    teks = "Ini adalah contoh artikel yang perlu disunting. Terdapat beberapa istilah asing seperti <b>example</b> dan <b>foreign word</b> yang perlu diberi format bold dan underline."
    hasil_format = bold_underline(teks)
    print(hasil_format)
    """

    # Kode fungsi bold_underline
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "kamu adalah seorang editor artikel professional. Kamu harus perhatikan dan mematuhi detail instruksi yang diberikan oleh user. Kamu memiliki keterbatasan untuk tidak boleh mengurangi atau bahkan menambahkan konten teks apapun ke dalam artikel."},
                {"role": "user", "content": "Lakukan penyuntingan pada artikel yang saya berikan:\n" + teks + "\nEdit  artikel diatas dengan dengan memberikan tag html bold(menebalkan kata) <b> bersamaan dengan tag html underline (garis bawah) <u> pada semua istilah-istilah atau kata asing maupun bahasa asing (di luar bahasa indonesia) yang kamu temukan.\nYOU MUST Do this FORMAT for the first 3 terms that appear and MUST be in a different term. IF THE SAME term appears twice or more, Ignore it, NO NEED TO BE FORMATTED.\n YOU MUST REFORMAT THE ARTICLE OUTPUT INTO AT LEAST 2 PARAGRAPH WITH HTML TAG.\nDo not write any explanation and any pleasantries. Respond only with the new formatted article using this format: {new formatted article}"}
            ],
            temperature=0,
            max_tokens=None,
            n=1,
            stop=None,
            api_key=api_key
        )
        hasil_format = response.choices[0].message.content.strip()

        while response.choices[0].finish_reason != 'stop':
            print('tidak berhasil berhenti di looping pertama pada fungsi bold_underline')
            continuation_token = response.choices[0].get('model_continuation', {}).get('index')
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "kamu adalah seorang editor artikel professional. Kamu harus perhatikan dan mematuhi detail instruksi yang diberikan oleh user. Kamu memiliki keterbatasan untuk tidak boleh mengurangi atau bahkan menambahkan konten teks apapun ke dalam artikel."},
                    {"role": "user", "content": teks}
                ],
                temperature=0,
                max_tokens=None,
                n=1,
                stop=None,
                api_key=api_key,
                model_continuation={'index': continuation_token}
            )

            hasil_format += response.choices[0].message.content.strip()

        return hasil_format
    except openai.error.OpenAIError as e:
        return 'server sibuk'



def convert_to_minutes_seconds(seconds):
        """
        Mengonversi jumlah detik menjadi menit dan detik.

        Args:
            seconds (int): Jumlah total detik.

        Returns:
            str: String yang mewakili menit dan detik dalam format "{M} menit {S} detik".

        Contoh:
            >>> seconds = 301
            >>> result = convert_to_minutes_seconds(seconds)
            >>> print(result)
            "5 menit 1 detik"
        """
        minutes = seconds // 60
        remaining_seconds = seconds % 60

        result = f"{minutes} menit {remaining_seconds} detik"
        return result


def start_scrap():
    # Mulai program
    # Catat waktu awal
    start_time = time.time()

    current_time = datetime.datetime.now()
    formatted_time = current_time.strftime("%Y-%m-%d %H:%M:%S")

    print("Waktu menjalankan program saat ini:", formatted_time)
    global key, categories
    key = random.choice(['ai','chatbot'])
    categories = {'ai': 70, 'chatbot': 57}
    all_artikel = scrap_aibusiness()
    if all_artikel is not None:
        # Perulangan untuk mentranslate judul dan konten
        print('Memulai membuat konten.........')
        trans_rewrite = all_artikel['title'] + ' ' + all_artikel['content']

        # Parafrase konten
        judul, konten, tags, headline = rewrite(trans_rewrite)

        # Memformat bold dan underline serta memberikan tag link
        new_konten = format_html(konten)
        new_konten = bold_underline(new_konten)

        # Mengganti judul dan konten dengan hasil terjemahan
        all_artikel['rewrite_title'] = judul
        all_artikel['rewrite_content'] = headline + ' ' + new_konten
        all_artikel['tags'] = tags

        # Post ke API Botika
        # ...

        while len(all_artikel['rewrite_content']) < 2700:
            print(all_artikel['rewrite_content'])
            print('PANJANG ARTIKEL DIBAWAH 2700')
            save_list_links_to_txt(new_urls)
            all_artikel = scrap_aibusiness()
            trans_rewrite = all_artikel['title'] + ' ' + all_artikel['content']
            judul, konten, tags, headline = rewrite(trans_rewrite)
            new_konten = format_html(konten)
            new_konten = bold_underline(new_konten)
            all_artikel['rewrite_title'] = judul
            all_artikel['rewrite_content'] = headline + ' ' + new_konten
            all_artikel['tags'] = tags

        print("PANJANG ARTIKEL DI ATAS 2700 ")

    save_list_links_to_txt(new_urls)

    file_path = "artikel.json"
    with open(file_path, "w") as file:
        json.dump(all_artikel, file)

    current_time = datetime.datetime.now()
    formatted_time = current_time.strftime("%Y-%m-%d %H:%M:%S")

    print("Waktu selesai menjalankan program saat ini:", formatted_time)

    end_time = time.time()

    # Hitung lama waktu eksekusi
    execution_time = end_time - start_time

    # Tampilkan lama waktu eksekusi
    execution_time_minutes_seconds = convert_to_minutes_seconds(int(execution_time))
    print(f"Lama waktu eksekusi: {execution_time_minutes_seconds}")


start_scrap()
