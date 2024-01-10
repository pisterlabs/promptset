from django.conf import settings
import os
import openai

if settings.OPENAI_API_KEY:
    openai.api_key = settings.OPENAI_API_KEY
else:
    raise Exception('OpenAI API Key not found')

def optimize(prompt, detail):
    o_base = "saya ingin mengoptimasi kursus saya agar lebih baik. saya akan memberikan beberapa masukan dan saran dibawah ini"
    o_long_sentence = "Buatkan semua kalimatnya diperpanjang"
    o_detail = f"Buatkan agar mendetail tendang `{detail}`"
    
    return f"{o_base}. {o_long_sentence}. {o_detail}."
# def get_completions(prompt, duration):
def get_completion2(prompt):
    
    base = f"menbuat panduan kursus dari `{prompt}`.terdiri dari `title`, `type_activity` (`indoor` dan `outdoor`), `theme_activity` (menyesuaikan), `desc`, `duration`, `subtitles`[], didalam `subtitles` ada `topic`, `shortdesc`, `content`, didalam content ada `opening`, `step`[], `closing`"
    
    
    base_prompt = f"Membuat panduan kursus dari `{prompt}` terdiri dari judul utama, sub-judul, ringkasan singkat dari kursus tersebut yang terdiri dari 10 - 25 kata, lama durasi dari kursus tersebut, jumlah sub-judul yang ada, tema kegiatan dari kursus tersebut seperti pendidikan, olahraga, teknologi dan sejenisya (lower-case), tipe kegiatan dari kursus tersebut dengan pilihan terbatas pada indoor, outdoor, hybrid (lower-case). Setiap sub-judul memiliki sebuah materi yang berisi kalimat pembuka yang menjelaskan tentang sub-judul tersebut, panduan langkah demi langkah, kalimat penutup yang dipisahkan oleh baris. Setiap sub-judul memiliki deksripsi yang menjelaskan tentang sub-judul tersebut. Setiap langkah pada panduan dijelaskan secara rinci dan jelas dengan minimal tiap langkah memiliki satu paragraf. Setiap sub-judul memiliki panduan dipisahkan menggunakan baris. Pastikan Setiap kursus memiliki minimal 3 sub-judul. Setiap panduan didalam sub-judul hanya memilki satu panduan didalamnya dan memiliki minimal 3 langkah didalamnya dengan setiap langkah hanya berada pada satu panduan. Setiap langkah pada panduan memiliki nomor serta dipisahkan oleh baris. Buatkan dalam format json dengan title, ringkasan singkat dan sub-judul terpisah. Serta Pastikan nama key durasi sama dengan `duration`, jumlah sub-judul sama dengan `lessons`, jenis kegiatan sama dengan `type_activity` , tipe kegiatan sama dengan `theme_activity` ,judul sama dengan title, ringkasan singkat sama dengan `desc`, sub-judul sama dengan `subtitles` berbentuk list yang didalamnya ada `topic` `content`, judul didalam `subtitles` sama dengan `topic`, deskripsi didalam sub-judul bernama `shortdesc` ,panduan sama dengan `content` benbentuk object (dictionary) berada didalam subtitles, kalimat pembuka pada panduan dengan nama `opening`, langkah pada panduan dengan nama `step` berbentuk list [str, str, ...], kalimat penutup pada panduan dengan nama `closing`. Pastikan output selalu konsisten dan memiliki format json yang selalu sama, nama key json sama dengan nama yang sudah diberikan. output hanya berupa json saja."
    # query = openai.Completion.create(
    query = openai.ChatCompletion.create(
        model=settings.MODEL,
        messages=[{"role": "system", "content": base }],
    )
    print(query.get('choices')[0]['message']['content'])
    # print(query.get('choices')[0]['text'])
    print(query)
    
    # make except if status code != 200
    # response = query.get('choices')[0]['text']
    # print(query)
    response = query.get('choices')[0]['message']['content']
    return response

# def get_completion_for_stream(prompt):
    
#     base_prompt = f"Membuat panduan kursus dari `{prompt}` terdiri dari judul utama, sub-judul, ringkasan singkat dari kursus tersebut, lama durasi dari kursus tersebut, jumlah sub-judul yang ada, jenis kegiatan dari kursus tersebut seperti pendidikan, olahraga, teknologi dan sejenisya, tipe kegiatan dari kursus tersebut dengan pilihan terbatas pada indoor, outdoor, hybrid. Setiap sub-judul memiliki sebuah materi yang berisi kalimat pembuka yang menjelaskan tentang sub-judul tersebut, panduan langkah demi langkah, kalimat penutup yang dipisahkan oleh baris. Setiap sub-judul memiliki deksripsi yang menjelaskan tentang sub-judul tersebut.Setiap langkah pada panduan dijelaskan secara rinci dan jelas dengan minimal tiap langkah memiliki satu paragraf. Setiap sub-judul memiliki panduan dipisahkan menggunakan baris. Pastikan Setiap kursus memiliki minimal 3 sub-judul. Setiap panduan didalam sub-judul hanya memilki satu panduan didalamnya dan memiliki minimal 3 langkah didalamnya dengan setiap langkah hanya berada pada satu panduan. Setiap langkah pada panduan memiliki nomor serta dipisahkan oleh baris. Buatkan dalam format json dengan title, ringkasan singkat dan sub-judul terpisah. Serta Pastikan nama key durasi sama dengan `duration`, jumlah sub-judul sama dengan `lessons`, jenis kegiatan sama dengan `type_activity` , tipe kegiatan sama dengan `theme_activity` ,judul sama dengan title, ringkasan singkat sama dengan `desc`, sub-judul sama dengan `subtitles` berbentuk list, judul didalam sub-judul sama dengan `topic`, deskripsi didalam sub-judul bernama `shortdesc` ,panduan sama dengan `content` benbentuk object (dictionary), kalimat pembuka pada panduan dengan nama `opening`, langkah pada panduan dengan nama `step` berbentuk list [str, str, ...], kalimat penutup pada panduan dengan nama `closing`. Pastikan output selalu konsisten dan memiliki format json yang selalu sama, nama key json sama dengan nama yang sudah diberikan."

#     query = openai.ChatCompletion.create(
#     # query = openai.Completion.create(
#         model='gpt-3.5-turbo',
#         messages=[{"role": "system", "content": base_prompt }],
#         stream= True,
#         # prompt=base_prompt,
#         # max_tokens=2049,
#     )
#     return query
#     # for chunk in query:
#         # print(chunk)
    
#     # make except if status code != 200
#     # response = query.get('choices')[0]['message']['content']
#     # response = query.get('choices')[0]['text']
    
#     # return response
