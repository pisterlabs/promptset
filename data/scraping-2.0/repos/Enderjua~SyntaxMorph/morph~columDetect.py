import openai


code = """
(ns example
  (:gen-class))

(defn factors [n]
  " Find the proper factors of a number "
  (into (sorted-set)
        (mapcat (fn [x] (if (= x 1) [x] [x (/ n x)]))
                (filter #(zero? (rem n %)) (range 1 (inc (Math/sqrt n)))) )))


(def find-pairs (into #{}
               (for [n (range  2 20000)
                  :let [f (factors n)     ; Factors of n
                        M (apply + f)     ; Sum of factors
                        g (factors M)     ; Factors of sum
                        N (apply + g)]    ; Sum of Factors of sum
                  :when (= n N)           ; (sum(proDivs(N)) = M and sum(propDivs(M)) = N
                  :when (not= M N)]       ; N not-equal M
                 (sorted-set n M))))      ; Found pair

;; Output Results
(doseq [q find-pairs]
  (println q))

"""


def columDetect(code):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": f"{code} bu python kodunda hangi etken kullanılmıştır? (etken => fonksiyon, sınıflar, değişkenler) yani hangi etken özelliği ağır olarak var? lütfen sadece 1 tane etken say."}
        ],
        temperature=0.9,
        max_tokens=150,
        top_p=1,
        frequency_penalty=0.0,
        presence_penalty=0.6,
        stop=["Human:", "AI:"]
    )

    response = response['choices'][0]['message']['content']

    etkenler = ["Değişkenler", "Sınıflar", "Fonksiyonlar", 'Fonksiyon', 'fonksiyon', 'fonksiyonlar', "Döngüler", "Koşullu ifadeler", "Kütüphaneler", "Veri yapıları", "Algoritmalar", "Modüller", "İşleçler", "variables", "classes", "functions", "loops", "conditional statements", "libraries", "data structures", "algorithms", "modules", "operators", "değişkenler", "sınıflar", "fonksiyonlar", "döngüler", "koşullu ifadeler", "kütüphaneler", "veri yapıları", "algoritmalar", "modüller", "işleçler"]

    detected_etken = "Bilinmeyen"

    for etken in etkenler:
        if etken in response:
            detected_etken = etken
            break

    return detected_etken


columDetect(code)