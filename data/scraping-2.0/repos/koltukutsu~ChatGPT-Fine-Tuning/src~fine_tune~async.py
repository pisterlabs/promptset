import asyncio
import os
import pandas as pd
import openai


async def generate_recommendations(
    prompt, sub_prompt, age, gender, activity, temperature, max_token
):
    prompt = prompt.format(age=age, gender=gender, activity=activity)
    sub_prompt = sub_prompt.format(gender=age, activity=gender, age=activity)

    response = await openai.Completion.create(
        engine="text-davinci-002",
        prompt=prompt,
        temperature=temperature,
        max_tokens=max_token,
        n=1,
        stop=None,
        frequency_penalty=0,
        presence_penalty=0,
        user="user_id",
    )

    finish_reason = response.choices[0].finish_reason
    response_txt = response.choices[0].text

    return {
        "age": age,
        "gender": gender,
        "activity": activity,
        "prompt": prompt,
        "sub_prompt": sub_prompt,
        "response_txt": response_txt,
        "finish_reason": finish_reason,
    }


async def main():
    openai.api_key = os.getenv("OPENAI_API_KEY")

    # l_ages = [age for age in range(18, 22)]
    l_ages = [age for age in range(18, 19)]
    l_genders = ["erkeğim", "kadınım"]
    l_activities = [
        "Bir kitap okumak",
        # "Film izlemek",
        # "Müzik dinlemek",
        # "Bir arkadaşla buluşmak",
        # "Yeni bir tarif denemek",
        # "Bahçe işleri yapmak",
        # "Sanat yapmak",
        # "Yeni bir dil öğrenmek",
        # "Gökyüzünü izlemek",
        # "Yeni bir spor denemek",
        # "Fotoğraf çekmek",
        # "Yeni bir hobi edinmek",
        # "Bir oyun oynamak",
        # "Gönüllü çalışmak",
        # "Yeni bir şehir keşfetmek",
        # "Atölye çalışmalarına katılmak",
        # "Güzel bir yemek yapmak",
        # "Bir müze ziyareti yapmak",
        # "Ressamlık yapmak",
    ]
    temperatures = {0: 1, 1: 0.85, 2: 1}
    max_token = 400
    iteration = 1

    prompt = """Ben bir {gender}, yaşım {age}. Boş zamanım var ve boş zamanımı {activity} için kullanmak istiyorum. Bana bu konuyla ilgili 5 öneride bulunup, bu önerilerin açıklamasını yapıp aynı zamanda bana bir maliyet çıkarabilir misin?"""
    sub_prompt = "{gender}, {activity}, {age}"

    tasks = []
    for age in l_ages:
        for gender in l_genders:
            for activity in l_activities:
                for i in range(iteration):  ## iteration times
                    task = asyncio.create_task(
                        generate_recommendations(
                            prompt,
                            sub_prompt,
                            age,
                            gender,
                            activity,
                            temperatures[i],
                            max_token,
                        )
                    )
                    tasks.append(task)

    results = await asyncio.gather(*tasks)
    print(results)
    df = pd.DataFrame(results)
    df.to_excel("./synthetic_dataset.xlsx", index=False)


if __name__ == "__main__":
    asyncio.run(main())
