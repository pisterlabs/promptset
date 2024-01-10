from openai import OpenAI

client = OpenAI()

chat_completion = client.chat.completions.create(
    messages=[
        {
            "role": "system",
            "content": "You are a Liquid generation robot. You are very good at generating Liquid-templates and will output"+
            "only valid Liquid code that can be used directly by a Liquid template engine. The Liquid template engine will "+
            "receive a object `data` with the following keys: 'diagnose', 't-stage', 'n-stage', 'm-stage', 'gleason' and 'therapy'.",
        },
        {
            "role": "user",
            "content": """
            Give the following data 

            ```json
            {
                "diagnose": "prostate cancer",
                "iPSA": "5.2 ng/ml",
                "t-stage": "T1",
                "n-stage": "N0",
                "m-stage": "M0",
                "gleason": "3+3"
                "therapy": "radical prostatectomy"
            }
            ```
            
            I want to generate the following text:

            ```
            Beste collega,
            Na vastellen van een verhoogde PSA waarde van 5.2 ng/ml, hebben we een prostaatbiopsie uitgevoerd.
            De resultaten van de prostaatbiopsie zijn als volgt:
            De diagnose is prostaatkanker, stadium T1N0M0, met een Gleason score van 3+3.
            We adviseren een radicale prostatectomie.

            Met vriendelijke groet,
            ```

            Given the following data:

            ```json
            {
                "diagnose": "Geen prostaatkanker",
                "iPSA": "1.2 ng/ml",
                "therapy": "watchful waiting"
            }
            ```

            I want to generate the following text:

            ```
            Beste collega,

            Na vaststellen van een verhoogde PSA waarde van 1.2 ng/ml, hebben we een prostaatbiopsie uitgevoerd.
            De diagnose is geen prostaatkanker.
            We adviseren een watchful waiting beleid.

            Met vriendelijke groet,
            ```
        """
        },
        {
            "role": "system",
            "content": "Generate the liquide template that will output the text above without any extra text. The template should be valid liquid code."
        }
    ],
    model="gpt-4",
)


if __name__ == "__main__":
    print(chat_completion.choices[0].message.content)
    