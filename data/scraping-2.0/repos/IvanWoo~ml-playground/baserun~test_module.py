import openai


def test_paris_trip():
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        temperature=0.7,
        messages=[
            {"role": "user", "content": "What are three activities to do in Paris?"}
        ],
    )

    assert "Eiffel Tower" in response["choices"][0]["message"]["content"]
