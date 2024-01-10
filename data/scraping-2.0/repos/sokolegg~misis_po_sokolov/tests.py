from main import openai_request
def test_openai_1():
    equation = "10 + 10"
    result = openai_request(equation)
    assert result == "20", f"Wrong result: {result}"

    print(result)

def test_openai_2():
    equation = "7 умножить на 9"
    result = openai_request(equation)
    assert result == "63", f"Wrong result: {result}"

    print(result)


if __name__ == "__main__":
    test_openai_1()
    test_openai_2()
