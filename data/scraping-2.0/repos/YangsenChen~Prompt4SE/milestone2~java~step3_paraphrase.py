import openai
import json
from step3_paraphrase import *
import re

# file_number = input("Enter file number: ")

def process_md_data(filename):

    with open(filename, 'r') as file:
        content = file.read()

    parsed_data = re.split("## chatgpt:|## user:|## result", content)[1:-1]

    formatted_data = []
    for i, data in enumerate(parsed_data):
        formatted_data.append({'role': 'user' if i % 2 == 0 else 'assistant', 'content': data.strip()})

    return formatted_data

for i in range(42, 51):
    f = str(i) + '.md'
    file_number = f.split('.')[0]
    # read the JSON data from file
    markdown_name = './chat_history/%s.md' % file_number

    json_file = './dataset/%s.json' % file_number

    conversation_history = process_md_data(markdown_name)

    with open('dataset/' + file_number + '.json', 'r') as file:
        data = json.load(file)
        original_string = data['original_string']
        func_name = data['func_name'].split('.')[-1]

    # conversation_history=[{'role': 'user', 'content': '```java\npublic final void setErrorConverter(Converter<Map<String, String>, OAuth2Error> errorConverter) {\n\t\tAssert.notNull(errorConverter, "errorConverter cannot be null");\n\t\tthis.errorConverter = errorConverter;\n\t}\n```\n\nWhere is this function from?'}, {'role': 'assistant', 'content': 'This function is written in Java and is a method of a class. The method `setErrorConverter` takes an argument of type `Converter<Map<String, String>, OAuth2Error>` and returns no value (`void`). \n\nWithout context, it is difficult to determine which class this method belongs to. However, based on the method signature, it seems to be related to OAuth2 authentication and error handling. The `Assert.notNull(errorConverter, "errorConverter cannot be null")` statement checks if the `errorConverter` argument is not null and throws an exception if it is.'}, {'role': 'user', 'content': '```java\npublic final void setErrorConverter(Converter<Map<String, String>, OAuth2Error> errorConverter) {\n\t\tAssert.notNull(errorConverter, "errorConverter cannot be null");\n\t\tthis.errorConverter = errorConverter;\n\t}\n```\n\n1. reason about code execution of this code\n\n2. explain the functionality of this code'}, {'role': 'assistant', 'content': '1. This code defines a method called `setErrorConverter` that takes a `Converter` object as its argument. The method first checks if the argument is not null using `Assert.notNull`, which will throw an exception if the argument is null. If the argument is not null, the method assigns it to the `errorConverter` field of the class that the method belongs to.\n\n2. This code sets an error converter for an OAuth2 client. The `Converter` object is a functional interface that is used to convert a `Map<String, String>` to an `OAuth2Error` object. The `setErrorConverter` method sets the error converter that will be invoked if an error occurs during an OAuth2 authentication flow. The `Assert.notNull` statement is used to ensure that a valid `Converter` is passed as an argument to the method.'}, {'role': 'user', 'content': 'Now I want you to act as a senior programmer.\n The most important thing is to only show me the code.\n\n```java\npublic final void setErrorConverter(Converter<Map<String, String>, OAuth2Error> errorConverter) {\n\t\tAssert.notNull(errorConverter, "errorConverter cannot be null");\n\t\tthis.errorConverter = errorConverter;\n\t}\n```\n\ngenerate a Main class wrapping this code and add public static void main to use this code snippet.\nyou can self implement any class that you think is needed.\nand the most important thing is that do not use any third party library.\nand the most important thing is do not change the content of setErrorConverter.'}, {'role': 'assistant', 'content': 'Sure, here is an example implementation of the `Main` class that wraps the given code and demonstrates its use:\n\n```java\nimport java.util.HashMap;\nimport java.util.Map;\nimport java.util.function.Function;\n\npublic class Main {\n\n    private Converter<Map<String, String>, OAuth2Error> errorConverter;\n\n    public Main() {\n        // Initialize any necessary objects or variables here\n    }\n\n    public final void setErrorConverter(Converter<Map<String, String>, OAuth2Error> errorConverter) {\n        Assert.notNull(errorConverter, "errorConverter cannot be null");\n        this.errorConverter = errorConverter;\n    }\n\n    public static void main(String[] args) {\n\n        Main main = new Main();\n\n        // Create a sample Converter implementation\n        Converter<Map<String, String>, OAuth2Error> converter = map -> {\n            return new OAuth2Error(map.get("error"), map.get("error_description"), map.get("error_uri"));\n        };\n\n        // Set the error converter\n        main.setErrorConverter(converter);        \n\n        // Use the error converter to convert a Map to an OAuth2Error object\n        Map<String, String> errorMap = new HashMap<>();\n        errorMap.put("error", "invalid_request");\n        errorMap.put("error_description", "The request is missing a required parameter");\n        OAuth2Error error = main.errorConverter.convert(errorMap);\n        System.out.println(error);\n        \n    }\n\n}\n\nclass OAuth2Error {\n    private String error;\n\n    private String errorDescription;\n\n    private String errorUri;\n\n    public OAuth2Error(String error, String errorDescription, String errorUri) {\n        this.error = error;\n        this.errorDescription = errorDescription;\n        this.errorUri = errorUri;\n    }\n\n    public String getError() {\n        return error;\n    }\n\n    public String getErrorDescription() {\n        return errorDescription;\n    }\n\n    public String getErrorUri() {\n        return errorUri;\n    }\n\n    @Override\n    public String toString() {\n        return "OAuth2Error{" +\n                "error=\'" + error + \'\\\'\' +\n                ", errorDescription=\'" + errorDescription + \'\\\'\' +\n                ", errorUri=\'" + errorUri + \'\\\'\' +\n                \'}\';\n    }\n}\n\ninterface Converter<F, T> extends Function<F, T> {\n}\n```\n\nIn this implementation, we have created a class called `Main` with a constructor, a `setErrorConverter` method that sets the error converter, and a `main` method that creates a sample `Converter` implementation and demonstrates its use by converting a sample `Map` object to an `OAuth2Error` object.\n\nWe have also defined the `OAuth2Error` class, which is used by the `Converter` implementation to create an `OAuth2Error` object from a `Map` object. Finally, we have defined the `Converter` interface, which extends the `Function` interface and is used to define the conversion logic for the `Map` to `OAuth2Error` conversion.'}, {'role': 'user', 'content': 'append three unit test methods inside the Main class.\n each test method should start with @test'}, {'role': 'assistant', 'content': 'Sure, here is an updated implementation of the `Main` class that includes three jUnit test methods:\n\n```java\nimport org.junit.Assert;\nimport org.junit.jupiter.api.Test;\n\nimport java.util.HashMap;\nimport java.util.Map;\nimport java.util.function.Function;\n\npublic class Main {\n\n    private Converter<Map<String, String>, OAuth2Error> errorConverter;\n\n    public Main() {\n        // Initialize any necessary objects or variables here\n    }\n\n    public final void setErrorConverter(Converter<Map<String, String>, OAuth2Error> errorConverter) {\n        Assert.notNull(errorConverter, "errorConverter cannot be null");\n        this.errorConverter = errorConverter;\n    }\n\n    public static void main(String[] args) {\n\n        Main main = new Main();\n\n        // Create a sample Converter implementation\n        Converter<Map<String, String>, OAuth2Error> converter = map -> {\n            return new OAuth2Error(map.get("error"), map.get("error_description"), map.get("error_uri"));\n        };\n\n        // Set the error converter\n        main.setErrorConverter(converter);\n\n        // Use the error converter to convert a Map to an OAuth2Error object\n        Map<String, String> errorMap = new HashMap<>();\n        errorMap.put("error", "invalid_request");\n        errorMap.put("error_description", "The request is missing a required parameter");\n        OAuth2Error error = main.errorConverter.convert(errorMap);\n        System.out.println(error);\n\n    }\n\n    @Test\n    public void testErrorConverterNotNull() {\n        Main main = new Main();\n        try {\n            main.setErrorConverter(null);\n            Assert.fail("Expected an IllegalArgumentException to be thrown");\n        } catch (IllegalArgumentException e) {\n            Assert.assertEquals("errorConverter cannot be null", e.getMessage());\n        }\n    }\n\n    @Test\n    public void testConvertErrorMap() {\n        Main main = new Main();\n\n        // Create a sample Converter implementation\n        Converter<Map<String, String>, OAuth2Error> converter = map -> {\n            return new OAuth2Error(map.get("error"), map.get("error_description"), map.get("error_uri"));\n        };\n\n        // Set the error converter\n        main.setErrorConverter(converter);\n\n        // Use the error converter to convert a Map to an OAuth2Error object\n        Map<String, String> errorMap = new HashMap<>();\n        errorMap.put("error", "invalid_request");\n        errorMap.put("error_description", "The request is missing a required parameter");\n        OAuth2Error error = main.errorConverter.convert(errorMap);\n\n        Assert.assertEquals("invalid_request", error.getError());\n        Assert.assertEquals("The request is missing a required parameter", error.getErrorDescription());\n        Assert.assertNull(error.getErrorUri());\n    }\n\n    @Test\n    public void testConvertErrorMapMissingError() {\n        Main main = new Main();\n\n        // Create a sample Converter implementation\n        Converter<Map<String, String>, OAuth2Error> converter = map -> {\n            return new OAuth2Error(map.get("error"), map.get("error_description"), map.get("error_uri"));\n        };\n\n        // Set the error converter\n        main.setErrorConverter(converter);\n\n        // Use the error converter to convert a Map to an OAuth2Error object\n        Map<String, String> errorMap = new HashMap<>();\n        errorMap.put("error_description", "The request is missing a required parameter");\n        OAuth2Error error = main.errorConverter.convert(errorMap);\n\n        Assert.assertNull(error.getError());\n        Assert.assertEquals("The request is missing a required parameter", error.getErrorDescription());\n        Assert.assertNull(error.getErrorUri());\n    }\n\n}\n\nclass OAuth2Error {\n    private String error;\n    private String errorDescription;\n    private String errorUri;\n\n    public OAuth2Error(String error, String errorDescription, String errorUri) {\n        this.error = error;\n        this.errorDescription = errorDescription;\n        this.errorUri = errorUri;\n    }\n\n    public String getError() {\n        return error;\n    }\n\n    public String getErrorDescription() {\n        return errorDescription;\n    }\n\n    public String getErrorUri() {\n        return errorUri;\n    }\n\n    @Override\n    public String toString() {\n        return "OAuth2Error{" +\n                "error=\'" + error + \'\\\'\' +\n                ", errorDescription=\'" + errorDescription + \'\\\'\' +\n                ", errorUri=\'" + errorUri + \'\\\'\' +\n                \'}\';\n    }\n}\n\ninterface Converter<F, T> extends Function<F, T> {\n}\n```\n\nIn this implementation, we have added three test methods, each starting with the `@Test` annotation. These test methods use jUnit assertions to verify that the `setErrorConverter` method and the `Converter` implementation are working correctly. The first test method checks that the `setErrorConverter` method throws an exception when a `null` converter is passed as an argument. The second and third test methods test the functionality of the `Converter` implementation by checking the conversion of a sample `Map` object to an `OAuth2Error` object, in two different scenarios.'}]

    openai.api_key = "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"


    def send_message_to_chatgpt(message, conversation_history):
        # conversation_history.append({'role': 'system', 'content': 'start the chat'})
        conversation_history.append({'role': 'user', 'content': message})

        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                         {'role': 'system',
                          'content': 'You are ChatGPT, a large language model trained by OpenAI, based on the GPT-4 architecture.'}
                     ] + conversation_history,
            # max_tokens=150,
            # n=1,
            # stop=None,
            # temperature=0.7,
        )

        assistant_message = response.choices[0].message.content
        conversation_history.append({'role': 'assistant', 'content': assistant_message})
        return assistant_message


    response = send_message_to_chatgpt("produce semantically equivalent code for %s" % original_string, [])

    with open('semantic_equivalent/'+file_number+'.md', 'w') as file:
        file.write(response)
        print(f)
        print(response)
        print("\n\n")
