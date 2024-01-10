import re
import time
import os
import openai
from run_test_case import run_pytest
import tiktoken

rerunList = []

envDir = ""
openai.api_key = ""

def get_reponse_from_openai(prompt):
    response = openai.Completion.create(
        model="text-davinci-003",
        prompt=prompt,
        temperature=0.3,
        max_tokens=300,
        top_p=1.0,
        frequency_penalty=0.0,
        presence_penalty=0.0,
        # stop=["\n"]
    )
    print('response: ', response)
    return response['choices'][0]['text']

def parse_response(response, originalAsserts):
    generatedAsserts = []
    if '#Generated assertion:' not in response:
        generatedAssertsContents = response.split('#Generated assertions:\n')[-1]
    else:
        generatedAssertsContents = response.split('#Generated assertion:\n')[-1]
    for line in generatedAssertsContents.split('\n'):
        if 'self.assert' in line:
            tempLine = line
            generatedAsserts.append(tempLine.strip())

    testAsserts = []
    for line in originalAsserts.split('\n'):

        tempLine = line
        tempAsserts = tempLine.strip().split('self.assert')
        for tempAssert in tempAsserts:
            if tempAssert != '':
                testAsserts.append('self.assert' + tempAssert.strip())
    return generatedAsserts, testAsserts


def num_tokens_from_string(string: str, encoding_name: str) -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens
def request_chat(greetingPrompt, questionPrompt, testCode, testCaseName, dirName, className, originalAsserts):

    initial_prompt = greetingPrompt + '\n\n' + 'Acknowledge, I am ready to provide instructions for the assertions. Please provide the method and the unfinished test case.\n\n' + questionPrompt + '\n'
    time.sleep(3)

    tempCount = num_tokens_from_string(initial_prompt, 'gpt2')

    response_withAnswer = get_reponse_from_openai(initial_prompt)
    initial_prompt += '\n' + response_withAnswer
    # check the accuracy of the response
    generatedAsserts, testAsserts = parse_response(response_withAnswer, originalAsserts)
    newGeneratedAsserts = []
    alter_prompt = initial_prompt
    cannotRun = False
    for index in range(len(generatedAsserts)):
        tempAssert = generatedAsserts[index]
        asssertCount = index+1
        alteredCode = []
        for line in testCode.split('\n'):
            if f'"<AssertPlaceholder{str(asssertCount)}>"' in line:
                line = line.replace(f'"<AssertPlaceholder{str(asssertCount)}>"', tempAssert)
                alteredCode.append(line)
            else:
                if '<AssertPlaceholder' in line:
                    continue
                alteredCode.append(line)
        alteredCode = '\n'.join(alteredCode)

        #run the test case
        print(dirName, testCaseName, envDir, alteredCode)
        result = run_pytest(dirName, testCaseName, envDir, alteredCode, className, False)
        if result is None or len(result) == 0 or result == 'ERROR':
            result = run_pytest(dirName, testCaseName, envDir, alteredCode, className, True)
        print('error: ', result)
        if result == 'ERROR':
            cannotRun = True
            print('cannot run the test case')
            break
        if result is None or result == '':
            print('no error')
            continue
        assertPrompt= f'You made a mistake on AssertPlaceholder{str(asssertCount)}, when I run \'{generatedAsserts[index]}\', I received the following error: {result}\nCan you generate a new statement for AssertPlaceholder{str(asssertCount)}\nThe assertion is:'

        tempResult = ""
        tempCount = num_tokens_from_string(assertPrompt, 'gpt2')
        tempCount += num_tokens_from_string(alter_prompt, 'gpt2')
        if tempCount > 4048:
            newAssertPrompt = f'You made a mistake on AssertPlaceholder{str(asssertCount)}, when I run \'{generatedAsserts[index]}\',I received an error. \nCan you generate a new statement for AssertPlaceholder{str(asssertCount)}\nThe assertion is:'
            tempCount = tempCount + num_tokens_from_string(newAssertPrompt, 'gpt2') - num_tokens_from_string(
                assertPrompt, 'gpt2')
            if tempCount > 4048:
                print('token exceeded limit')
                continue
            print('fixed')
            assertPrompt = newAssertPrompt
        alter_prompt += '\n' + assertPrompt
        tempResult = get_reponse_from_openai(alter_prompt)
        alter_prompt = initial_prompt
        for line in tempResult.split('\n'):
            if line.strip().startswith('self.assert'):
                newGeneratedAsserts[index] = line.strip()
                break

        time.sleep(3)

    if cannotRun:
        return response_withAnswer, None
    return response_withAnswer, newGeneratedAsserts

def extract_string(a):
    start_index = a.index('(') + 1
    end_index = a.rindex(')')
    return a[start_index:end_index]


def main():
    # read the existing files
    tarPath = ''
    result_list = []
    fileByDir = {}
    for root, dirs, files in os.walk(tarPath):
        for file in files:
            if file.endswith('_result.txt'):
                result_list.append(file)
            elif file.endswith('_prompt.txt'):
                continue
            elif file.endswith('_greeting.txt') or file.endswith('_result_second.txt'):
                continue
            else:
                dirName =  root.replace(tarPath+'/', '')
                if dirName not in fileByDir:
                    fileByDir[dirName] = []
                fileByDir[dirName].append(file)


    # read through the pair_result
    idCount = 0
    for root, dirs, files in os.walk(tarPath):
        for file in files:
            if not file.endswith('.txt'):
                continue
            if file.endswith('_result.txt') or file.endswith('_greeting.txt') or file.endswith('_prompt.txt') or file.endswith('_result_second.txt'):
                continue

            if file.replace('.txt', '_greeting.txt') not in files:
                continue
            has_result = False
            for result_file in result_list:
                if file.replace('.txt', '_result.txt') in result_file:
                    print('has skipped')
                    has_result = True
                    break
            if has_result:
                continue
            with open(os.path.join(root, file), 'r') as f:
                originalFile = f.read()
            print(os.path.join(root, file))
            [focalMethod, testCode, supportMethod] = originalFile.split('\n----------\n')
            focalMethodName = ''
            for line in focalMethod.split('\n'):
                if 'def ' in line:
                    break
            # skip the file is there is no assert
            if 'self.assert' not in testCode:
                continue
            # skip the file is all lines are asserts
            if len(testCode.split('\n')) - len([line for line in testCode.split('\n') if 'self.assert' in line or 'def ' in line]) <= 2:
                print('skip')
                continue
            # find the class name
            className = ''
            for tempLine in supportMethod.split('\n'):
                if 'Test Class Name: ' in tempLine:
                    className = tempLine.replace('Test Class Name: ', '').strip()
                    break

            if os.path.exists(os.path.join(root, file.replace('.txt', '_greeting.txt'))):
                with open(os.path.join(root, file.replace('.txt', '_greeting.txt')), 'r') as f:
                    greetingPrompt = f.read()
                # count the number of <AssertPlaceholder> in greetingPrompt
            else:
                continue

            # prepare for the second prompt
            newLines = []
            testCaseName = ''
            originalAssert = ''
            assertCount = 1
            for line in testCode.split('\n'):
                if 'def ' in line and len(testCaseName) == 0:
                    testCaseName = line.split(' ')[1].split('(')[0]
                if 'self.assert' in line:
                    originalAssert += line.replace('\n', ' ')
                    newLine = re.sub(r'self.assert.*', f'"<AssertPlaceholder{assertCount}>"', line)
                    newLines.append(newLine.replace('\n', ''))
                    assertCount += 1
                else:
                    newLines.append(line.replace('\n', ''))
            content = '\n'.join(newLines)
            testCode = content
            tempPrompt = f'#Suggest assert sentences for the following unit test case {testCaseName}:\n\n#Method to be tested:\n{focalMethod}#Unit test:\n{content}\n\n#Generate assertion to replace AssertPlaceholder.\nNOTE:"Please provide EXACTLY one assert statement for each AssertPlaceholder in the unit test case. Start each assertion with self.assert\nLet\'s think the answer step by step:\n'
            dirName =  root.replace(tarPath+'/', '')
            response, secondRoundAsserts = request_chat(greetingPrompt, tempPrompt, testCode, testCaseName, dirName, className, ''.join(originalAssert))
            idCount += 1
            # print("response: ", response)
            with open(os.path.join(root, file.replace('.txt', '_prompt.txt')), 'w') as f:
                f.write(tempPrompt)
            resultPath = os.path.join(root, file.replace('.txt', '_result.txt'))
            with open(resultPath, 'w') as f:
                f.write(f'#Method to be tested:\n{focalMethod}#Unit test:\n{testCode}\n\n#Generated assertions:\n{response}\n\n')
                f.write('\n----------\n')
                f.write(''.join(originalAssert))
            if secondRoundAsserts is not None and len(secondRoundAsserts) > 0:
                # print(secondRoundAsserts)
                secondAsserts = '\n'.join(secondRoundAsserts)
                with open(resultPath.replace('_result.txt', '_result_second.txt'), 'w') as f:
                    f.write(
                    f'#Method to be tested:\n{focalMethod}#Unit test:\n{testCode}\n\n#Generated assertions:\n{secondAsserts}\n\n')
                    f.write('\n----------\n')
                    f.write(''.join(originalAssert))


            print('Finished: ', os.path.join(root, file))
            time.sleep(3)

if __name__ == '__main__':

    main()