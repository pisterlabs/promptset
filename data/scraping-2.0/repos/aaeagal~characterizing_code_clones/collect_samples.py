# Collect 20 samples of code from openai give NL
import hashlib
import os
import sys
import json
import random
import openai
import re

openai.api_key = os.environ["OPENAI_API_KEY"]


def get_code(prompt):
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
        {
            "role": "user",
            "content": f'{prompt}\n'
        }
        ],
        temperature=1,
        max_tokens=2048,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )
    return response["choices"][0]["message"]["content"]

def main():
    # prompt for code samples
    prompt ="Write a java function that given an integer n and an array of integers primes, return the nth super ugly number. A super ugly number is a positive integer whose prime factors are in the array primes. The nth super ugly number is guaranteed to fit in a 32-bit signed integer. Constraints: 1 <= n <= 10^5, 1 <= primes.length <= 100, 2 <= primes[i] <= 1000, primes[i] is guaranteed to be a prime number. All the values of primes are unique and sorted in ascending order. Do not provide an explanation. Provide the source code beginning with the comment //CODESTART and ending with the comment //CODEEND"
    prompt_id = "superUglyNumber29"
   # Create a directory to store the samples
    if not os.path.exists("data/gpt4"):
        os.mkdir("data/gpt4")

    # Create a file to store the samples
    with open(f"data/gpt4/{prompt_id}.Java", "w") as f:
        #write the prompt to the file & add a new line
        #f.write(prompt + "\n")

        #counter for the number of unique samples and total samples
        unique_samples = 0
        total_samples = 0

        #set of hashes for the samples
        hashes = set()

        #loop until we have 20 unique samples
        while unique_samples < 20:
            #increment the number of samples
            total_samples += 1

            #get the code
            code = get_code(prompt)

            print("Code before preprocessing: -------------------------------------------------------------------------------\n", code)

            # write a regular expression to remove everything outside of source code
            regex = "//CODESTART(.*?)//CODEEND"

            code = re.findall(regex, code, re.DOTALL)

            #check if code and if there's anything in code outside of " " characters 
            if code and code[0].strip(): 
                code = code[0]
            else:
                print("No code found")
                continue


            print("Code after preprocessing: -------------------------------------------------------------------------------\n", code)
            #hash the code
            code_hash = hashlib.sha256(code.encode()).hexdigest()

            #check if the code is unique
            if code_hash not in hashes:
               print("Unique code")
                #write the code to the file
               f.write(code + "\n")
               unique_samples += 1
               print("Unique samples: ", unique_samples)
               hashes.add(code_hash)
        print(f"Total samples: {total_samples}")
        print(f"Unique samples: {unique_samples}")


if __name__ == "__main__":
    main()