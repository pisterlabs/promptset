import asyncio
import os
import openai
import tiktoken
import openai_async
import matplotlib.pyplot as plt

with open('secretkey.txt', 'r') as f:
    secret = f.readline()

low=1
high=5
samples=1000

enc = tiktoken.encoding_for_model("gpt-3.5-turbo")
possible_nums=range(low,high+1) # int from 1 to 10
possible_tokens = {f"{enc.encode(f'{i}')[0]}": 100 for i in possible_nums} #coerces model to generate a number from 1 to 10

async def generateRating(metrics):
    try:
        completion = await openai_async.chat_complete(
            secret,
            timeout=60,
            payload={
                "model": "gpt-3.5-turbo",
                "messages": [{"role": "user", "content": f"generate a random number from {str(low)} to {(high)}"}],
                "temperature": 0.7,
                "max_tokens": 1,
                "logit_bias": possible_tokens #won't work for floats or large numbers due to tokenization 

            },
        )
    except:
        return None
    try:
        # print(completion.json())
        res = completion.json()['choices'][0]['message']['content']
        res = res.strip().lower()
        return res
    
    except:
        print(completion.json())
        return None


async def run_concurrent_calls():

    # Semaphore to limit the number of concurrent tasks. Kind of finicky with OpenAI API w/r/t how many concurrent calls you can make without timing out
    semaphore = asyncio.Semaphore(100)

    async def bounded_generateRating(metrics):
        async with semaphore:
            return await generateRating(metrics)

    # List to store the coroutines
    tasks = []

    for _ in range(samples+1):
        task = bounded_generateRating(metrics=[])
        tasks.append(task)

    # Run the coroutines concurrently
    results = await asyncio.gather(*tasks)


    # Print or process the results as needed
    print(len(results))
    results=[int(i) for i in results if i is not None]
    print("mean: ", sum(results)/len(results), "versus expected: ", (high+low)/2)
    # Plot histogram
    # Compute weights for each data point such that the histogram sums up to 1
    #save results to a csv
    with open(f'{str(low)}_{str(high)}_n{len(results)}.csv', 'w') as f:
        for i in results:
            f.write(str(i)+",\n")
    data=results
    
    weights = [1/len(data) for _ in data]

    # Plot histogram with proportions
    plt.hist(data, bins=range(low, high+2), edgecolor="k", align='left', weights=weights) #works well for ints, but not for floats
    expected_proportion = 1/len(range(low, high+1))
    plt.axhline(y=expected_proportion, color='r', linestyle='--', label="Expected Proportion")

    # Set the title and labels
    plt.title(f'Integers from {low} to {high}, n= '+str(len(data)))
    plt.xlabel('Value')
    plt.ylabel('Proportion')
    plt.xticks(list(range(low, high+1)))

    # Display the histogram
    plt.show()


if __name__ == "__main__":
    # Setup asyncio event loop
    loop = asyncio.get_event_loop()

    # Run the concurrent calls function
    loop.run_until_complete(run_concurrent_calls())


