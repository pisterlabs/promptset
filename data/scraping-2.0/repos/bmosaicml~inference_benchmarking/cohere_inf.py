import cohere
import time
from dotenv import dotenv_values
config = dotenv_values(".env")

prod_key = config['COHERE']

co = cohere.Client(prod_key)




ITERS = 5
with open('cohere.tsv', 'w') as f:
    f.write(f"model_name\tprompt_len\toutput_len\tcall_latency\tsecs_per_tok\n")
    for num1 in [1, 16, 32, 64, 128, 256]:
        for num2 in [1, 16, 32, 64, 128, 256]:
            prompt = f"Here are the first {num1} even numbers: " + ', '.join([str(2 * i) for i in range(1, num1+1)]) + f"\nGive the first {num2} odd numbers:"
            for engine in ["command-medium-nightly", "command-xlarge-nightly"]:
                avg_latency = 0.0
                avg_latency_per_token = 0.0
                avg_prompt_len = 0.0
                avg_output_len = 0.0
                for i in range(ITERS):
                    print(f"{num1}-{num2}-{engine}")
                    start = time.time()
                    response = co.generate(  
                        model=engine,  
                        prompt = prompt,  
                        max_tokens=num2*3,  
                        temperature=0.25,  
                        stop_sequences=["--"])
                    end = time.time()
                    result = response.generations[0].text
                    prompt_len = co.tokenize(text=prompt).length
                    output_len = co.tokenize(text=result).length

                    avg_latency += end-start
                    avg_latency_per_token += (end-start)/output_len
                    avg_prompt_len += prompt_len
                    avg_output_len += output_len

                f.write(f"{engine}\t{avg_prompt_len/ITERS}\t{avg_output_len/ITERS}\t{avg_latency/ITERS}\t{avg_latency_per_token/ITERS}\n")
                f.flush()