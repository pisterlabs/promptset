import os

from openai import OpenAI
client = OpenAI()

def prompt_openai_general(full_prompt, cache, temperature, n, model, max_tokens, stop) -> tuple[str, list[str]]:
    cache_key = f"{full_prompt}_{model}_{str(temperature)}" 
    if cache_key not in cache or (cache_key in cache and n > len(cache[cache_key])):
        cache_result = []
        if cache_key in cache:
            n -= len(cache[cache_key])
            cache_result = cache[cache_key]
        system_prompt = "You are an expert at mathematical problem solving." 
        result = call_openai_api(system_prompt, full_prompt, temperature, n=n, model=model, max_tokens=max_tokens, stop=stop)
        cache[cache_key] = cache_result + result
    else:
        result = cache[cache_key]
        pass
    return (cache_key, cache[cache_key])

def call_openai_api(system_prompt, prompt, temperature, n, model, max_tokens, stop) -> list[str]:
    print("not cached")
    prompt = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt}
    ]
    while True:
        try:
            result = client.chat.completions.create(
                model=model,
                messages=prompt,
                temperature=temperature,
                n=n,
                max_tokens=max_tokens,
                stop=stop
            )
            break
        except:
            import time; time.sleep(10); pass
    return [result.choices[i].message.content for i in range(n)]