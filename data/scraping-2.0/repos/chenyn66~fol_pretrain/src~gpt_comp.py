from tqdm.autonotebook import tqdm
import random
import data
import openai
import syllo_gen
openai.api_key = open('key.txt').read().strip()





def test_composition(dmax=5, shuffle_story=False):
    '''
    adj templates, 20 samples, 100 test samples
    '''
    results = {d+1:[] for d in range(1, dmax)}

    all_possilbe = set()
    base_prompt = 'Answer questions about syllogisms, ignoring semantics.\n\n'
    for j in range(100):
        if len(all_possilbe) == 24:
            break
        real = random.choice([True])
        q, v = syllo_gen.get_syllo(1)

        qs = syllo_gen.question_to_string(q)
        if qs in all_possilbe:
            continue
        all_possilbe.add(qs)

        if not real:
            q = syllo_gen.negate_quesion(q)

        v = syllo_gen.random_assign_adjs(v)

        
        q = syllo_gen.question2template(q, v, rand=True, noun=False)

        base_prompt += f'Story: {" ".join(q["story"])} Question: {q["conclusion"]}\n'
        base_prompt += f'Answer: {real}\n\n'


    for d in tqdm(range(1, dmax)):
        prompt = base_prompt
        
        for j in range(10):
            real = random.choice([True, False])
            q, v = syllo_gen.get_syllo(d+1)
            if not real:
                q = syllo_gen.negate_quesion(q)

            v = syllo_gen.random_assign_adjs(v)
            q = syllo_gen.question2template(q, v, rand=True, noun=False)
            story = q["story"]
            if shuffle_story:
                random.shuffle(story)
            prompt += f'Story: {" ".join(story)} Question: {q["conclusion"]}\n'
            prompt += f'Answer: {real}\n\n'
    
        test_results = dict()
        for test_d in range(1, dmax):
            corret = 0

            for j in range(100):

                real = random.choice([True, False])
                q, v = syllo_gen.get_syllo(test_d+1)
                if not real:
                    q = syllo_gen.negate_quesion(q)

                v = syllo_gen.random_assign_adjs(v)
                q = syllo_gen.question2template(q, v, rand=True, noun=False)
                story = q["story"]
                if shuffle_story:
                    random.shuffle(story)
                
                tmp_prompt = prompt + f'Story: {" ".join(q["story"])} Question: {q["conclusion"]}'
                tmp_prompt += "\nAnswer:"

                response = openai.Completion.create(
                model="text-davinci-003",
                prompt=tmp_prompt,
                temperature=0,
                max_tokens=10,
                top_p=1,
                frequency_penalty=0,
                presence_penalty=0,
                stop=["\n"]
                )

                text = response['choices'][0]['text'].strip()
                if text not in ['True', 'False']:
                    print('Error')
                    print(text)
                    continue
                if text == 'True' and real:
                    corret += 1
                elif text == 'False' and not real:
                    corret += 1

            test_results[test_d] = corret/100
            print(f'Test Depth: {test_d+1}, Test Acc: {corret/100:.4f}')
        results[d+1].append(test_results)


            
    return results