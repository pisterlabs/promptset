import datasets
from datetime import datetime
from memas.interface.encoder import TextEncoder
from memas.encoder import openai_ada_encoder, universal_sentence_encoder
from memas.text_parsing import text_parsers


def prep_dataset():
    wikipedia = datasets.load_dataset("wikipedia", "20220301.en")
    test_sentences = []
    i = 0
    
    start = datetime.now()
    for row in wikipedia["train"]:
        test_sentences.extend(text_parsers.split_doc(row["text"], 1024))
        i += 1
        if i > 10:
            break
    end = datetime.now()
    print(f"Splitting {i} documents into {len(test_sentences)} sentences took {(end - start).total_seconds()}s")

    batch_sentences = {}
    for batch_size in [5, 10, 20, 50, 100]:
        batched_list = [test_sentences[i:i+batch_size] for i in range(0, len(test_sentences), batch_size)]
        # pop the last one since likely not fully populated
        batched_list.pop()
        batch_sentences[batch_size] = batched_list

    return test_sentences, batch_sentences


def benchmark_single(test_sentences: list[str], encoder: TextEncoder):
    start = datetime.now()
    i = 0
    for sentence in test_sentences:
        i += 1
        try:
            encoder.embed(sentence)
        except Exception as err:
            print(err)
            print(f"{i}!", sentence)

    end = datetime.now()
    return (end - start).total_seconds()


def benchmark_batch(batched_list: list[list[str]], encoder: TextEncoder):
    start = datetime.now()
    i = 0
    for batch in batched_list:
        i += 1
        try:
            encoder.embed_multiple(batch)
        except Exception as err:
            print(err)
            print(f"{i}!", batch)
    end = datetime.now()
    return (end - start).total_seconds()


def compare_encoders(encoders: dict[str, TextEncoder]):
    test_sentences, batch_sentences = prep_dataset()
    print(len(test_sentences))
    output = {"single": {}}
    for name, encoder in encoders.items():
        single = benchmark_single(test_sentences, encoder)
        print(f"[{name}] Single: total {single}s, avg {single/len(test_sentences)}s per item")
        output["single"][name] = (single, single/len(test_sentences))

    for batch_size, batched_list in batch_sentences.items():
        output[batch_size] = {}
        for name, encoder in encoders.items():
            batch_time = benchmark_batch(batched_list, encoder)
            output[batch_size][name] = (batch_time, batch_time/len(batched_list))
            print(f"[{name}] {batch_size} batch: total {batch_time}s, avg {batch_time/len(batched_list)}s per item")
    return output


if __name__ == "__main__":
    USE_encoder = universal_sentence_encoder.USETextEncoder()
    USE_encoder.init()
    output = compare_encoders({
        "ada": openai_ada_encoder.ADATextEncoder("PLACE_HOLDER"), 
        "use": USE_encoder
    })
    print(output)
