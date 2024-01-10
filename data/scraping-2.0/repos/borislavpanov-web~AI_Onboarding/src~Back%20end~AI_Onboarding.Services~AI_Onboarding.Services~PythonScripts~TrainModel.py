import os
from transformers import T5ForConditionalGeneration, AutoTokenizer, pipeline,AutoModelForSeq2SeqLM, Seq2SeqTrainer, Seq2SeqTrainingArguments, DataCollatorForSeq2Seq
from langchain.text_splitter import RecursiveCharacterTextSplitter
from datasets import Dataset
import pandas as pd
import pymongo

# Connect to MongoDB and retrieve text based on a field
client = pymongo.MongoClient("mongodb://mongo:mongo@localhost:27017/")  

db = client["DocumentsDB"] 
collection = db["Documents"]

# Retrieve the text from MongoDB based on a field
documents = collection.find({"IsForTraining": True}) 
documents = list(documents)

if len(documents) != 0:
    texts = []
    for document in documents:
        texts.append(document["ExtractedText"])

    qmodel_name = "ThomasSimonini/t5-end2end-question-generation"
    amodel_name = "deepset/roberta-base-squad2"
    save_dir = os.path.join(os.path.expanduser("~"), "Desktop", "models", "flan_t5")
    base_model = "google/flan-t5-base"

    qmodel = T5ForConditionalGeneration.from_pretrained(qmodel_name)
    qtokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")


    def run_qmodel(input_string, **generator_args):
        generator_args = {
        "max_length": 256,
        "num_beams": 4,
        "length_penalty": 1.5,
        "no_repeat_ngram_size": 3,
        "early_stopping": True,
        }
        input_string = "generate questions: " + input_string + " </s>"
        input_ids = qtokenizer.encode(input_string, return_tensors="pt")
        res = qmodel.generate(input_ids, **generator_args)
        output = qtokenizer.batch_decode(res, skip_special_tokens=True)
        output = [item.split("? ") for item in output]
        output = output[0]
        return output

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=512,
        chunk_overlap=0
    )

    for text in texts:
        chunks = text_splitter.split_text(text)
        questions = []
        for chunk in chunks:
            questions.extend(run_qmodel(chunk)) 

        answers = []
        question_answerer = pipeline('question-answering', model=amodel_name, tokenizer=amodel_name)

        # Print the generated questions
        for question in questions:
            # Generate an answer based on the question and the input text
            answer = question_answerer(question=question, context=text)
            # Print the generated answer
            answers.append(answer["answer"])

        qa_pairs = []
        # Iterate over the questions and answers arrays to create the question-answer pairs
        for question, answer in zip(questions, answers):
            qa_pair = {
                "question": question,
                "answer": answer
            }
            qa_pairs.append(qa_pair)

        # Set up paths and load the model
        model_path = save_dir if os.path.exists(save_dir) else base_model
        model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
        tokenizer = AutoTokenizer.from_pretrained(base_model)

        df = pd.DataFrame(qa_pairs)
        dataset = Dataset.from_pandas(df)

        def preprocess_function(examples):
            inputs = examples["question"]
            targets = examples["answer"]
        
            model_inputs = tokenizer(inputs, targets, padding="max_length", truncation=True)

            # Setup the tokenizer for targets
            with tokenizer.as_target_tokenizer():
                labels = tokenizer(examples["answer"], max_length=512, truncation=True)

            model_inputs["labels"] = labels["input_ids"]
            return model_inputs

        train_dataset = dataset.map(preprocess_function, batched=True)

        # Step 4: Define the Training Arguments
        training_args = Seq2SeqTrainingArguments(
            output_dir="./output",
            num_train_epochs=5,
            per_device_train_batch_size=8,
            learning_rate=1e-4,
            save_strategy="epoch",
        )

        data_collator = DataCollatorForSeq2Seq(tokenizer,model)

        # Step 5: Create the Trainer
        trainer = Seq2SeqTrainer(
            model,
            training_args,
            train_dataset=train_dataset,
            data_collator=data_collator,
            tokenizer=tokenizer
        )

        # Step 6: Start Training
        trainer.train()

        os.makedirs(save_dir, exist_ok=True)

        model.save_pretrained(save_dir)
        tokenizer.save_pretrained(save_dir)

    collection.update_many({"IsForTraining": True}, {"$set": {"IsForTraining": False}})

# Close the MongoDB connection
client.close()