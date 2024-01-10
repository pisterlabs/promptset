import os
import glob
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from langchain.document_loaders import PyPDFLoader
import PyPDF2

# Function to extract text from a PDF file
def extract_text_from_pdf(file):
    # Open the PDF file
    with open(file, 'rb') as f:
        # Create a PDF reader object
        pdf_reader = PyPDF2.PdfReader(f)
        # Initialize an empty string to hold the text content
        text = ''
        # Loop over all the pages in the PDF file
        for page_num in range(len(pdf_reader.pages)):
            # Get the current page object
            page = pdf_reader.pages[page_num]
            # Extract the text from the page
            page_text = page.extract_text()
            # Add the page text to the overall text content
            text += page_text
        # Return the final text content
        return text

# Function to load data from a directory of PDF files
def load_data(data_dir: str) -> torch.Tensor:
    # Find all PDF files in the specified directory
    pdf_files = glob.glob(os.path.join(data_dir, "*.pdf"))
    print(f"Found {len(pdf_files)} PDF files in directory {data_dir}")

    # Extract text from each PDF file and concatenate into a single string
    documents = []
    for i, pdf_file in enumerate(pdf_files):
        print(f"Processing PDF file {i+1}/{len(pdf_files)}: {pdf_file}")
        pdf_text = extract_text_from_pdf(pdf_file)
        documents.append(pdf_text)

    # Tokenize the concatenated text using the GPT-Neo tokenizer
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-2.7B", pad_token="[PAD]")
    inputs = tokenizer('\n\n'.join(documents), return_tensors='pt', padding=True, truncation=True)
    print(f"Tokenized {len(documents)} documents into {inputs['input_ids'].shape[0]} input sequences")

    # Return the input IDs as a PyTorch tensor
    return inputs["input_ids"]



# Function to fine-tune the GPT-Neo model on a dataset of PDFs
def fine_tune_model(train_dataset: torch.Tensor) -> None:
    # Load the GPT-Neo model
    model = AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-neo-2.7B")

    # Define the training arguments
    training_args = TrainingArguments(
        output_dir='./results',                      # Directory to save the fine-tuned model
        num_train_epochs=5,                          # Number of training epochs
        per_device_train_batch_size=8,               # Batch size per GPU during training
        per_device_eval_batch_size=8,                # Batch size per GPU during evaluation
        logging_steps=5000,                          # Log every n steps
        save_steps=10000,                            # Save checkpoint every n steps
        evaluation_strategy="epoch",                 # Evaluate every epoch
        debug=True                                   # Enable debug output
    )

    # Define the trainer object with the model, training arguments, and train dataset
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
    )

    # Train the model
    print("Starting model training...")
    trainer.train()
    print("Finished model training")

    # Save the fine-tuned model
    print("Saving fine-tuned model...")
    model.save_pretrained("./fine_tuned_model")
    print("Finished saving fine-tuned model")


# Load the data from the specified directory
print("Loading data...")
train_dataset = load_data("./data/v1")

# Fine-tune the model on the data
print("Fine-tuning model...")
fine_tune_model(train_dataset)
