import pandas as pd
import torch
from datasets import Dataset
from transformers import AutoTokenizer, DataCollatorWithPadding, AutoModelForSequenceClassification
from transformers import TrainingArguments, Trainer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from datasets import DatasetDict
import gc

from utils import explode_df_to_single_record

# Ensure we use the CPU for training
device = torch.device("cpu")
print("Using CPU for training.")

model_name = "onlplab/alephbert-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Load the model for sequence classification
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels = 2)
model.to(device)


# Tokenize the data
def tokenize_function(example):
    try:
        return tokenizer(example["answer"], truncation = True)
    except Exception as e:
        print(e)


if __name__ == '__main__':
    # Load your data
    df = pd.read_csv('/Users/seanlavi/dev/Schizophrenic_Speech/data/clean_data.csv')
    df = explode_df_to_single_record(df).dropna()

    # Convert answers to a list
    answers_list = df['answer'].tolist()

    # Create Hugging Face Dataset from pandas DataFrame
    dataset = Dataset.from_pandas(df)

    # Map tokenization function to the dataset
    tokenized_datasets = dataset.map(tokenize_function, batched = True)

    # Split dataset into training and testing sets
    train_df, test_df = train_test_split(df, test_size = 0.2, random_state = 42)

    # Convert to Hugging Face Dataset format
    train_dataset = Dataset.from_pandas(train_df)
    test_dataset = Dataset.from_pandas(test_df)

    # Tokenize datasets
    tokenized_train_dataset = train_dataset.map(tokenize_function, batched = True)
    tokenized_test_dataset = test_dataset.map(tokenize_function, batched = True)

    # Prepare dataset dict
    tokenized_datasets = DatasetDict({
        "train": tokenized_train_dataset,
        "test": tokenized_test_dataset
    })

    # Data collator for dynamic padding
    data_collator = DataCollatorWithPadding(tokenizer = tokenizer)

    # Define training arguments without mixed precision
    training_args = TrainingArguments(
        output_dir = "./results",
        eval_strategy = "epoch",  # Update to avoid deprecation warning
        learning_rate = 2e-5,
        per_device_train_batch_size = 16,
        per_device_eval_batch_size = 16,
        num_train_epochs = 3,
        weight_decay = 0.01,
        logging_dir = './logs',
        logging_steps = 10,
        report_to = "none"  # Disable reporting to external services
    )

    # Initialize the Trainer
    trainer = Trainer(
        model = model,
        args = training_args,
        train_dataset = tokenized_datasets["train"],
        eval_dataset = tokenized_datasets["test"],
        tokenizer = tokenizer,
        data_collator = data_collator,
    )

    # Train the model
    trainer.train()

    # Evaluate the model
    trainer.evaluate()

    # Make predictions
    preds_output = trainer.predict(tokenized_datasets["test"])
    predictions = preds_output.predictions.argmax(axis = -1)

    # Print classification report
    print(classification_report(test_dataset['label'], predictions))

    # Save the fine-tuned model and tokenizer
    model.save_pretrained("./fine_tuned_model")
    tokenizer.save_pretrained("./fine_tuned_model")
