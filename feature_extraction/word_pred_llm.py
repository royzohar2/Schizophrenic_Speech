import numpy as np
import pandas as pd
import json
from tqdm import tqdm
import torch
from transformers import AutoTokenizer, BertForMaskedLM

model_name = "onlplab/alephbert-base"  # AlephBERT base model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = BertForMaskedLM.from_pretrained(model_name)  # Masked language model for next token prediction
device = torch.device('mps')
model.to(device)


class WordPredictLLM:
    def __init__(self):
        # Load your data
        self.df = pd.read_csv("data/clean_data.csv")
        # self.df = self.df.loc[32:33]

    def get_token_scores(self, text):
        """Get token-wise scores from the model for the given text."""
        # Tokenize the input text
        tokenized_input = tokenizer(text, return_tensors = "pt", truncation = True)
        input_ids = tokenized_input['input_ids'][0].tolist()  # Token IDs

        position_list = []  # To store token-wise scores

        # Loop through each token in the sequence
        for i in range(len(input_ids) - 1):  # Stop one token before the last
            # Prepare input sequence up to the current token
            current_input = input_ids[:i + 1]
            current_tensor = torch.tensor([current_input]).to(device)

            # Get the model's output (logits) for the next token prediction
            with torch.no_grad():
                outputs = model(current_tensor)

            # Extract logits for the last token
            logits = outputs.logits[:, -1, :]  # Get logits for the last predicted token
            probs = torch.softmax(logits, dim = -1)  # Convert logits to probabilities

            # # Get the score for the actual next token
            # next_token = input_ids[i + 1]
            # next_token_score = probs[0, next_token].item()  # Get score for the actual next token
            
            # # Print the current input tokens as text
            # current_tokens = tokenizer.convert_ids_to_tokens(current_input)
            # current_text = tokenizer.convert_tokens_to_string(current_tokens)
            # print("Current input as text:", current_text)

            # # Get the actual next token
            # next_real = input_ids[i + 1]
            # next_real_tokens = tokenizer.convert_ids_to_tokens([next_real])
            # next_real_text = tokenizer.convert_tokens_to_string(next_real_tokens)
            # print("Next actual token as text:", next_real_text)

            # Get the token ID for the actual next token

            next_token_id = input_ids[i + 1]

            # Sort all tokens by their probability
            sorted_token_ids = torch.argsort(probs, dim=-1, descending=True)

            # Find the position of the next token ID in the full sorted list
            position = (sorted_token_ids == next_token_id).nonzero(as_tuple=True)[1].item()

            # Print the position of the next token in the full corpus
            # print(f"Position of the next token in the full corpus: {position}")

            position_list.append(position)

        return position_list

    def process_dataframe(self):
        """Process the DataFrame and compute token-wise scores for each string, then save as JSON with progress bars."""
        results = []  # Store results in JSON format

        # Use tqdm to create a progress bar for rows
        for _, row in tqdm(self.df.iterrows(), total = self.df.shape[0], desc = "Processing rows"):
            result_entry = {
                'file_name': row['file_name'],
                'label': row['label'],
                'questions': {}
            }

            # Use tqdm to create a progress bar for columns
            for column in tqdm(self.df.columns[:-2], desc = f"Processing questions for {row['file_name']}",
                               leave = False):
                try:
                    position_list = self.get_token_scores(row[column])  # Get the vector of scores for the question
                except Exception as e:
                    print(f"Error: {e}")
                    print(f"File: {row['file_name']} | Column: {column}")
                    position_list = []  # Assign an empty list to scores in case of error

                result_entry['questions'][column] = position_list

            # Append the result entry to the results list
            results.append(result_entry)

        # Save results as a JSON file
        with open("data/word_pred_llm.json", 'w', encoding = 'utf-8') as f:
            json.dump(results, f, ensure_ascii = False, indent = 4)

        print("Data saved to data/word_pred_llm.json")


if __name__ == '__main__':
    # Usage example
    word_predict_llm = WordPredictLLM()
    word_predict_llm.process_dataframe()
