import pandas as pd
from transformers import AutoTokenizer, BertForMaskedLM ,AutoModel ,AutoModelForCausalLM
import torch
from scipy.spatial.distance import cosine
from tqdm import tqdm


MODEL_NAME = "yam-peleg/Hebrew-Mistral-7B" 
MODEL_NAME = "onlplab/alephbert-base"  # Use the AlephBERT base model
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

device = torch.device('cpu')
model.to(device)


class DisorderDetector:
    def __init__(self):
        self.df = pd.read_csv("data/clean_data.csv")
        # self.df = self.df.loc[41:42]

    def detect_disorder(self, text):
        """
        Given a piece of text, the method checks for disordered thinking by prompting the model.
        Returns a score based on the distance between the predicted output and the word 'כן'.
        """
        custom_prompt = (
            "אני נותן לך טקסט\n"
            "אני צריך שתענה בלא או כן\n"
            "האם הטקסט מאופיין בדיבור מבולבל:\n"
        )

        prompt = custom_prompt
        prompt += text + "\n"  # Add the input text with a newline
        prompt += "לא או כן?\n"  # Add a newline before the mask
        prompt += "התשובה שלך היא "
        prompt += "[MASK]"
        inputs = tokenizer(prompt, return_tensors="pt").to(device)


        # Check the dimensions of the tensors
        expected_size = 512  # Adjust based on the model's maximum sequence length
        input_size = inputs.input_ids.size(1)  # Number of tokens in the input

        if input_size > expected_size:
            return None

        # Get predictions for the masked token
        with torch.no_grad():
            outputs = model(**inputs)
            predictions = outputs.logits

        # Get the index of the masked token
        mask_token_index = torch.where(inputs.input_ids == tokenizer.mask_token_id)[1]

        if len(mask_token_index) == 0:
            raise ValueError("No mask token found in the input text.")

        # Use the correct dimensions for indexing predictions
        mask_token_index = mask_token_index.item()  # Convert to scalar if only one mask token is present

        # Extract top token IDs
        top_k = 10
        top_token_ids = torch.topk(predictions[0, mask_token_index, :], top_k).indices

        # Convert token IDs to tokens
        top_tokens = tokenizer.convert_ids_to_tokens(top_token_ids.squeeze().tolist())
        # print("Top tokens:", top_tokens)

        # Check if 'כן' or 'לא' are in the top 10 predictions
        yes_token = 'כן'
        no_token = 'לא'
        yes_token_id = tokenizer.convert_tokens_to_ids(yes_token)
        no_token_id = tokenizer.convert_tokens_to_ids(no_token)

        yes_in_top = yes_token in top_tokens
        no_in_top = no_token in top_tokens

        if yes_in_top and no_in_top:
            # Both 'כן' and 'לא' are in the top 10 predictions
            yes_rank = top_tokens.index(yes_token)
            no_rank = top_tokens.index(no_token)

            # Score based on ranking (lower rank means higher score)
            if yes_rank < no_rank:
                return 1  # 'כן' is ranked higher
            else:
                return 0  # 'לא' is ranked higher
        elif yes_in_top:
            return 1  # Only 'כן' is in the top 10
        elif no_in_top:
            return 0  # Only 'לא' is in the top 10
        else:
            # Neither 'כן' nor 'לא' is in the top 10, compute distances
            yes_token_embedding = model.get_input_embeddings()(torch.tensor([yes_token_id], device=device))
            no_token_embedding = model.get_input_embeddings()(torch.tensor([no_token_id], device=device))
            most_probable_token_id = torch.argmax(predictions[0, mask_token_index, :]).item()
            most_probable_token_embedding = model.get_input_embeddings()(torch.tensor([most_probable_token_id], device=device))

            # Compute distances
            distance_to_yes = cosine(
                most_probable_token_embedding.squeeze().detach().cpu().numpy(), 
                yes_token_embedding.squeeze().detach().cpu().numpy()
            )
            distance_to_no = cosine(
                most_probable_token_embedding.squeeze().detach().cpu().numpy(), 
                no_token_embedding.squeeze().detach().cpu().numpy()
            )

            # Score based on proximity
            if distance_to_yes < distance_to_no:
                return 1  # Closer to 'כן'
            else:
                return 0  # Closer to 'לא'


    def process_dataframe(self):
        """
        Creates a copy of the DataFrame with the text replaced by binary disorder detection output (yes/no).
        The original text columns are replaced with the disorder detection result.
        """
        # Create a new DataFrame to store the disorder detection results
        df_copy = self.df.copy()

        # Iterate through each row and each column (except the last two columns which you want to skip)
        for idx, row in tqdm(df_copy.iterrows(), total = len(df_copy), desc = "Processing dataframe"):
            for column in tqdm(df_copy.columns[0:-2], desc = f"Processing questions for {row['file_name']}",
                               leave = False):

                if not pd.isna(row[column]):
                    # Replace text with binary disorder detection output
                    df_copy.at[idx, column] = self.detect_disorder(row[column])

        return df_copy  # Return the modified DataFrame


# Initialize the disorder detection class
detector = DisorderDetector()
if __name__ == '__main__':
    detector = DisorderDetector()

    new_df = detector.process_dataframe()
    new_df.to_csv("data/processed_disorder_scores.csv", index = False)
