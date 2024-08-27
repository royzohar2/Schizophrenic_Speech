import pandas as pd
from transformers import AutoTokenizer, BertForMaskedLM
import torch
from tqdm import tqdm

model_name="onlplab/alephbert-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = BertForMaskedLM.from_pretrained(model_name)
device = torch.device('cpu')
model.to(device)

class DisorderDetector:
    def __init__(self):
        self.df = pd.read_csv("Data/clean_data.csv")
        # self.df = self.df.loc[41:42]


    def detect_disorder(self, text):
        """
        Given a piece of text, the method checks for disordered thinking by prompting the model.
        Returns a score based on the distance between the predicted output and the word 'כן'.
        """
        custom_prompt = "האם הטקסט הזה מראה בלבול ? "
        prompt = f"{text} {custom_prompt} "
        prompt += "[MASK]"
        inputs = tokenizer(prompt, return_tensors="pt")
        
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
        mask_token_logits = predictions[0, mask_token_index, :].squeeze()

        # Get the token ID for 'כן' (yes)
        yes_token_id = tokenizer.convert_tokens_to_ids('כן')

        # Compute the score for 'כן' based on its logit value
        yes_score = mask_token_logits[yes_token_id].item()

        return yes_score
        

    def process_dataframe(self):
        """
        Creates a copy of the DataFrame with the text replaced by binary disorder detection output (yes/no).
        The original text columns are replaced with the disorder detection result.
        """
        # Create a new DataFrame to store the disorder detection results
        df_copy = self.df.copy()

        # Iterate through each row and each column (except the last two columns which you want to skip)
        for idx, row in tqdm(df_copy.iterrows(), total=len(df_copy), desc="Processing dataframe"):
            for column in tqdm(df_copy.columns[0:-2], desc=f"Processing questions for {row['file_name']}", leave=False):

                if not pd.isna(row[column]):
                    # Replace text with binary disorder detection output
                    df_copy.at[idx, column] = self.detect_disorder(row[column])
        
        return df_copy  # Return the modified DataFrame



# Initialize the disorder detection class
detector = DisorderDetector()
if __name__ == '__main__':

    detector = DisorderDetector()

    new_df = detector.process_dataframe()
    new_df.to_csv("Data/processed_disorder_scores.csv", index=False)
