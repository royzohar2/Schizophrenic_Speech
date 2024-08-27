
from pydantic import BaseModel, ConfigDict
from infra.yap_wrapper.yap_api import YapApi
import pandas as pd
import torch
from transformers import BertModel, BertTokenizerFast

# Load BERT model and tokenizer
tokenizer = BertTokenizerFast.from_pretrained('onlplab/alephbert-base')
model = BertModel.from_pretrained('onlplab/alephbert-base')
device = torch.device('cpu')
model.to(device)


class YapFeaturesMetadata(BaseModel):
    tokenized_text: str
    segmented_text: str
    lemmas: str
    dep_tree: pd.DataFrame
    md_lattice: pd.DataFrame
    ma_lattice: pd.DataFrame
    model_config = ConfigDict(arbitrary_types_allowed = True, extra = "allow")

    def parse_to_df(self) -> pd.DataFrame:
        data_combined = pd.concat([self.dep_tree, self.md_lattice], axis = 1)
        data_combined = data_combined.loc[:, ~data_combined.columns.duplicated()]
        data_combined["num"] = data_combined["num"].astype(int)
        data_combined["dependency_arc"] = data_combined["dependency_arc"].astype(int)
        return data_combined


class YapFeatureExtractor:
    def __init__(self):
        self.yap_api_provider = YapApi()

    def get_text_mrl_analysis(self, text: str, **kwargs) -> YapFeaturesMetadata:
        extracted_data = self.yap_api_provider.run(text)
        return YapFeaturesMetadata.model_validate({
            "tokenized_text": extracted_data[0],
            "segmented_text": extracted_data[1],
            "lemmas": extracted_data[2],
            "dep_tree": extracted_data[3],
            "md_lattice": extracted_data[4],
            "ma_lattice": extracted_data[5],
        })
