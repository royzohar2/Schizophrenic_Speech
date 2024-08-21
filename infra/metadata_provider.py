import abc
from plistlib import Dict

import pandas as pd
from pydantic import BaseModel


class FeatureStudy(abc.ABC, BaseModel):
    pass


class FeatureMetadata(BaseModel):
    id: str
    tag: int
    features: Dict[str, FeatureStudy]


class MetadataProvider:
    def __init__(self, data: pd.DataFrame = None):
        self.data = data if data else pd.read_csv("../data.csv", index_col = False)

    def get_by_id(self, id: str):
        pass

    def transform_to_df(self):
        pass

    def transform_df_to_feature_obj(self):
        for _, row in self.data.iterrows():
            pass


