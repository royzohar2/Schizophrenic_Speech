import numpy as np
import torch
from transformers import AutoTokenizer
from transformers import Trainer, TrainingArguments, AutoModelForSequenceClassification
from sklearn.decomposition import NMF, PCA
from sklearn.manifold import TSNE

model_name = "onlplab/alephbert-base"  # Use the AlephBERT base model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels = 2)  # Binary classification
device = torch.device('cpu')
model.to(device)


class FeatureFusion:
    def __init__(self, n_components=50):
        self.pca = PCA(n_components = n_components)

    def combine_features(self, sentence_embeddings, additional_features):
        combined_features = np.hstack((sentence_embeddings, additional_features))
        return self.pca.fit_transform(combined_features)


class AlephBERTFineTuner:
    def __init__(self, model_name=model_name):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels = 2)

    def tokenize_function(self, examples):
        return self.tokenizer(examples['text'], padding = "max_length", truncation = True)

    def train_model(self, train_data, test_data):
        train_data = train_data.map(self.tokenize_function, batched = True)
        test_data = test_data.map(self.tokenize_function, batched = True)

        training_args = TrainingArguments(
            output_dir = "./results",
            evaluation_strategy = "epoch",
            learning_rate = 2e-5,
            per_device_train_batch_size = 16,
            per_device_eval_batch_size = 16,
            num_train_epochs = 10,
            load_best_model_at_end = True,
            save_total_limit = 1,
            save_strategy = "epoch",
        )

        trainer = Trainer(
            model = self.model,
            args = training_args,
            train_dataset = train_data,
            eval_dataset = test_data,
        )

        trainer.train()


if __name__ == '__main__':
    train_data = []
    test_data = []
    # Example of combining features
    feature_fusion = FeatureFusion()
    combined_features = feature_fusion.combine_features(sentence_embeddings, additional_features)
    fine_tuner = AlephBERTFineTuner()
    fine_tuner.train_model(train_data, test_data)
