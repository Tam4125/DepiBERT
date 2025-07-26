from typing import Optional
from dep_tree_gen import subword_dep_matrix
from transformers import AutoTokenizer
import torch

class DepiBertTokenizer():
    def __init__(self, tokenizer, spacy_processor, alpha=1, seq_len=512):
        self.tokenizer = tokenizer
        self.spacy_processor = spacy_processor
        self.alpha = alpha
        self.seq_len = seq_len

    def __call__(
            self, 
            sentence1, 
            sentence2,
            return_tensors = "pt",
    ):
        encoded = self.tokenizer(
            sentence1,
            sentence2,
            truncation=True,
            padding = "max_length",
            max_length = self.seq_len,
            return_tensors=return_tensors,
        )
        

        dep_matrix = subword_dep_matrix(
            sentence1=sentence1,
            sentence2=sentence2,
            tokenizer=self.tokenizer,
            alpha=self.alpha,
            processor=self.spacy_processor,
            seq_len=self.seq_len,
        )

        encoded['dependency_matrix'] = torch.tensor(dep_matrix, dtype=torch.float32).unsqueeze(0)

        return encoded

    def save_pretrained(self, save_directory):
        self.tokenizer.save_pretrained(save_directory)

    def from_pretrained(cls, pretrained_model_name_or_path, spacy_processor, seq_len=512, alpha=2):
        tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path)
        return cls(tokenizer, spacy_processor, seq_len=seq_len, alpha=alpha)

