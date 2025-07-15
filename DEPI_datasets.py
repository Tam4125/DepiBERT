from datasets import load_dataset, DatasetDict, Dataset
from transformers import DataCollatorWithPadding
from dep_tree_gen import subword_dep_matrix
import torch
import spacy
import os

# Glue Datasets: MRPC, QQP, STS-B, MNLI, RTE

class GLUE_datasets():
    def __init__(self, tokenizer, alpha=1, seq_length=128):
        self.tokenizer = tokenizer
        self.seq_length = seq_length
        self.alpha = alpha
        self.processor = spacy.load("en_core_web_sm")
    
    def create_datasets(self, name):

        field_map = {
        "mrpc": ("sentence1", "sentence2"),
        "qqp": ("question1", "question2"),
        "stsb": ("sentence1", "sentence2"),
        "mnli": ("premise", "hypothesis"),
        "rte": ("sentence1", "sentence2"),
        }

        # using cache_dir if you wanna specific the target folder for installing glue datasets
        raw_data = load_dataset("glue", name, cache_dir="D:\huggingface\datasets")
        # if name == "mnli":
        #     new_data = {}
        #     new_data['train'] = raw_data['train']
        #     new_data['validation'] = raw_data['validation_matched']
        #     new_data['test'] = raw_data['test_matched']
        #     raw_data = new_data
        
        field1, field2 = field_map[name]

        def preprocess(sample):
            label = sample['label']
            s1 = sample[field1]
            s2 = sample[field2]
            encoded = self.tokenizer(
                s1, s2,
                truncation=True,
                padding="max_length",
                max_length=self.seq_length,
            )

            dep_matrix = subword_dep_matrix(s1, s2, processor=self.processor, tokenizer=self.tokenizer, alpha=self.alpha, seq_len=self.seq_length)
            
            return {
                'input_ids': encoded['input_ids'],
                'attention_mask': encoded['attention_mask'],
                'token_type_ids': encoded['token_type_ids'],
                'label': label,
                'dependency_matrix': dep_matrix
            }

        processed_data = {}
        for split in list(raw_data.keys()):
            print(f"Processing {name} [{split}] ...")
            max_proc = os.cpu_count()//4
            check=False
            while not check:
                try:
                    processed_data[split] = raw_data[split].map(
                        preprocess,
                        desc=f"Encoding {split}",
                        remove_columns=raw_data[split].column_names,
                        num_proc=max_proc,
                    )
                    check=True
                except RuntimeError:
                    print(f"False with max proc = {max_proc}")
                    max_proc -= 1
                    continue
        dataset_dict = DatasetDict(processed_data)
        
        return dataset_dict

class DataCollatorWithDependencyMatrix():
    def __init__(self, tokenizer, seq_length):
        self.default_collator = DataCollatorWithPadding(tokenizer, max_length=seq_length, padding="max_length")
    
    def __call__(self, datasets):
        features = [datasets[i] for i in range(len(datasets))]
        # Stack everything else using built-in collator
        batch = self.default_collator(features)

        # Stack precomputed dependency matrices
        dep_matrices = [torch.tensor(f['dependency_matrix'], dtype=torch.float32) for f in features]
        batch['dependency_matrix'] = torch.stack(dep_matrices)

        return batch


