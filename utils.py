import evaluate
import numpy as np
from torch.utils.data import DataLoader
import torch
from tqdm import tqdm
from datasets import load_dataset
import os

def compute_metrics(eval_pred):
    accuracy = evaluate.load("accuracy")
    peasonr = evaluate.load("pearsonr")
    """
    logits: raw model outputs (before softmax)
    labels: ground-truth labels from the dataset
    """
    logits, labels = eval_pred
    if logits.shape[-1] == 1:   # STS-B regression: predict a float
        preds = np.squeeze(logits)  # shape: (batch,)
        return peasonr.compute(predictions=preds, references=labels)
    else: # Classification: use argmax for predicted class
        preds = logits.argmax(-1)   # -1 -> which axis for compute the maximum along
        return accuracy.compute(predictions=preds, references=labels)
    

class Tester():
    def __init__(self, model, collator, batch_size, dataset_name, device, path):
        self.model = model
        self.collator = collator
        self.batch_size = batch_size
        self.dataset_name = dataset_name
        self.device = device
        self.path = path
    
    def save_output(self, preds, labels):
        os.makedirs("test_output", exist_ok=True)

        with open(f"test_output/{self.path}_preds.txt", "w", encoding="utf-8") as f:
            for pred in preds:
                f.write(f"{str(pred)}\n")

        with open(f"test_output/{self.path}_truth.txt", "w", encoding="utf-8") as f:
            for label in labels:
                f.write(f"{str(label)}\n")

    
    def prediction(self):
        accuracy = evaluate.load("accuracy")

        testing_data = load_dataset(f"TTam4105/prepared_{self.dataset_name}_dataset2")['test']
        dataLoader = DataLoader(testing_data, batch_size=self.batch_size, collate_fn=self.collator)

        all_preds, all_labels = [], []

        print(f"Testing with {self.dataset_name}...")

        with torch.no_grad():
            for batch in tqdm(dataLoader, desc="Evaluating"):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                token_type_ids = batch["token_type_ids"].to(self.device)
                dependency_matrix = batch["dependency_matrix"].to(self.device)
                labels = batch["labels"].to(self.device)

                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids,
                    dependency_matrix=dependency_matrix,
                )

                logits = outputs.logits

                all_labels.extend(labels.cpu().numpy())

                preds = None
                num_labels = logits.shape[-1]
                if num_labels == 1:
                    preds = logits.squeeze(-1)
                else:
                    preds = torch.argmax(logits, dim=-1)
                all_preds.extend(preds.cpu().numpy())
            
            self.save_output(all_preds, all_labels)
            
            if self.dataset_name == "mrpc":

                if num_labels == 1:
                    mse = np.mean((preds - labels)**2)
                    print(f"MSE loss: {mse}")
                    return mse
                else:
                    acc = accuracy.compute(predictions=all_preds, references=all_labels)['accuracy']
                    print(f"Accuracy: {acc}")
                    return acc
            else:
                print("Truth is not publicly available")
    
    def to_glue_benmarck(self):

        label_map = {
            "mnli": {
                0: "entailment",
                1: "neutral",
                2: "contradiction"
            },
            "mrpc": {
                0: "not_equivalent",
                1: "equivalent"
            },
            "qqp": {
                0: "not_duplicate",
                1: "duplicate"
            },
            "rte": {
                0: "not_entailment",
                1: "entailment"
            },
            # STS-B is regression, no label map
        }
        os.makedirs("glue_benchmark_format", exist_ok=True)

        pred_path = f"test_output/{self.path}_preds.txt"
        with open(pred_path, "r", encoding="utf-8") as f1:
            with open(f"glue_benchmark_format/{self.path}.tsv", "w", encoding="utf-8") as f2:
                for line in f1:
                    pred = int(line.strip())
                    if self.dataset_name == "stsb":
                        f2.write(str(line.strip()) + "\n")
                        continue
                    f2.write(label_map[self.dataset_name][pred] + "\n")
        
    def run(self):
        self.prediction()
        self.to_glue_benmarck()


def save_finetuned_infor(trainer, name):
    os.makedirs("finetuned_infor", exist_ok=True)

    path = f"finetuned_infor/{name}"
    best_eval_metrics = trainer.evaluate()
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"All step: {trainer.state.global_step}" + "\n")
        for k,v in best_eval_metrics.items():
            f.write(f"{k}: {v:.4f}" + "\n")

        
                

