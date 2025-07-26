import evaluate
import numpy as np
from torch.utils.data import DataLoader
import torch
from transformers import DataCollatorWithPadding, TrainerCallback
from tqdm import tqdm
from datasets import load_dataset
import os
from modeling_dependency_bert import DependencyBertForSequenceClassification
from collections import defaultdict
from tokenization_depibert import DepiBertTokenizer

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


def save_finetuned_infor(trainer, name):
    os.makedirs("finetuned_infor", exist_ok=True)

    path = f"finetuned_infor/{name}"
    best_eval_metrics = trainer.evaluate()
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"All step: {trainer.state.global_step}" + "\n")
        for k,v in best_eval_metrics.items():
            f.write(f"{k}: {v:.4f}" + "\n")


class DataCollatorWithDependencyMatrix():
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
    
    def __call__(self, features):
        batch = {
            "input_ids": [],
            "attention_mask": [],
            "token_type_ids": [],
            "dependency_matrix": [],
            "labels": []
        }

        for sample in features:
            batch["input_ids"].append(torch.tensor(sample["input_ids"]).squeeze(0))
            batch["attention_mask"].append(torch.tensor(sample["attention_mask"]).squeeze(0))
            batch["token_type_ids"].append(torch.tensor(sample["token_type_ids"]).squeeze(0))
            batch["dependency_matrix"].append(torch.tensor(sample["dependency_matrix"]).squeeze(0))
            batch["labels"].append(torch.tensor(sample["labels"]).squeeze(0))
        
        batch = {
            k:torch.stack(v) for k,v in batch.items()
        }


        return batch



class SaveTopKModelCallback(TrainerCallback):
    def __init__(self, k=3, metric_name="eval_accuracy", mode="max", output_dir="./top_models"):
        self.k = k
        self.metric_name = metric_name
        self.mode = mode
        self.output_dir = output_dir
        self.top_k_models = []  # List of (score, path)

    def on_evaluate(self, args, state, control, metrics, **kwargs):
        score = metrics.get(self.metric_name)
        if score is None:
            return

        # Evaluate whether current checkpoint is in top-k
        improved = False
        if len(self.top_k_models) < self.k:
            improved = True
        elif ((self.mode == "max" and score > self.top_k_models[-1][0]) or
              (self.mode == "min" and score < self.top_k_models[-1][0])):
            improved = True

        if improved:
            # Save model
            save_path = os.path.join(self.output_dir, f"step-{state.global_step}_{score:.4f}")
            kwargs["model"].save_pretrained(save_path)
            torch.save(kwargs["model"].state_dict(), os.path.join(save_path, "pytorch_model.bin"))

            # Track it
            self.top_k_models.append((score, save_path))
            self.top_k_models = sorted(
                self.top_k_models, key=lambda x: x[0], reverse=(self.mode == "max")
            )

            # Optional: delete worst if exceed top-k
            if len(self.top_k_models) > self.k:
                score_rm, path_rm = self.top_k_models.pop()
                if os.path.exists(path_rm):
                    self.delete_directory(path_rm)

    def delete_directory(self, path):
        if not os.path.exists(path):
            print(f"Path {path} does not exist")
            return

        # Delete all files and subdirectories
        for root, dirs, files in os.walk(path, topdown=False):
            for name in files:
                os.remove(os.path.join(root, name))
            for name in dirs:
                os.rmdir(os.path.join(root, name))
        
        os.rmdir(path)


class Tester():
    def __init__(self, collator, batch_size, dataset_name, device, main_path):
        self.collator = collator
        self.batch_size = batch_size
        self.dataset_name = dataset_name
        self.device = device
        self.main_path = main_path
    
    def save_output(self, preds, acc):
        os.makedirs("test_output", exist_ok=True)

        with open(f"test_output/{self.main_path}_preds.txt", "w", encoding="utf-8") as f:
            f.write(f"Accuracy: {acc}\n")
            for pred in preds:
                f.write(f"{str(pred)}\n")

    
    def prediction(self):

        testing_data = load_dataset(f"TTam4105/prepared_{self.dataset_name}_dataset2")['test']
        dataLoader = DataLoader(testing_data, batch_size=self.batch_size, collate_fn=self.collator)

        model_path = None
        f_path = f"{self.dataset_name}_top_models/{self.main_path}"
        max_acc = -10000.0
        for path_to_model in os.listdir(f_path):
            eval_acc = float(path_to_model.split("_")[-1])
            if eval_acc >= max_acc:
                max_acc = eval_acc
                model_path = os.path.join(f_path, path_to_model)
        
        model = DependencyBertForSequenceClassification.from_pretrained(model_path)
        model.to(self.device)

        all_preds = []

        print(f"Testing with {self.dataset_name}...")

        with torch.no_grad():
            for batch in tqdm(dataLoader, desc="Evaluating"):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                token_type_ids = batch["token_type_ids"].to(self.device)
                dependency_matrix = batch["dependency_matrix"].to(self.device)
                
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids,
                    dependency_matrix=dependency_matrix,
                )

                logits = outputs.logits

                preds = None
                num_labels = logits.shape[-1]
                if num_labels == 1:
                    preds = logits.squeeze(-1)
                else:
                    preds = torch.argmax(logits, dim=-1)
                all_preds.extend(preds.cpu().numpy())

        self.save_output(preds=all_preds, acc = max_acc)
    
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

        pred_path = f"test_output/{self.main_path}_preds.txt"
        with open(pred_path, "r", encoding="utf-8") as f1:
            with open(f"glue_benchmark_format/{self.main_path}.tsv", "w", encoding="utf-8") as f2:
                for line in f1:
                    if len(line.strip()) >= 8:
                        continue
                    pred = int(line.strip())
                    if self.dataset_name == "stsb":
                        f2.write(str(line.strip()) + "\n")
                        continue
                    f2.write(label_map[self.dataset_name][pred] + "\n")
        
    def run(self):
        self.prediction()
        self.to_glue_benmarck()