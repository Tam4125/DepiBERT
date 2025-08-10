# DepiBERT: Incorporating Dependency Structures into Pre-trained Language Models for Semantic Textual Similarity
DepiBERT is a novel BERT-based architecture that comprehensively integrates syntactic dependency structures into pre-trained language models to improve performance on Semantic Textual Similarity (STS) and related NLP tasks.

## 🔍 Motivation
While models like BERT excel at capturing semantic relationships via self-attention, they often overlook grammatical and syntactic structures such as subject–object roles. Dependency syntax explicitly encodes these relationships, enabling models to better understand sentence meaning — especially in tasks where subtle structural changes alter semantics.

## 🏗 Architecture
DepiBERT extends BERT-base by:
* Building intra-sentence dependency matrices (grammatical relations within a sentence) and inter-sentence structural similarity matrices (syntactic alignment between sentence pairs).
* Introducing a Dependency Attention Layer between the embedding layer and first encoder layer.
* Combining standard self-attention (semantic) and dependency-based attention (syntactic) via a learnable mixing mechanism.

This design is lightweight, modular, and interpretable, avoiding full architecture rewrites.
## 📊 Datasets & Evaluation
Evaluated on six GLUE benchmark datasets:
* Sentence Similarity: MRPC, QQP, STS-B
* Sentence Inference: MNLI, QNLI, RTE
  
Result:
* State-of-the-art in all sentence similarity tasks.
* Competitive in inference tasks.
## 🔑 Key Contributions
1.	Comprehensive dependency integration: Combines both intra-sentence and inter-sentence dependency knowledge.
2.	Novel Dependency Attention Layer with dynamic mixing of semantic and syntactic signals.
3.	Extensive evaluation on diverse GLUE tasks, with SOTA performance in semantic similarity.
## 📂 Implementation
* Built with PyTorch + Hugging Face Transformers.
* SpaCy for dependency parsing.
* Supports fine-tuning on custom datasets.
## ⚠ Limitations & Future Work
* Slight instability due to dependency parsing noise.
* Tested only on GLUE; broader evaluation needed.
* Currently limited to BERT-base; larger models (BERT-large, DeBERTa, T5) could yield further gains.
