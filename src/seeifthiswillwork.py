# %%
import pandas as pd
import json
import os
import re 
# import pymupdf4llm
# import pymupdf
import sqlite3
import numpy as np

# %%
# train = train.to_dict(orient="records")
# validation = validation.to_dict(orient="records")
# test = test.to_dict(orient="records")

# %%
from sentence_transformers import SentenceTransformer
from sentence_transformers.losses import CachedMultipleNegativesRankingLoss



# %%


# %%
# source: https://huggingface.co/blog/train-sentence-transformers by Tom Aarsen

from torch.utils.data import DataLoader
from datasets import load_dataset
from sentence_transformers import (SentenceTransformer, 
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
    SentenceTransformerModelCardData, SimilarityFunction
)
from sentence_transformers.losses import CachedMultipleNegativesRankingLoss
from sentence_transformers import LoggingHandler
from sentence_transformers.evaluation import TripletEvaluator
from sentence_transformers.training_args import BatchSamplers
import wandb

run = wandb.init(project = "finetune-embeddings")

# 1. Load a model to finetune with 2. (Optional) model card data
model = SentenceTransformer(
    "textgain/allnli-GroNLP-bert-base-dutch-cased",
    model_card_data=SentenceTransformerModelCardData(
        language="Dutch",
        license="",
        model_name="textgain/allnli-GroNLP-bert-base-dutch-cased_finetuned_on_Kamervragentest",
    )
)
# 3. Load a dataset to finetune on
# dataset = load_dataset("sentence-transformers/all-nli", "triplet")

data_files = {
    "train": "C:\\Users\\victo\\LearningLion-Finetune\\LearningLion-FineTuneEmbeddings\\datasets\\train.csv",
    "test": "C:\\Users\\victo\\LearningLion-Finetune\\LearningLion-FineTuneEmbeddings\\datasets\\test.csv",
    "eval": "C:\\Users\\victo\\LearningLion-Finetune\\LearningLion-FineTuneEmbeddings\\datasets\\validation.csv"
}
datasets = load_dataset('csv', data_files=data_files)

# %%
print(datasets['train']['anchor'])

# %%
print(datasets.column_names)

# %%
# 6. (Optional) Create an evaluator & evaluate the base model

anchor_eval = datasets['eval']["anchor"]#.to_list()
positive_eval = datasets['eval']["positive"]#.to_list()
negative_eval = datasets['eval']["negative"]#.to_list()
anchor_test = datasets["test"]["anchor"]
positive_test = datasets["test"]["positive"]
negative_test = datasets["test"]["negative"]

# %%
# 6. (Optional) Create an evaluator & evaluate the base model

dev_evaluator = TripletEvaluator(
    anchors=anchor_eval,
    positives=positive_eval,
    negatives=negative_eval,
    main_distance_function =SimilarityFunction.COSINE, 
    name="tripleteval",
    show_progress_bar=True
)
print(model)
dev_evaluator(model)
evaluation_score = dev_evaluator(model)
wandb.log({"loss_before": evaluation_score})

# %%
# 4. Define a loss function
loss = CachedMultipleNegativesRankingLoss(model)

# 5. (Optional) Specify training arguments
args = SentenceTransformerTrainingArguments(
    # Required parameter:
    output_dir="./",
    # Optional training parameters:
    num_train_epochs=4,
    per_device_train_batch_size=512,
    per_device_eval_batch_size=512,
    warmup_ratio=0.1,
    fp16=True,  # Set to False if GPU can't handle FP16
    bf16=False,  # Set to True if GPU supports BF16
    batch_sampler=BatchSamplers.NO_DUPLICATES,  # MultipleNegativesRankingLoss benefits from no duplicates
    # Optional tracking/debugging parameters:
    logging_dir="./logs",  # Directory for logs
    logging_steps=10,      # Log every 10 steps
    save_strategy="steps", # Save the model after each epoch
    eval_strategy="steps", # Evaluate after each epoch,
    report_to='wandb',
    eval_steps=10,
    save_steps=10,
    save_total_limit=2,
    run_name="fiddlingaround",
    load_best_model_at_end=True,
)
print(args)
args =args.set_logging(strategy="steps",report_to=["wandb"])
print(args)

# %%
# adjusted loop for eval during training

# Training with inline WandB logging #source: ChatGPT
def wandb_logging_evaluator(model):
    """
    Evaluate the model and log metrics directly to WandB.
    """

    scores = dev_evaluator(model)  # Compute metrics
    # wandb.log({f"eval_{key}": value for key, value in scores.items()})
    wandb.log(scores.main)
    print(f"Evaluation results: {scores}")


trainer = SentenceTransformerTrainer(
    model=model,
    args=args,
    train_dataset=datasets['train'].select_columns(["anchor", "positive", "negative"]),
    eval_dataset=datasets['eval'].select_columns(["anchor", "positive", "negative"]),
    loss=loss,
    evaluator= dev_evaluator   #wandb.log({'evalscore': evaluation_score[dev_evaluator.primary_metric]})
    #show_progress_bar=True

)
trainer.train()
wandb.finish()
# # # Chatgpt Custom Training Loop
# epochs = 3 #int(args.num_train_epochs)
# for epoch in range(epochs):
#     print(f"Starting Epoch {epoch + 1}/{epochs}")
    
#     # Train for one epoch
#     trainer.train()
    
#     # Evaluate at the end of the epoch
#     eval_metrics =  wandb_logging_evaluator(model, epoch + 1)
#     print(f"Epoch {epoch + 1} Evaluation: {eval_metrics}")
    
#     # Optionally save model after evaluation
#     model.save(f"output_model_epoch_{epoch + 1}")

# # for epoch in range(num_epochs):
# #     print(f"Starting epoch {epoch + 1}/{num_epochs}")
# #     trainer.train()
# #     # Evaluate and log after each epoch
# #     current_step = (epoch + 1) * steps_per_epoch
# #     wandb_logging_evaluator(model, epoch + 1, current_step)


# %%
# #Basic predefined sentencetransformer trainingloop
# # 7. Create a trainer & train
# trainer = SentenceTransformerTrainer(
#     model=model,
#     args=args,
#     train_dataset=datasets['train'].select_columns(["anchor", "positive", "negative"]),
#     eval_dataset=datasets['eval'].select_columns(["anchor", "positive", "negative"]),
#     loss=loss,
#     evaluator= dev_evaluator(model)   #wandb.log({'evalscore': evaluation_score[dev_evaluator.primary_metric]})
#     #show_progress_bar=True

# )
# trainer.train()

# %%

# (Optional) Evaluate the trained model on the test set, after training completes

test_evaluator = TripletEvaluator(
    anchors=anchor_test,
    positives=positive_test,
    negatives=negative_test,
    main_distance_function = SimilarityFunction.COSINE,
    name="aftertrained",
    show_progress_bar=True
)
test_evaluator(model)

# 8. Save the trained model
model.save_pretrained(r"C:\Users\victo\LearningLion-Finetune\LearningLion-FineTuneEmbeddings\models")


# %% [markdown]
# ## Trainingsdata: validation
# 


