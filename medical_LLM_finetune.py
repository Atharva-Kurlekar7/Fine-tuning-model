# -*- coding: utf-8 -*-
"""Untitled0.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/18GPsrNxEMuC1gopPoLEFYKsCGGhfFUCn
"""

# Install required packages
!pip install -q transformers datasets torch

import os
import torch
import pandas as pd
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq
)
from datasets import Dataset

# 1. DATASET PREPARATION
print("1. Preparing dataset...")

# Create a high-quality medical QA dataset with clear instruction format
medical_data = [
    {
        "instruction": "What are the symptoms of type 2 diabetes?",
        "output": "The main symptoms of type 2 diabetes include increased thirst, frequent urination, excessive hunger, fatigue, blurred vision, slow-healing sores, and recurring infections."
    },
    {
        "instruction": "How is hypertension diagnosed?",
        "output": "Hypertension is diagnosed when blood pressure readings consistently show 130/80 mmHg or higher on multiple separate occasions using a calibrated blood pressure monitor."
    },
    {
        "instruction": "What medications are commonly prescribed for asthma?",
        "output": "Common asthma medications include short-acting bronchodilators like albuterol, inhaled corticosteroids like fluticasone, and combination medications containing both types of drugs."
    },
    {
        "instruction": "What is the difference between Alzheimer's and dementia?",
        "output": "Alzheimer's is a specific disease causing memory loss and cognitive decline, while dementia is a general term for symptoms affecting memory and thinking severe enough to interfere with daily life."
    },
    {
        "instruction": "How is strep throat diagnosed and treated?",
        "output": "Strep throat is diagnosed using a rapid strep test or throat culture, and treated with antibiotics like penicillin or amoxicillin to prevent complications and reduce symptoms."
    },
    {
        "instruction": "What are the warning signs of a stroke?",
        "output": "Warning signs of stroke can be remembered with the acronym FAST: Facial drooping, Arm weakness, Speech difficulties, and Time to call emergency services."
    },
    {
        "instruction": "How effective is the flu vaccine?",
        "output": "The flu vaccine is typically 40-60% effective at preventing infection, and even when it doesn't prevent infection, it often reduces the severity and complications of the illness."
    },
    {
        "instruction": "What are the side effects of statins?",
        "output": "Common side effects of statins include muscle pain, liver enzyme elevation, increased blood sugar, memory problems, and in rare cases, a serious condition called rhabdomyolysis."
    },
    {
        "instruction": "How is rheumatoid arthritis different from osteoarthritis?",
        "output": "Rheumatoid arthritis is an autoimmune disease causing joint inflammation, while osteoarthritis results from wear and tear on joints. RA affects joints symmetrically and causes morning stiffness."
    },
    {
        "instruction": "What is the recommended treatment for mild depression?",
        "output": "Mild depression is often treated with psychotherapy like cognitive behavioral therapy, lifestyle changes including exercise and improved sleep, and sometimes antidepressant medications."
    }
]

# Add 30% more examples (3 more to make 13 total)
additional_examples = [
    {
        "instruction": "What causes migraines?",
        "output": "Migraines are caused by abnormal brain activity affecting nerves and blood vessels, often triggered by stress, hormonal changes, certain foods, bright lights, and changes in sleep patterns."
    },
    {
        "instruction": "How is pneumonia diagnosed?",
        "output": "Pneumonia is diagnosed through physical examination, chest X-rays showing lung infiltrates, blood tests for infection markers, and sometimes sputum cultures to identify the specific pathogen."
    },
    {
        "instruction": "What are the early signs of heart attack?",
        "output": "Early signs of heart attack include chest pain or discomfort, shortness of breath, pain radiating to the arm, jaw or back, nausea, cold sweat, and unusual fatigue, especially in women."
    }
]

# Combine the datasets
medical_data.extend(additional_examples)

# Create a simple dataset - sometimes less is more with fine-tuning
train_dataset = Dataset.from_pandas(pd.DataFrame(medical_data))

print(f"Dataset created with {len(train_dataset)} examples")

# 2. MODEL SELECTION
print("\n2. Selecting model...")
# Use flan-t5-base instead of small for better baseline performance
model_name = "google/flan-t5-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
print(f"Selected model: {model_name}")

# 3. DATA PREPROCESSING - Simplified
print("\n3. Preprocessing data...")

# Max sequence lengths
max_source_length = 128
max_target_length = 128

def preprocess_function(examples):
    # Simply prefix the instruction with "Medical question: "
    sources = ["Medical question: " + i for i in examples["instruction"]]
    targets = examples["output"]

    # Tokenize inputs
    inputs = tokenizer(
        sources,
        max_length=max_source_length,
        padding="max_length",
        truncation=True
    )

    # Tokenize targets
    outputs = tokenizer(
        targets,
        max_length=max_target_length,
        padding="max_length",
        truncation=True
    )

    batch = {
        "input_ids": inputs.input_ids,
        "attention_mask": inputs.attention_mask,
        "labels": outputs.input_ids,
    }

    # Replace padding token id with -100 in labels
    batch["labels"] = [
        [label if label != tokenizer.pad_token_id else -100 for label in labels]
        for labels in batch["labels"]
    ]

    return batch

# Process the dataset
tokenized_dataset = train_dataset.map(preprocess_function, batched=True)

# 4. TRAINING SETUP - Simplified
print("\n4. Setting up training...")

# Data collator
data_collator = DataCollatorForSeq2Seq(
    tokenizer=tokenizer,
    model=model
)

# Training arguments - very simple
training_args = TrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=2,
    learning_rate=1e-4,
    num_train_epochs=10,  # More epochs on small dataset
    weight_decay=0.01,
    save_strategy="epoch",
    save_total_limit=2,
    logging_steps=10,
    fp16=False,  # Disable mixed precision
    report_to="none"  # Disable wandb
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=data_collator,
)

# 5. TRAINING
print("\n5. Starting training...")
trainer.train()

# Save model
model_output_dir = "./medical-qa-finetuned-model"
trainer.save_model(model_output_dir)
tokenizer.save_pretrained(model_output_dir)
print(f"Model and tokenizer saved to {model_output_dir}")

# 6. INFERENCE SETUP
print("\n6. Creating inference pipeline...")

def generate_answer(question):
    input_text = f"Medical question: {question}"
    inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True, max_length=max_source_length)

    # Move to GPU if available
    if torch.cuda.is_available():
        inputs = {k: v.to("cuda") for k, v in inputs.items()}
        model.to("cuda")

    # Generate answer - use the dict correctly
    outputs = model.generate(
        input_ids=inputs["input_ids"],  # Fixed here
        attention_mask=inputs["attention_mask"],  # Fixed here
        max_length=max_target_length,
        do_sample=True,
        temperature=0.3,  # Lower temperature for more focused outputs
        num_beams=4,
        no_repeat_ngram_size=3  # Prevent repetition
    )

    # Decode and return the answer
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Create inference script
inference_script = """
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# Load the fine-tuned model and tokenizer
model_path = "./medical-qa-finetuned-model"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSeq2SeqLM.from_pretrained(model_path)

# Move to GPU if available
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

def generate_answer(question):
    # Prepare input - use consistent formatting
    input_text = f"Medical question: {question}"
    inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True, max_length=128)

    # Move to device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Generate answer with careful parameters - use dict keys correctly
    outputs = model.generate(
        input_ids=inputs["input_ids"],  # Access as dict key
        attention_mask=inputs["attention_mask"],  # Access as dict key
        max_length=128,
        do_sample=True,
        temperature=0.3,  # Lower temperature for more focused outputs
        num_beams=4,
        no_repeat_ngram_size=3  # Prevent repetition
    )

    # Decode and return the answer
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Interactive mode
if __name__ == "__main__":
    print("Medical QA System")
    print("Type 'exit' or 'quit' to end the session")

    while True:
        question = input("\\nMedical question: ")
        if question.lower() in ["exit", "quit"]:
            break

        answer = generate_answer(question)
        print(f"\\nAnswer: {answer}")
"""

with open("medical_qa_inference.py", "w") as f:
    f.write(inference_script)

print("Inference script created as 'medical_qa_inference.py'")

# Test with example questions
print("\nTesting with example questions:")
test_questions = [
    "What are the symptoms of type 2 diabetes?",
    "How is hypertension diagnosed?",
    "What are the side effects of statins?",
    "What is the difference between Alzheimer's and dementia?",
    "How is strep throat diagnosed and treated?",
    "What causes migraines?",           # Testing one of the new examples
    "How is pneumonia diagnosed?",      # Testing another new example
    "What are the early signs of heart attack?"  # Testing the third new example
]

for test_question in test_questions:
    print(f"\nQuestion: {test_question}")
    print(f"Generated Answer: {generate_answer(test_question)}")

print("\nMedical QA fine-tuning project completed successfully!")