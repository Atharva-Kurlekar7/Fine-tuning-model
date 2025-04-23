# Fine-Tuning a Large Language Model for Medical Question Answering

## Abstract

This report documents the implementation of a medical question-answering system developed through fine-tuning of Google's Flan-T5-base model. The system was trained on a small but focused dataset of 13 medical question-answer pairs covering common medical topics. Despite the limited dataset size, the fine-tuned model demonstrates an ability to generate coherent, relevant responses to medical queries. This work illustrates how even modest fine-tuning efforts can adapt large language models to specialized domains, providing a foundation for more sophisticated medical AI systems with additional data and resources.

## 1. Introduction

Large Language Models (LLMs) have demonstrated remarkable general knowledge capabilities but often lack the specialized knowledge required for domain-specific applications such as medical question answering. This project explores how targeted fine-tuning can help adapt a general-purpose LLM to the medical domain, potentially creating a more reliable source of medical information.

Medical information requires precision and accuracy, making it an interesting challenge for language model adaptation. Through fine-tuning on a small set of carefully crafted examples, we aim to improve the model's ability to provide concise, relevant answers to common medical questions.

## 2. Methodology

### 2.1 Dataset Preparation

The dataset consists of 13 medical question-answer pairs, covering a range of common medical topics including:

- Chronic conditions (diabetes, hypertension, Alzheimer's)
- Diagnostic procedures (hypertension diagnosis, strep throat diagnosis)
- Medications and treatments (asthma medications, statins)
- Disease comparisons (Alzheimer's vs. dementia, rheumatoid vs. osteoarthritis)
- Emergency conditions (stroke warning signs, heart attack symptoms)

Each example follows a consistent instruction-output format:

```
instruction: "What are the symptoms of type 2 diabetes?"
output: "The main symptoms of type 2 diabetes include increased thirst, frequent urination, excessive hunger, fatigue, blurred vision, slow-healing sores, and recurring infections."
```

The dataset was intentionally kept small but high-quality, with carefully crafted answers that provide concise, accurate medical information. The initial set of 10 examples was supplemented with 3 additional examples to expand coverage of medical topics.

### 2.2 Model Selection

Google's Flan-T5-base model was selected as the foundation for fine-tuning based on several factors:

1. **Prior Instruction Tuning**: Flan-T5 has already been exposed to instruction-following tasks, making it well-suited for a question-answering application.

2. **Size and Efficiency**: The "base" variant (approximately 250 million parameters) offers a good balance between capability and computational efficiency, allowing for effective fine-tuning without excessive resource requirements.

3. **Architecture**: The encoder-decoder architecture of T5 is well-suited for text generation tasks like question answering.

4. **Performance**: Flan-T5-base has demonstrated strong performance on a range of natural language tasks, suggesting good transfer learning potential.

### 2.3 Fine-Tuning Implementation

The fine-tuning process implemented in our code follows these steps:

**Preprocessing**:
- Each question is prefixed with "Medical question: " to provide consistent context
- Both inputs and outputs are tokenized with a maximum sequence length of 128 tokens
- Padding and truncation are applied for consistent tensor dimensions
- Padding tokens in the labels are replaced with -100 so they're ignored in loss calculation

**Training Configuration**:
- Batch size: 4 (with gradient accumulation steps of 2 for an effective batch size of 8)
- Learning rate: 1e-4
- Weight decay: 0.01
- Training epochs: 10
- FP16 precision: Disabled for stability

The code implements this configuration using the Hugging Face Transformers library's Trainer API, which handles the training loop, optimization, and checkpointing.

## 3. Evaluation

The evaluation in our implementation is primarily qualitative, testing the model on both seen and unseen medical questions to assess response quality.

### 3.1 Testing Methodology

The evaluation approach in our code consists of:

1. **Testing on training examples**: Verifying the model's ability to reproduce learned information.
2. **Testing on variations**: Checking if the model can handle slight rephrasing of trained questions.
3. **Manual review**: Assessing the correctness, relevance, and clarity of generated responses.

The code implements a simple test function that takes a medical question as input and generates a response using the fine-tuned model.

### 3.2 Results

The fine-tuned model shows promising results in generating relevant, concise answers to medical questions. When tested on the training examples, the model produces responses that closely match the expected outputs, indicating successful learning of the provided information.

For example, when asked "What are the symptoms of type 2 diabetes?", the model correctly lists the key symptoms including increased thirst, frequent urination, fatigue, and blurred vision.

The model also demonstrates some ability to handle variations of the training questions, suggesting limited generalization capability despite the small dataset size.

Due to the limited dataset size and domain scope, the model's knowledge is naturally constrained to the topics covered in the training data. Questions outside this scope may receive less accurate or complete responses.

## 4. Inference Pipeline

A key component of our implementation is the inference pipeline, which provides a simple interface for interacting with the fine-tuned model.

### 4.1 Pipeline Implementation

The inference pipeline implemented in our code includes:

```python
def generate_answer(question):
    input_text = f"Medical question: {question}"
    inputs = tokenizer(input_text, return_tensors="pt", padding=True, 
                      truncation=True, max_length=max_source_length)

    # Move to GPU if available
    if torch.cuda.is_available():
        inputs = {k: v.to("cuda") for k, v in inputs.items()}
        model.to("cuda")

    # Generate answer
    outputs = model.generate(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_length=max_target_length,
        do_sample=True,
        temperature=0.3,
        num_beams=4,
        no_repeat_ngram_size=3
    )

    # Decode and return the answer
    return tokenizer.decode(outputs[0], skip_special_tokens=True)
```

The pipeline includes careful configuration of generation parameters to produce high-quality responses:

- Temperature of 0.3 to reduce randomness
- Beam search with 4 beams to explore multiple generation paths
- N-gram repetition prevention to avoid redundant text

The code also includes a standalone script `medical_qa_inference.py` that provides an interactive command-line interface for querying the model.

## 5. Limitations and Future Improvements

Our implementation has several limitations that suggest directions for future work:

### 5.1 Current Limitations

1. **Limited Dataset Size**: With only 13 examples, the model's medical knowledge is extremely limited in scope.

2. **Lack of Formal Evaluation**: The current implementation doesn't include quantitative metrics to formally assess performance.

3. **Single Hyperparameter Configuration**: The code implements only one set of hyperparameters without optimization or comparison.

4. **No Uncertainty Expression**: The model doesn't express uncertainty when asked questions outside its knowledge scope.

### 5.2 Future Improvements

Based on these limitations, several improvements could enhance the system:

1. **Dataset Expansion**: Increasing the dataset to hundreds or thousands of examples would significantly improve coverage and performance.

2. **Formal Evaluation Framework**: Adding quantitative metrics like BLEU, ROUGE, and exact match would enable objective assessment.

3. **Hyperparameter Optimization**: Implementing a systematic search across learning rates, batch sizes, and training durations could improve performance.

4. **Medical Verification**: Establishing a process for medical experts to verify responses would be essential for any practical application.

5. **Uncertainty Quantification**: Adding mechanisms for the model to express confidence levels would improve trustworthiness.

## 6. Conclusion

This project demonstrates a basic implementation of fine-tuning a language model for medical question answering. Despite using a minimal dataset of just 13 examples, the fine-tuned Flan-T5-base model shows an ability to generate relevant, coherent responses to medical questions within its training scope.

The implementation includes a complete pipeline from data preparation through fine-tuning to inference, providing a foundation for more sophisticated medical AI systems. While the current system has significant limitations due to data constraints and the absence of formal evaluation, it illustrates the potential of targeted fine-tuning for domain adaptation of large language models.

The code structure and implementation details conform to best practices in the field, utilizing the Transformers library for efficient implementation of model training and inference. With the additional improvements outlined in this report, this basic system could evolve into a more robust and comprehensive medical question-answering tool.

## References

1. Chung, H. W., et al. (2022). Scaling Instruction-Finetuned Language Models. arXiv preprint arXiv:2210.11416.

2. Wei, J., et al. (2022). Finetuned Language Models Are Zero-Shot Learners. arXiv preprint arXiv:2109.01652.

3. Raffel, C., et al. (2020). Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer. Journal of Machine Learning Research, 21(140), 1-67.

4. Wolf, T., et al. (2020). Transformers: State-of-the-art Natural Language Processing. In Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing: System Demonstrations, 38–45.