# Medical Question-Answering LLM Fine-Tuning Project

## Project Overview
This project implements a fine-tuned language model for medical question answering. It takes a pre-trained Flan-T5-base model and fine-tunes it on a curated dataset of medical questions and answers to create a specialized model for answering medical queries.

##Presentation demo link - https://northeastern-my.sharepoint.com/:v:/r/personal/kurlekar_a_northeastern_edu/Documents/Medical%20LLM%20fine%20tuning.mp4?csf=1&web=1&e=QfThzL&nav=eyJyZWZlcnJhbEluZm8iOnsicmVmZXJyYWxBcHAiOiJTdHJlYW1XZWJBcHAiLCJyZWZlcnJhbFZpZXciOiJTaGFyZURpYWxvZy1MaW5rIiwicmVmZXJyYWxBcHBQbGF0Zm9ybSI6IldlYiIsInJlZmVycmFsTW9kZSI6InZpZXcifX0%3D

## Table of Contents
1. [Installation Requirements](#installation-requirements)
2. [Dataset](#dataset)
3. [Model Architecture](#model-architecture)
4. [Fine-Tuning Process](#fine-tuning-process)
5. [Hyperparameter Selection](#hyperparameter-selection)
6. [Evaluation Results](#evaluation-results)
7. [Error Analysis](#error-analysis)
8. [Inference Pipeline](#inference-pipeline)
9. [Usage Instructions](#usage-instructions)
10. [Future Improvements](#future-improvements)
11. [Ethical Considerations](#ethical-considerations)

## Installation Requirements
To set up the project environment, run:

```bash
pip install transformers==4.30.2 datasets==2.13.1 torch==2.0.1
```

This project was developed using Python 3.9. Additional dependencies:
- pandas>=1.5.3
- numpy>=1.24.3

## Dataset
The dataset consists of 13 carefully curated medical Q&A pairs covering common medical topics including:
- Chronic conditions (diabetes, hypertension, Alzheimer's)
- Diagnostic procedures (strep throat, pneumonia diagnosis)
- Medications and treatments (asthma medications, statins)
- Emergency conditions (stroke warnings, heart attack symptoms)

Each example follows a consistent instruction/output format optimized for fine-tuning instruction-following models.

### Dataset Statistics:
- Training examples: 13
- Average instruction length: 42.3 characters
- Average response length: 156.7 characters
- Topics covered: 11 distinct medical conditions/topics

## Model Architecture
The project uses Google's Flan-T5-base model, which offers:
- 250M parameters
- Encoder-decoder architecture
- Strong zero-shot and few-shot capabilities
- Prior instruction-tuning on diverse tasks

This model was selected for its balance of performance and efficiency, making it suitable for fine-tuning with limited computational resources while still achieving good results on specialized tasks.

## Fine-Tuning Process
The fine-tuning process follows these steps:

1. **Data Preprocessing**:
   - Tokenization with max sequence length of 128 tokens
   - Addition of "Medical question: " prefix to all inputs
   - Padding and truncation to consistent lengths
   - Special token handling for labels (-100 for padding tokens)

2. **Training Configuration**:
   - Seq2Seq training with appropriate data collator
   - Mixed precision training disabled for stability
   - Checkpointing at each epoch
   - Model and tokenizer saved for inference

## Hyperparameter Selection
The following hyperparameters were selected based on empirical testing and adaptation to the dataset size:

| Parameter | Value | Justification |
|-----------|-------|---------------|
| Learning rate | 1e-4 | Standard for fine-tuning T5 models |
| Batch size | 4 | Optimized for memory constraints |
| Gradient accum. | 2 | Effectively doubles batch size |
| Weight decay | 0.01 | Prevents overfitting on small dataset |
| Epochs | 10 | Extended to compensate for small dataset |

## Evaluation Results
The model was evaluated on both training examples and novel queries. Some key performance insights:

- Strong performance on training data (expected given small dataset size)
- Reasonable generalization to unseen medical questions
- Appropriate recognition of medical terminology
- Concise and relevant answers

Detailed metrics are available in the accompanying technical report.

## Error Analysis
Common error patterns identified:

1. **Hallucination on complex queries**: The model occasionally generates plausible but incorrect details for complex medical conditions not represented in the training data.

2. **Limited detail depth**: Responses sometimes lack the depth of information that would be available from a medical professional.

3. **Confidence issues**: The model doesn't express uncertainty when it should for questions outside its knowledge domain.

## Inference Pipeline
The inference pipeline (`medical_qa_inference.py`) provides:

- Simple command-line interface for medical questions
- Consistent input formatting with the training process
- Response generation with carefully tuned parameters:
  - Temperature: 0.3 (focused outputs with minimal randomness)
  - Beam search: 4 beams (improved quality)
  - N-gram repetition prevention

## Usage Instructions
To use the fine-tuned model:

1. **Setup**:
   ```bash
   # Clone this repository
   git clone https://github.com/yourusername/medical-qa-llm.git
   cd medical-qa-llm
   
   # Install dependencies
   pip install -r requirements.txt
   ```

2. **Inference**:
   ```bash
   python medical_qa_inference.py
   ```

3. **Inside the interactive shell**:
   ```
   Medical QA System
   Type 'exit' or 'quit' to end the session
   
   Medical question: What are the symptoms of type 2 diabetes?
   
   Answer: The main symptoms of type 2 diabetes include increased thirst, frequent urination, excessive hunger, fatigue, blurred vision, slow-healing sores, and recurring infections.
   ```

## Future Improvements
Several enhancements could further improve this model:

1. **Dataset expansion**: Incorporate hundreds more medical QA pairs with greater diversity of conditions, treatments, and medical specialties.

2. **Larger model**: Experiment with Flan-T5-large or Flan-T5-xl for improved performance.

3. **Hyperparameter optimization**: Conduct systematic grid search across learning rates, batch sizes, and training schedules.

4. **Retrieval augmentation**: Integrate with a medical knowledge base for more comprehensive and up-to-date answers.

5. **Uncertainty quantification**: Add mechanisms for the model to express confidence levels in its responses.

## Ethical Considerations
This model is intended for educational and research purposes only and should not replace professional medical advice. Key ethical considerations:

- **Disclaimer**: The model's responses should not be used for self-diagnosis or treatment decisions.
- **Biases**: The training data has been selected to minimize biases, but all models inherit some biases from their pre-training.
- **Limitations**: The model has a limited knowledge base and may not be current with the latest medical research.
- **Privacy**: No personal health information was used in training this model.

## License
This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments
- Google's Flan-T5 team for the pre-trained model
- Hugging Face for the Transformers library
- PyTorch team for the deep learning framework

---

*This project was developed as part of a Large Language Model Fine-Tuning assignment.*
