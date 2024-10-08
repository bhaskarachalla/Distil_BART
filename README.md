# Fine-Tuning DistilBART for Summarization Task

This project focuses on fine-tuning the DistilBART model, a more efficient version of the BART model, to generate text summaries. The model is trained using a dataset containing few chapters from the Mahabharata. The task involves generating concise summaries based on the provided "parv" (chapter) and key events.

# Disclimar: This project is intended solely for educational purposes. The information and results presented may contain inaccuracies due to the limitations of the training data. Users are encouraged to further refine and update the model by incorporating their own datasets, as necessary, to achieve more accurate outcomes.

## Table of Contents
- [Overview](#overview)
- [Model](#model)
- [Dataset](#dataset)
- [Preprocessing](#preprocessing)
- [Training](#training)
- [Evaluation](#evaluation)
- [Usage](#usage)
- [Results](#results)
- [Future Work](#future-work)

  ![image](https://github.com/user-attachments/assets/edbfab27-a1af-41c8-8030-c44172e9bb5e)


## Overview
This notebook fine-tunes the DistilBART model to summarize text from a historical dataset. The model generates summaries based on input "parv" (chapters), helping to condense large text sections into concise summaries.

# Warning:A GPU is strongly recommended for optimal performance. Running this process without a GPU may significantly strain your system, potentially leading to performance issues.

## Model
- **Model Used**: DistilBART (`sshleifer/distilbart-cnn-12-6`).
- DistilBART is a lightweight version of the BART model designed for sequence-to-sequence tasks like summarization. The model is optimized for efficiency while maintaining high performance.

## Dataset
The dataset consists of four columns:
- **Section**: Larger sections of the text (e.g., books or volumes).
- **Parv**: Specific chapters or sub-sections from the Mahabharata.
- **Key Event**: The central event or topic within the section.
- **Summary**: The human-written summary corresponding to the key event.

The dataset is divided into training and validation sets for model training and evaluation.

## Preprocessing
The preprocessing steps include:
- Tokenizing the input "Key Event" and the target "Summary" using DistilBART’s tokenizer.
- Padding and truncating inputs and labels to maintain a uniform length for easier batch processing.
- Splitting the dataset into a training set (90%) and a validation set (10%) for proper evaluation.

## Training
The training process is handled using Hugging Face's `Seq2SeqTrainer` with key parameters:
- **Batch Size**: 16 for both training and validation.
- **Learning Rate**: 2e-5.
- **Number of Epochs**: 18.
- **Evaluation Strategy**: Evaluation takes place at the end of every epoch to assess model performance.
- **Mixed Precision**: FP16 is enabled to optimize training speed and memory usage.

## Evaluation
The model is evaluated on a validation set, using cross-entropy loss as the primary metric. During evaluation, the model also generates summaries based on the input text to assess how well it generalizes to unseen data.

## Usage
After fine-tuning, the model can generate concise summaries based on a "parv" input. This is useful for condensing large text sections into a more manageable and readable format.

## Results
The fine-tuned DistilBART model provides summaries with **moderate** cross-entropy loss on the validation set. It generalizes **okay** to unseen data, making it a useful tool for summarization tasks, particularly for large texts like the Mahabharata, including refining the dataset.

## Future Work
Potential enhancements to this project could include:
- Experimenting with various learning rates, batch sizes, and optimizer settings to further improve model performance.
- Incorporating more diverse datasets or fine-tuning on additional related tasks for better generalization.
- Using techniques such as knowledge distillation or model ensembling to improve the summarization quality.
