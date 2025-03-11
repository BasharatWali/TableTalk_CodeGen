# Table Data Code Generator

This repository is the implementation of a fine-tuned version of meta-llama/CodeLlama-7b-Instruct-hf hosted on [CodeLlama-Instruct-Python-7b](https://huggingface.co/basharatwali/CodeLlama-Instruct-Python-7b)
, optimized for generating concise, single-line Python code solutions to answer questions over tabular data. The model was fine-tuned using the CardiffNLP/DataBench dataset, which encompasses over 65 diverse tabular datasets, ensuring robust performance across various data types.

## Overview

The goal of this project is to streamline the generation of Python code snippets tailored to query and manipulate tabular data. By leveraging the efficiency of TRL (Transformer Reinforcement Learning) and the memory-friendly QLoRA fine-tuning method, this model provides accurate and efficient one-liners that simplify routine data analysis tasks. Our evaluation shows that the model achieves a 76% accuracy rate on a dedicated evaluation set, demonstrating its practical utility in real-world scenarios.

## Pipeline

The fine-tuning process involves the following steps:

    Data Cleaning:
    Raw data from the CardiffNLP/DataBench collection is cleaned and preprocessed to ensure consistency and quality.

    Data Preparation:
    The cleaned data is formatted and structured appropriately for the fine-tuning process.

    Model Loading:
    The base model, CodeLlama-7B-Instruct-hf, is loaded to serve as the starting point for fine-tuning.

    Fine-tuning with QLoRA:
    The model is fine-tuned using QLoRA, a quantization method that enables efficient training with reduced memory footprint while preserving performance.

    Deployment:
    Once fine-tuning is complete, the model is pushed to Hugging Face, making it available for inference and further testing.

    Evaluation:
    The fine-tuned model is rigorously evaluated on a set of test questions related to tabular data, achieving an accuracy of 76%.

## Features

    Single-line Code Generation:
    Generates concise Python code snippets that address specific queries over tabular data.

    Broad Dataset Coverage:
    Fine-tuned on a comprehensive dataset covering over 65 different tabular data sources.

    Efficient Training:
    Utilizes QLoRA for fine-tuning, ensuring computational efficiency without compromising accuracy.

    High Evaluation Accuracy:
    Achieves a robust 76% accuracy on an evaluation set, validating its effectiveness for real-world applications.

    TRL Framework:
    Leverages Transformer Reinforcement Learning (TRL) to optimize training dynamics.



