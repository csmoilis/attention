# Portfolio Assignment 1 (M4): SGD Mechanics & Attention Context

## Project Overview
This project is a technical exploration into the core mechanics of machine learning, conducted as part of the **Business Data Science Master's program at Aalborg University**. It bridges the gap between manual optimization math and modern deep learning architectures.

### Authors
* **Maleha Afzal**
* **Faraiba Farnan**
* **Cristian Smoilis**

---

## Part A: Manual Stochastic Gradient Descent (SGD)
This section demonstrates the fundamental learning mechanics of a neural network by manually tracing how weights update over individual samples.

* **Objective:** To understand how SGD incrementally adjusts parameters to minimize prediction error.
* **Dataset:** The Kaggle insurance dataset, specifically using the relationship between `age` and `expenses`.
* **Model:** A simple linear model defined by the formula:  
  $$\hat{y} = x \cdot w$$
* **Mechanics Covered:**
    * Initializing weights and learning rates ($\alpha$).
    * Calculating the gradient: $2 \cdot x \cdot (\hat{y} - t)$.
    * Updating the weight based on the direction and magnitude of the error.

[Image of stochastic gradient descent weight update steps on a loss curve]

---

## Part B: Contextualization Using Self-Attention
This section explores the **Self-Attention mechanism**, which is the cornerstone of Transformer models. It demonstrates how a word's vector representation is reweighted based on surrounding tokens.

* **Concept:** Word Sense Disambiguation for homonyms (words that look the same but have different meanings).
* **Case Study:** The word **"Bark"**.
    * *Sentence 1:* "The rough **bark** of the oak tree" (Biological context).
    * *Sentence 2:* "We heard a loud **bark** from the backyard" (Acoustic context).
* **Workflow:**
    * **Embeddings:** Initial 2D vector representations for words.
    * **Similarity:** Calculating dot products between Queries ($Q$) and Keys ($K$) to determine importance.
    * **Weights:** Applying the **Softmax** function to normalize attention scores
