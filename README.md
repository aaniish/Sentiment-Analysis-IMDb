
# Sentiment Analysis on IMDb Reviews

## Overview
This project implements a **sentiment classification model** for IMDb movie reviews using **Natural Language Processing (NLP)** and **Machine Learning (ML)** techniques. The goal is to classify movie reviews as **positive or negative**, leveraging both traditional ML models and deep learning approaches.

## Models Used
- **Naïve Bayes Classifier**: A probabilistic model optimized for text classification.
- **Artificial Neural Network (ANN)**: A deep learning model fine-tuned using **Keras Tuner** for optimal performance.

## Performance Metrics
| Model | Precision | Recall | Accuracy | Error Rate |
|--------|------------|----------|-----------|------------|
| **Naïve Bayes** | 81.5% | 80.4% | 81.3% | 18.7% |
| **ANN** | 85.9% | 90.7% | 88.0% | 11.96% |

## Project Structure
- `data_preprocessing.py` - Data preprocessing and preparation.
- `naive_bayes_classifier.py` - Training a **Naïve Bayes classifier** using Scikit-learn.
- `ann_trainer.py` - Training an **Artificial Neural Network (ANN)** using TensorFlow/Keras with hyperparameter tuning via **Keras Tuner**.
- `model_evaluation.py` - Model evaluation using accuracy, precision, recall, and error rate.
- `train.data.zip` - Training dataset.
- `test.data` - Testing dataset.
- `model1.obj` - Saved Naïve Bayes model.
- `model2.obj` - Saved ANN model.

## Results
- The **ANN model outperformed Naïve Bayes**, achieving a **7.7% improvement in accuracy** and a **36% reduction in error rate**.
- Demonstrates **NLP-driven sentiment classification** and **AI model optimization** techniques.

## Key Features
✅ **Natural Language Processing (NLP)** for sentiment analysis  
✅ **AI Model Optimization** using **Keras Tuner**  
✅ **Machine Learning Model Deployment** for inference  
✅ **Performance Metrics (Precision, Recall, Accuracy, Error Rate)**  

## Future Improvements
- Extend to **Transformer-based models** (e.g., BERT, GPT) for improved accuracy.
- Deploy as a **REST API using Flask or FastAPI**.
- Experiment with **different feature extraction techniques** (TF-IDF, Word Embeddings).

## Author
**Anish Thiriveedhi**  
[GitHub](https://github.com/aaniish) | [LinkedIn](https://linkedin.com/in/anishthiriveedhi)

---
This project demonstrates expertise in **AI model deployment, NLP, and deep learning**
