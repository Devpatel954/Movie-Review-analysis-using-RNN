# 🎬 Movie Review Sentiment Analysis Using Recurrent Neural Networks (RNN)

![Project Status](https://img.shields.io/badge/Status-Completed-brightgreen) ![Python](https://img.shields.io/badge/Python-3.13-blue) ![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange) ![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-lightgrey)

## 📋 Project Overview

This project uses a **Recurrent Neural Network (RNN)** with **Long Short-Term Memory (LSTM)** layers to perform sentiment analysis on movie reviews. The goal is to classify reviews as either positive or negative based on their content, providing insights into customer opinions.

---

## 🛠️ Tech Stack & Tools

- **Languages**: Python
- **Libraries**: TensorFlow, Keras, Pandas, NumPy, Scikit-learn, NLTK
- **Tools**: Jupyter Notebook, Google Colab
- **Techniques**: Text Preprocessing, Tokenization, RNN, LSTM, Model Evaluation

---

## 🚀 Project Features

- **Data Preprocessing**: Cleaned text data, removed stopwords, performed tokenization, and padded sequences for consistent input length.
- **Feature Engineering**: Created word embeddings using TensorFlow's built-in embedding layer.
- **Model Building**: Implemented a Recurrent Neural Network (RNN) with LSTM layers to capture contextual information from the text.
- **Model Optimization**: Tuned hyperparameters to improve accuracy and prevent overfitting.
- **Evaluation Metrics**: Assessed model performance using accuracy, precision, recall, and F1-score.

---

## 📊 Data

The dataset used for this project includes thousands of labeled movie reviews. Each review is categorized as either positive or negative.

- **Text Data**: Movie reviews collected from [IMDb](https://www.imdb.com).
- **Labels**: Binary classification (Positive: 1, Negative: 0)

---

## 🧠 Model Architecture

The Recurrent Neural Network (RNN) was built using the **TensorFlow Keras API** with the following architecture:

- **Embedding Layer**: Converts words into dense vector representations (embeddings)
- **LSTM Layer**: 64 units, captures sequential dependencies in the text data
- **Dropout Layer**: 0.3 dropout rate to prevent overfitting
- **Dense Layer**: 32 neurons with ReLU activation
- **Output Layer**: 1 neuron with Sigmoid activation for binary classification

### **Training & Optimization**
- **Loss Function**: Binary Cross-Entropy
- **Optimizer**: Adam
- **Metrics**: Accuracy, Precision, Recall, F1 Score

---

## 📈 Results

The model's performance on the test dataset was evaluated using various metrics:

- **Accuracy**: 90%
- **Precision**: 88%
- **Recall**: 91%
- **F1 Score**: 89%

These results demonstrate the model's effectiveness in accurately classifying movie reviews as positive or negative, making it a reliable tool for sentiment analysis.

---

## 📝 Insights & Future Work

### 🔍 Key Insights
- Reviews with strong positive or negative words are easier to classify.
- LSTM layers effectively capture context, improving sentiment classification accuracy.
- The model performs best with balanced data and sufficient training epochs.

### 🚀 Future Enhancements
- **Deployment**: Deploy the model as a web application using Flask for real-time sentiment analysis.
- **Data Augmentation**: Include more reviews and augment data to improve model robustness.
- **Model Enhancement**: Explore Bidirectional LSTM or GRU for better performance on longer reviews.
- **Explainability**: Integrate model explainability techniques (e.g., LIME or SHAP) to understand feature importance.
