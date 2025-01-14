# Fake News Prediction

A machine learning model that predicts whether a news article is real or fake using Natural Language Processing (NLP) techniques and Logistic Regression.

## Overview

This project implements a binary classification model to detect fake news articles. It uses TF-IDF (Term Frequency-Inverse Document Frequency) vectorization for feature extraction and Logistic Regression for classification.

## Features

- Text preprocessing using NLTK
- Porter Stemming for word normalization
- TF-IDF vectorization
- Logistic Regression classifier
- High accuracy on both training and test datasets

## Dependencies

- Python 3.x
- NumPy
- Pandas
- NLTK
- scikit-learn
- re (Regular Expressions)

## Dataset

The dataset contains the following columns:
- id: Unique identifier for each news article
- title: Title of the news article
- author: Author of the article
- text: Main content of the article
- label: Binary classification (0 for real, 1 for fake)

## Model Performance

- Training Accuracy: 98.64%
- Testing Accuracy: 97.91%

## Implementation Details

1. Data Preprocessing:
   - Handling missing values
   - Merging author and title information
   - Text cleaning and normalization
   - Stemming using Porter Stemmer
   - Removing stopwords

2. Feature Engineering:
   - Converting text data to numerical format using TF-IDF vectorization
   - Feature scaling and normalization

3. Model Training:
   - Dataset split: 80% training, 20% testing
   - Stratified sampling to maintain class distribution
   - Logistic Regression training

## Usage

1. Install required packages:
```bash
pip install numpy pandas nltk scikit-learn
```

2. Download NLTK stopwords:
```python
import nltk
nltk.download('stopwords')
```

3. Load and preprocess your data:
```python
news_dataset = pd.read_csv("path_to_your_dataset.csv")
```

4. Train the model:
```python
model = LogisticRegression()
model.fit(X_train, Y_train)
```

5. Make predictions:
```python
prediction = model.predict(X_new)
if prediction[0] == 0:
    print('The news is Real')
else:
    print('The news is Fake')
```

## Future Improvements

1. Implement additional feature engineering techniques
2. Try other classification algorithms
3. Add cross-validation
4. Include more text preprocessing steps
5. Develop a web interface for real-time predictions

## License

[Add your chosen license here]
