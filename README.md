# Twitter Sentiment Analysis

This repository contains a Python script for sentiment analysis of tweets using logistic regression and TF-IDF. It classifies tweets into positive, negative, and neutral sentiments. It leverages the TfidfVectorizer to convert tweet text into numerical feature vectors and uses logistic regression for classification.

## Dataset

The sentiment analysis is performed on a Twitter dataset. This dataset can be found which can be found [here](https://www.kaggle.com/datasets/saurabhshahane/twitter-sentiment-dataset).

## Usage

- Install the required packages:

   ```shell
   pip install -r requirements.txt
- To train this model [sentiment_analysis.py]() and evaluate it on the test data, you can run the script and modify it as necessary.

   ```shell
   python sentiment_analysis.py
   
## Results

- The model achieved a test accuracy of 92.28%
- Precision, Recall, and F1-score for each sentiment label:

 ```shell
  
              precision   recall f1-score  support

     -1.0       0.92      0.82      0.86      7179
      0.0       0.91      0.98      0.94     11034
      1.0       0.93      0.94      0.93     14383

accuracy                            0.92    32596
macro avg       0.92      0.91      0.91    32596
weighted avg    0.92      0.92      0.92    32596
