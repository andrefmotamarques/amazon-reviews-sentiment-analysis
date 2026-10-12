# Sentiment Analysis of Amazon Food Reviews

Binary sentiment classification (positive / negative) of Amazon food product reviews, comparing four generations of NLP approaches on the same test set: lexicon and rule-based models, classical machine learning, Transformers and LLMs.

Team project for the *Text Mining* course, MSc in Data Science, ISCTE (2025/26), developed with Afonso Abreu and Afonso Claudino.

## Approaches compared

- **Lexicon / rule-based:** TextBlob, VADER, Stanza, NRC EmoLex (with negation handling and valence shifting)
- **Classical ML:** Naive Bayes, Logistic Regression, SVM, Decision Tree, Random Forest, using Bag of Words, TF-IDF and Word2Vec features
- **Transformers:** DistilBERT (SST-2 off the shelf, and fine-tuned on the training set)
- **LLM:** GPT-4o (zero-shot and few-shot prompts, on a small sample)

## Results (selected)

| Model | Accuracy | Macro F1 |
|---|---|---|
| VADER | 0.79 | 0.71 |
| Naive Bayes, BoW, 10K features, negation | 0.85 | 0.83 |
| Logistic Regression, TF-IDF, 5K features | 0.84 | 0.83 |
| SVM, TF-IDF, 15K features, negation | 0.86 | 0.84 |
| DistilBERT, fine-tuned (3 epochs) | **0.92** | **0.91** |
| GPT-4o, zero-shot domain prompt* | 0.92 | 0.92 |

\*GPT-4o was evaluated on a much smaller sample because of API cost and rate limits.

Macro F1 was used as the main metric because the dataset is imbalanced (about 70% positive reviews), so accuracy alone hides poor performance on the negative class.

**Takeaway:** fine-tuned DistilBERT gave the best trade-off between accuracy and cost. A linear SVM with TF-IDF and negation handling stayed surprisingly competitive at a fraction of the compute.

![Words with the highest SVM weights](images/svm_top_words.png)

## Repository structure

```
sentiment_analysis.ipynb   # full pipeline: preprocessing, models, evaluation
report.pdf                 # full written report (Portuguese)
data/                      # place the dataset files here (not included)
```

## How to run

1. Put `amazon_reviews_train.csv`, `amazon_reviews_test.csv` and `NRC-lexicon.csv` in `data/`. The datasets were provided by the course and are not redistributed here.
2. Install the main dependencies: `pandas scikit-learn nltk textblob vaderSentiment stanza gensim transformers torch wordcloud matplotlib seaborn`.
3. Run `sentiment_analysis.ipynb`. The notebook keeps its outputs, so results can be read without re-running it.

## Tech stack

Python · pandas · scikit-learn · NLTK · Stanza · Gensim · Hugging Face Transformers · PyTorch

## Team

André Marques, Afonso Abreu and Afonso Claudino. The full report with the team's methodology and discussion is in [`report.pdf`](report.pdf) (Portuguese).
