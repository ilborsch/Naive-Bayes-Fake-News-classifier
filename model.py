from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from database import read_all_articles

import os
import joblib
import pandas as pd


MODEL_SAVE_PATH = 'models/naive_bayes_model.pkl'
VECTORIZER_SAVE_PATH = 'models/tfidf_vectorizer.pkl'

TRUE_DATASET_PATH = 'datasets/true_news.csv'
FAKE_DATASET_PATH = 'datasets/fake_news.csv'

CONFIDENCE_THRESHOLD = 0.6


class ModelNotLoadedError(Exception):
    pass


class NaiveBayesModel:
    """
    Naive Bayes model for article classification
    """

    def __init__(self):
        """
        Initializes the model. If the model is already saved, then loads it from the file
        Otherwise, trains the model from tbe datasets.
        """
        self._model_path = MODEL_SAVE_PATH
        self._vectorizer_path = VECTORIZER_SAVE_PATH
        self.confidence_threshold = CONFIDENCE_THRESHOLD
        self.model = None
        self.vectorizer = None

        if os.path.exists(self._model_path) and os.path.exists(self._vectorizer_path):
            result = self._load_model()
        else:
            result = self._train_model()

        if not result:
            raise ModelNotLoadedError("Could not load the Naive Bayes Model or TF-IDF Vectorizer. Please, check logs.")

    def _load_model(self) -> bool:
        """Load trained model and vectorizer from disk"""
        try:
            self.model = joblib.load(self._model_path)
            self.vectorizer = joblib.load(self._vectorizer_path)
            print("Model loaded successfully")
            return True

        except Exception as e:
            print(f"Error loading model: {str(e)}")
            return False

    def _train_model(self) -> bool:
        """Trains TF-IDF vectorizer and the Multinomial Naive Bayes model"""
        try:
            true_news = pd.read_csv(TRUE_DATASET_PATH)
            fake_news = pd.read_csv(FAKE_DATASET_PATH)

            true_news['label'] = 1
            fake_news['label'] = 0

            df = pd.concat([true_news, fake_news], ignore_index=True)
            df['content'] = df['title'] + ' ' + df['text']

            X = df['content']
            y = df['label']

            vectorizer = TfidfVectorizer(
                max_features=5000,
                stop_words='english',
                ngram_range=(1, 2)
            )
            X_vectorized = vectorizer.fit_transform(X)

            model = MultinomialNB()
            model.fit(X_vectorized, y)

            joblib.dump(model, self._model_path)
            joblib.dump(vectorizer, self._vectorizer_path)

            print(f"Model saved to: {self._model_path}")
            print(f"Vectorizer saved to: {self._vectorizer_path}")
            return True

        except Exception as e:
            print(f"Error training model: {str(e)}")
            self.model = None
            self.vectorizer = None
            return False

    def predict_article_label(self, text) -> tuple[str, float]:
        """
        Predict if article is Fake, True, or Uncategorized
        """
        # If model not loaded, use placeholder
        if self.model is None or self.vectorizer is None:
            return "Model Error", 0

        try:
            text_tfidf = self.vectorizer.transform([text])

            probabilities = self.model.predict_proba(text_tfidf)[0]

            fake_prob = probabilities[0]
            true_prob = probabilities[1]

            max_confidence = max(probabilities)

            if max_confidence < self.confidence_threshold:
                label = "Uncertain"
                confidence = max_confidence
            elif true_prob > fake_prob:
                label = "True Article"
                confidence = true_prob
            else:
                label = "Fake Article"
                confidence = fake_prob

            return label, confidence

        except Exception as e:
            print(f"Error during prediction: {str(e)}")
            return "Model Error", 0

    def retrain_model(self) -> bool:
        """
        Adds articles from the articles CSV to the datasets based on their labels, then retrains the model
        """
        try:
            articles_df = read_all_articles()

            if len(articles_df) == 0:
                print("No articles in CSV to add.")
                return True

            true_news = pd.read_csv(TRUE_DATASET_PATH)
            fake_news = pd.read_csv(FAKE_DATASET_PATH)

            true_articles = articles_df[articles_df['fake_label'] == 'True Article']
            fake_articles = articles_df[articles_df['fake_label'] == 'Fake Article']

            # Append to datasets
            for _, article in true_articles.iterrows():
                new_row = pd.DataFrame([{
                    'title': article['title'],
                    'text': article['content'],
                    'subject': article['category'],
                    'date': article['date_added']
                }])
                true_news = pd.concat([true_news, new_row], ignore_index=True)

            for _, article in fake_articles.iterrows():
                new_row = pd.DataFrame([{
                    'title': article['title'],
                    'text': article['content'],
                    'subject': article['category'],
                    'date': article['date_added']
                }])
                fake_news = pd.concat([fake_news, new_row], ignore_index=True)

            # save
            true_news = true_news.drop_duplicates(['title', 'content'])
            fake_news = fake_news.drop_duplicates(['title', 'content'])

            true_news.to_csv(TRUE_DATASET_PATH, index=False)
            fake_news.to_csv(FAKE_DATASET_PATH, index=False)

            print(f"Added {len(true_articles)} true and {len(fake_articles)} fake articles")

        except Exception as e:
            print(f"Error adding articles: {str(e)}")
            return False

        # Retrain model
        return self._train_model()

    def is_model_loaded(self) -> bool:
        """Checks if model is loaded"""
        return self.model is not None and self.vectorizer is not None

