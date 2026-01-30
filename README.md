# 📰 Fact Checked News - AI-Powered News Article Manager

A Streamlit-based web application that uses machine learning to classify news articles as **True**, **Fake**, or **Uncertain**. The app provides a complete CRUD interface for managing news articles with real-time AI-powered fact-checking.

## 🎯 About the Project

This application combines a user-friendly news management system with a Naive Bayes classifier to help identify potentially fake news articles. Users can create, read, update, and delete news articles while the ML model automatically analyzes and labels each article based on its content.

### Key Features
- **AI-Powered Classification**: Automatic detection of fake news using Naive Bayes algorithm
- **CRUD Operations**: Full article management (Create, Read, Update, Delete)
- **Search & Filter**: Search by title/category and filter by article authenticity
- **Confidence Scoring**: View model confidence levels for each prediction
- **CSV-Based Storage**: Lightweight data persistence without database setup
- **Interactive UI**: Clean, modern interface built with Streamlit

## 🧠 Machine Learning Model

### Model Architecture
- **Algorithm**: Multinomial NB model
- **Vectorization**: TF-IDF
- **Features**: 5000 max features with 1-2 word tokens
- **Confidence Threshold**: 60% (articles below this are marked as "Uncertain")

### Training Data
The model is trained on two datasets:
- `datasets/true_news.csv` - Verified authentic news articles
- `datasets/fake_news.csv` - Known fake news articles

The dataset is free to use and downloaded from this source: https://www.kaggle.com/datasets/emineyetm/fake-news-detection-datasets

### Model Files
Pre-trained models are stored in the `models/` directory:
- `naive_bayes_model.pkl` - Trained Naive Bayes classifier
- `tfidf_vectorizer.pkl` - Fitted TF-IDF vectorizer

### Classification Logic
1. Combines article title and content
2. Transforms text using TF-IDF vectorizer
3. Predicts probability of being fake/true
4. Returns label based on confidence threshold:
   - **True Article**: High confidence (>60%) that article is authentic
   - **Fake Article**: High confidence (>60%) that article is fake
   - **Uncertain**: Low confidence (<60%) - needs human review

## 🛠️ Implementation

### Project Structure
```
absbs/
├── app.py                    # Main Streamlit application
├── model.py                  # ML model class and training logic
├── database.py               # CSV-based data operations
├── datasets/                 # Training datasets
│   ├── fake_news.csv
│   └── true_news.csv
├── models/                   # Saved ML models
│   ├── naive_bayes_model.pkl
│   └── tfidf_vectorizer.pkl
├── news_articles.csv         # User-created articles storage
└── training model.ipynb      # Model training notebook
```

### Core Components

**app.py** - Main application with:
- Streamlit UI components
- Page routing (home, article detail, create)
- Article cards with CRUD operations
- Search and filtering functionality

**model.py** - ML model handler:
- Model loading/training
- Article classification
- Confidence scoring
- Model retraining capability

**database.py** - Data layer:
- CSV file operations
- CRUD functions for articles
- Search and pagination support

## 🚀 How to Run

### Installation

1. **Install required dependencies**
```bash
pip install -r requirements.txt
```

### Running the Application

**Start the Streamlit app:**
```bash
streamlit run app.py
```

The application will automatically:
- Load the pre-trained model (or train a new one if models don't exist.)
- Initialize the CSV database
- Open in your default browser at `http://localhost:8501`

### First-Time Setup
If model files (/model/*.pkl) don't exist, the app will automatically train the model using the datasets. This may take a few moments on first launch.

## 📝 Usage

1. **View Articles**: Browse all articles on the home page with their AI-generated labels
2. **Create Article**: Click "Create New Article" → Fill in details → Review AI prediction → Confirm
3. **Read Article**: Click "Read the article" to view full content
4. **Update Article**: Click "Update" → Modify content → Review new AI prediction → Save
5. **Delete Article**: Click "Delete" → Confirm deletion
6. **Search**: Use the search bar to filter by title or category
7. **Filter**: Toggle "Include True only" to show only verified articles

## 📦 Dependencies

- **streamlit** - Web application framework
- **pandas** - Data manipulation and CSV handling
- **scikit-learn** - Machine learning (Naive Bayes, TF-IDF)
- **joblib** - Model serialization
