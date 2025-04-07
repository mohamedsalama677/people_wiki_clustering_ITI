from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
import re
import nltk
import pandas as pd
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('punkt_tab')
def preprocess_text_simple(text):
    if pd.isna(text):
        return ""

    # Convert to lowercase
    text = str(text).lower()

    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)

    # Remove email addresses
    text = re.sub(r'\S+@\S+', '', text)

    # Remove special characters and numbers
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()

    # Tokenization
    tokens = word_tokenize(text)

    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    custom_stopwords = {'also', 'would', 'could', 'may', 'might', 'must', 'need'}
    stop_words.update(custom_stopwords)

    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens if token not in stop_words]

    # Remove short words (length < 3)
    tokens = [token for token in tokens if len(token) > 2]

    return ' '.join(tokens)

def tfidf_scaled(df):

    tfidf = TfidfVectorizer(
        max_features=500,          # Adjust based on your needs
        min_df=5,                   # Minimum document frequency
        max_df=0.95,               # Maximum document frequency
        ngram_range=(1, 2),        # Include bigrams
        stop_words='english',
        use_idf=True,
        smooth_idf=True,
        sublinear_tf=True          # Apply sublinear scaling
    )
    text_features = tfidf.fit_transform(df)
    # Convert to dense array
    text_features_dense = text_features.toarray()
    # 3. Standardize the features
    scaler = StandardScaler()
    text_features_scaled = scaler.fit_transform(text_features_dense)
    return text_features_scaled



