import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import nltk
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, GRU, Dense, Dropout, SpatialDropout1D
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import gensim.downloader as api
from gensim.models import Word2Vec

# Download necessary NLTK data
nltk.download('punkt')       # Corrected from 'punkt_tab' to 'punkt'
nltk.download('stopwords')

# Load dataset from Excel file
data = pd.read_excel("Sarcasm_Headlines_Dataset.xlsx")

# Preprocess text data
def preprocess_text(text):
    stop_words = set(stopwords.words('english'))
    tokens = word_tokenize(text.lower())
    tokens = [word for word in tokens if word.isalnum() and word not in stop_words]
    return tokens

data['tokens'] = data['headline'].apply(preprocess_text)

# Split data into train and test sets
X = data['headline']
y = data['is_sarcastic']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Tokenize train and test sets
tokenizer = Tokenizer()
tokenizer.fit_on_texts(X_train)
X_train_seq = tokenizer.texts_to_sequences(X_train)
X_test_seq = tokenizer.texts_to_sequences(X_test)

# Padding sequences
max_sequence_length = 100
X_train_pad = pad_sequences(X_train_seq, maxlen=max_sequence_length, padding='post')
X_test_pad = pad_sequences(X_test_seq, maxlen=max_sequence_length, padding='post')

# Vocabulary size
vocab_size = len(tokenizer.word_index) + 1

# 1. TF-IDF with GRU
def tfidf_gru_model():
    vectorizer = TfidfVectorizer(max_features=5000)
    X_train_tfidf = vectorizer.fit_transform(X_train).toarray()
    X_test_tfidf = vectorizer.transform(X_test).toarray()

    # Reshape TF-IDF output to 3D for GRU (samples, timesteps, features)
    X_train_tfidf = np.expand_dims(X_train_tfidf, axis=1)
    X_test_tfidf = np.expand_dims(X_test_tfidf, axis=1)

    model = Sequential([
        GRU(128, input_shape=(X_train_tfidf.shape[1], X_train_tfidf.shape[2]), dropout=0.2, recurrent_dropout=0.2),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(X_train_tfidf, y_train, epochs=50, batch_size=64, validation_split=0.2, verbose=1)

    y_pred = (model.predict(X_test_tfidf) > 0.5).astype("int32")
    print("TF-IDF Classification Report:\n")
    print(classification_report(y_test, y_pred))

# 2. GloVe Embeddings with GRU
def create_glove_embedding_matrix():
    glove_model = api.load("glove-wiki-gigaword-100")  # Ensure you have internet connection for downloading
    embedding_dim = 100
    embedding_matrix = np.zeros((vocab_size, embedding_dim))
    for word, i in tokenizer.word_index.items():
        if word in glove_model:
            embedding_matrix[i] = glove_model[word]
    return embedding_matrix

def glove_gru_model():
    embedding_matrix = create_glove_embedding_matrix()
    embedding_dim = embedding_matrix.shape[1]

    model = Sequential([
        Embedding(input_dim=vocab_size, output_dim=embedding_dim, weights=[embedding_matrix],
                  input_length=max_sequence_length, trainable=False),
        SpatialDropout1D(0.2),
        GRU(128, dropout=0.2, recurrent_dropout=0.2),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# 3. Word2Vec Embeddings with GRU
def create_word2vec_embedding_matrix():
    w2v_model = Word2Vec(sentences=data['tokens'], vector_size=100, window=5, min_count=1, workers=4)
    embedding_dim = w2v_model.vector_size
    embedding_matrix = np.zeros((vocab_size, embedding_dim))
    for word, i in tokenizer.word_index.items():
        if word in w2v_model.wv:
            embedding_matrix[i] = w2v_model.wv[word]
    return embedding_matrix

def word2vec_gru_model():
    embedding_matrix = create_word2vec_embedding_matrix()
    embedding_dim = embedding_matrix.shape[1]

    model = Sequential([
        Embedding(input_dim=vocab_size, output_dim=embedding_dim, weights=[embedding_matrix],
                  input_length=max_sequence_length, trainable=True),
        SpatialDropout1D(0.2),
        GRU(128, dropout=0.2, recurrent_dropout=0.2),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Train and Evaluate Models
def train_and_evaluate_gru(method):
    if method == "TF-IDF":
        tfidf_gru_model()
    elif method == "GloVe":
        model = glove_gru_model()
        model.fit(X_train_pad, y_train, epochs=50, batch_size=64, validation_split=0.2, verbose=1)
        y_pred = (model.predict(X_test_pad) > 0.5).astype("int32")
        print("GloVe Classification Report:\n")
        print(classification_report(y_test, y_pred))
    elif method == "Word2Vec":
        model = word2vec_gru_model()
        model.fit(X_train_pad, y_train, epochs=50, batch_size=64, validation_split=0.2, verbose=1)
        y_pred = (model.predict(X_test_pad) > 0.5).astype("int32")
        print("Word2Vec Classification Report:\n")
        print(classification_report(y_test, y_pred))
    else:
        print("Invalid method. Choose from 'TF-IDF', 'GloVe', or 'Word2Vec'.")

# Run all methods
methods = ["TF-IDF", "GloVe", "Word2Vec"]
for method in methods:
    print(f"\nRunning {method} Method with GRU:\n")
    train_and_evaluate_gru(method)
