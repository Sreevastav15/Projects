import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import nltk
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, SimpleRNN, Dense, Dropout, SpatialDropout1D
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import gensim.downloader as api
from gensim.models import Word2Vec

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('stopwords')

# Load dataset
data = pd.read_excel("/content/Sarcasm_Headlines_Dataset.xlsx")

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
X_train_tokens = X_train.apply(preprocess_text)
X_test_tokens = X_test.apply(preprocess_text)

# Tokenizer for text sequences
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

# 1. TF-IDF Method
def tfidf_method():
    vectorizer = TfidfVectorizer(max_features=5000)
    X_train_tfidf = vectorizer.fit_transform(X_train).toarray()
    X_test_tfidf = vectorizer.transform(X_test).toarray()

    model = Sequential([
        Dense(128, activation='relu', input_shape=(X_train_tfidf.shape[1],)),
        Dropout(0.3),
        Dense(128, activation='relu'),
        Dropout(0.3),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(X_train_tfidf, y_train, epochs=50, batch_size=64, validation_split=0.2, verbose=1)

    y_pred = (model.predict(X_test_tfidf) > 0.5).astype("int32")
    print("TF-IDF Classification Report:\n")
    print(classification_report(y_test, y_pred))

# 2. GloVe Embeddings
def create_glove_embedding_matrix():
    glove_model = api.load("glove-wiki-gigaword-100")
    embedding_dim = 100
    embedding_matrix = np.zeros((vocab_size, embedding_dim))
    for word, i in tokenizer.word_index.items():
        if word in glove_model:
            embedding_matrix[i] = glove_model[word]
    return embedding_matrix

def glove_rnn_model():
    embedding_matrix = create_glove_embedding_matrix()
    embedding_dim = embedding_matrix.shape[1]

    model = Sequential([
        Embedding(input_dim=vocab_size, output_dim=embedding_dim, weights=[embedding_matrix],
                  input_length=max_sequence_length, trainable=False),
        SpatialDropout1D(0.2),
        SimpleRNN(128, dropout=0.2, return_sequences=False),  # SimpleRNN expects 3D input
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# 3. Word2Vec Embeddings
def create_word2vec_embedding_matrix():
    w2v_model = Word2Vec(sentences=X_train_tokens, vector_size=100, window=5, min_count=1, workers=4)
    embedding_dim = w2v_model.vector_size
    embedding_matrix = np.zeros((vocab_size, embedding_dim))
    for word, i in tokenizer.word_index.items():
        if word in w2v_model.wv:
            embedding_matrix[i] = w2v_model.wv[word]
    return embedding_matrix

def word2vec_rnn_model():
    embedding_matrix = create_word2vec_embedding_matrix()
    embedding_dim = embedding_matrix.shape[1]

    model = Sequential([
        Embedding(input_dim=vocab_size, output_dim=embedding_dim, weights=[embedding_matrix],
                  input_length=max_sequence_length, trainable=True),
        SpatialDropout1D(0.2),
        SimpleRNN(128, dropout=0.2, return_sequences=False),  # SimpleRNN expects 3D input
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Train and Evaluate Models
def train_and_evaluate_rnn(method):
    if method == "TF-IDF":
        tfidf_method()
    elif method == "GloVe":
        model = glove_rnn_model()
        print(model.summary())
        model.fit(X_train_pad, y_train, epochs=50, batch_size=64, validation_split=0.2, verbose=1)
        y_pred = (model.predict(X_test_pad) > 0.5).astype("int32")
        print("GloVe Classification Report:\n")
        print(classification_report(y_test, y_pred))
    elif method == "Word2Vec":
        model = word2vec_rnn_model()
        print(model.summary())
        model.fit(X_train_pad, y_train, epochs=50, batch_size=64, validation_split=0.2, verbose=1)
        y_pred = (model.predict(X_test_pad) > 0.5).astype("int32")
        print("Word2Vec Classification Report:\n")
        print(classification_report(y_test, y_pred))
    else:
        print("Invalid method. Choose from 'TF-IDF', 'GloVe', or 'Word2Vec'.")

# Run all methods
methods = ["TF-IDF", "GloVe", "Word2Vec"]
for method in methods:
    print(f"\nRunning {method} Method:\n")
    train_and_evaluate_rnn(method)
