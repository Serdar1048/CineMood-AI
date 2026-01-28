import pandas as pd
import numpy as np
import config
from data_preprocessing import TextPreprocessor
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split

class DataProvider:
    def __init__(self):
        self.preprocessor = TextPreprocessor()
        self.tokenizer = Tokenizer(num_words=config.MAX_WORDS)

    def prepare_data(self):
        print(f"Loading data from {config.DATA_PATH}...")
        df = pd.read_csv(config.DATA_PATH)
        
        print("Cleaning reviews (this may take a while)...")
        df['cleaned_review'] = df['review'].apply(self.preprocessor.clean)
        df['sentiment'] = df['sentiment'].map({'positive': 1, 'negative': 0})
        
        print("Tokenizing and Padding...")
        self.tokenizer.fit_on_texts(df['cleaned_review'])
        sequences = self.tokenizer.texts_to_sequences(df['cleaned_review'])
        
        X = pad_sequences(sequences, maxlen=config.MAX_LEN)
        y = df['sentiment'].values
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=config.VALIDATION_SPLIT, random_state=42
        )
        
        print(f"Data Prepared. Training samples: {len(X_train)}, Test samples: {len(X_test)}")
        return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    provider = DataProvider()
    provider.prepare_data()
