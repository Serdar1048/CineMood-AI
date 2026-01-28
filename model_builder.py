from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, SimpleRNN, LSTM, GRU, Dense, Dropout
import config

def build_rnn_model():
    """Builds a Simple RNN model."""
    model = Sequential([
        Embedding(config.MAX_WORDS, config.EMBEDDING_DIM, input_length=config.MAX_LEN),
        SimpleRNN(64, dropout=0.2, recurrent_dropout=0.2),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def build_lstm_model():
    """Builds an LSTM model."""
    model = Sequential([
        Embedding(config.MAX_WORDS, config.EMBEDDING_DIM, input_length=config.MAX_LEN),
        LSTM(64, dropout=0.2, recurrent_dropout=0.2),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def build_gru_model():
    """Builds a GRU model."""
    model = Sequential([
        Embedding(config.MAX_WORDS, config.EMBEDDING_DIM, input_length=config.MAX_LEN),
        GRU(64, dropout=0.2, recurrent_dropout=0.2),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model
