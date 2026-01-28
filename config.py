# Project Configuration

DATA_PATH = "IMDB Dataset.csv"
MODEL_PATH = "sentiment_model.h5"
TOKENIZER_PATH = "tokenizer.pickle"

# Preprocessing Settings
MAX_WORDS = 10000  # Size of the vocabulary
MAX_LEN = 500      # Review length (padding/truncating threshold)

# Model Training Settings
EMBEDDING_DIM = 100
BATCH_SIZE = 64
EPOCHS = 5
VALIDATION_SPLIT = 0.2
