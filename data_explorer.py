import pandas as pd
import config
from data_preprocessing import TextPreprocessor

def run_exploration():
    print("--- Starting Data Exploration ---")
    df = pd.read_csv(config.DATA_PATH)
    preprocessor = TextPreprocessor()
    
    print(f"Dataset Size: {len(df)} rows")
    
    # Analyze raw text lengths
    raw_lengths = df['review'].apply(lambda x: len(x.split()))
    
    print(f"Max Raw Review Length: {raw_lengths.max()} words")
    print(f"Avg Raw Review Length: {raw_lengths.mean():.2f} words")
    
    # Process a sample (exploring 1000 reviews for speed)
    print("Analyzing cleaned lengths (Sample of 1000 reviews for speed)...")
    sample_cleaned = df['review'].head(1000).apply(preprocessor.clean)
    cleaned_lengths = sample_cleaned.apply(lambda x: len(x.split()))
    
    print(f"Max Cleaned Review Length (Sample): {cleaned_lengths.max()} words")
    print(f"Avg Cleaned Review Length (Sample): {cleaned_lengths.mean():.2f} words")
    print(f"Selected MAX_LEN in config: {config.MAX_LEN}")
    
    num_exceeding = (cleaned_lengths > config.MAX_LEN).sum()
    print(f"Sample reviews exceeding {config.MAX_LEN} words: {num_exceeding}")
    print("--- Exploration Finished ---")

if __name__ == "__main__":
    run_exploration()
