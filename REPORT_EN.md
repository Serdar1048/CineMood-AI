# Project Report: IMDB Sentiment Analysis

## 1. Project Overview
This project aims to classify 50,000 IMDB movie reviews based on their sentiment. Three different Recurrent Neural Network (RNN) architectures were compared to achieve the most accurate sentiment prediction (positive/negative).

## 2. Methodology and Workflow

### A. Data Preprocessing
- **HTML Cleanup:** Removed tags like `<br />` using BeautifulSoup.
- **Normalization:** Converted characters to lowercase and removed punctuation.
- **Stopwords:** Discarded non-informative words to focus on meaningful content.
- **Lemmatization:** Reduced words to their base forms (e.g., *movies* -> *movie*).

### B. Vectorization
- **Tokenizer:** Built a vocabulary of the top 10,000 most frequent words.
- **Padding:** Normalized all reviews to a fixed length of 500 words.

## 3. Models and Architectural Comparison

We trained and evaluated three primary architectures for this project:

### 1. Simple RNN (Baseline)
- **Logic:** Processes the data sequentially, word by word.
- **Issue:** Suffers from the "Vanishing Gradient" problem, meaning it tends to forget information from the beginning of a long sentence.
- **Outcome:** Fast to train but struggles with accuracy on long, complex reviews. It serves as our performance baseline.

### 2. GRU (Gated Recurrent Unit)
- **Logic:** A modern and efficient variant of LSTM with only two gates (Update and Reset Gate).
- **Why it was used:** It has fewer parameters than LSTM, making it faster to train while maintaining high accuracy.
- **Outcome:** Typically performs very close to LSTM but is more computationally efficient.

### 3. LSTM (Long Short-Term Memory) - The Core Engine
- **Logic:** Uses three gates (Forget, Input, Output) to explicitly control information flow and memory duration.
- **Why it was used:** It is the most robust model for capturing "long-term dependencies" in text (e.g., when a "not" at the start flips the meaning of the entire review).
- **Outcome:** Generally provides the most consistent and highest accuracy for sentiment analysis in this project.

## 4. Final Comparison and Results

| Model | Speed | Long-term Memory | Accuracy |
| :--- | :--- | :--- | :--- |
| **Simple RNN** | ⭐⭐⭐ | ⭐ | Low |
| **GRU** | ⭐⭐ | ⭐⭐⭐ | High |
| **LSTM** | ⭐ | ⭐⭐⭐⭐ | Very High |

**Why did LSTM and GRU outperform Simple RNN?**
Movie reviews often contain multi-sentence narrations like "I thought the plot was weak at first, however..." Simple RNNs easily lose the context of the beginning of the review. LSTM and GRU, thanks to their gated memory mechanisms, can hold onto critical context (like "initially weak" vs "eventual brilliance") and arrive at the correct sentiment prediction.
