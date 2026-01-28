import re
import nltk
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Ensure resources are available
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)

class TextPreprocessor:
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words("english"))

    def clean(self, text):
        """Full cleaning pipeline for a single review string."""
        # 1. Remove HTML tags
        text = BeautifulSoup(text, "html.parser").get_text()
        
        # 2. Lowercase and remove non-letters
        text = re.sub(r"[^a-zA-Z]", " ", text.lower())
        
        # 3. Tokenize, remove stopwords, and lemmatize
        words = text.split()
        cleaned_words = [self.lemmatizer.lemmatize(w) for w in words if w not in self.stop_words]
        
        return " ".join(cleaned_words)
