import streamlit as st
import pandas as pd
import numpy as np
import config
import pickle
import random
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from data_preprocessing import TextPreprocessor

# --- Page Config ---
st.set_page_config(page_title="Movie Sentiment Master 🎬", page_icon="🍿")

# --- Load Model and Tokenizer ---
@st.cache_resource
def load_assets():
    model = load_model(config.MODEL_PATH)
    with open(config.TOKENIZER_PATH, 'rb') as handle:
        tokenizer = pickle.load(handle)
    return model, tokenizer

# Initialize preprocessor
preprocessor = TextPreprocessor()

# --- Fun Responses ---
positive_responses = [
    "Ooo, looks like you’ve got great taste! You really loved this one! ✨",
    "Superb! This movie seems to have captured your heart... ❤️",
    "Wow, you're practically spellbound! Excellent choice! 🍿",
    "Clearly, this film has climbed to the top of your list. You're enjoying this! 🚀",
    "You're radiating positive energy! The movie must be as bright as you! 💎",
    "Your love for this movie is higher than its Metascore! 😍",
    "That’s an 'Oscar-worthy' level of praise, I admire your taste! 🏆",
    "It seems this film took you to a dream world and you never wanted to come back! 🌈",
    "You must be over the moon right now, this movie was made for you! 🎈",
    "If Hollywood saw this comment, they’d remake the movie just for you! 🎬",
    "Taste is subjective, but yours is objectively fantastic! 🍷",
    "You’ve scattered stars all over this review, they’re glowing off the screen! ⭐",
    "If you like it, we respect it—that’s a king-level choice! 👑",
    "I can tell that smile is still on your face even after the credits... 😊",
    "A true cinephile's review! You truly lived the movie! 🔥"
]

negative_responses = [
    "Oh, come on! Was it really that bad? 🥺",
    "Yikes! It looks like this film really let you down... 💔",
    "Seems like there was no 'spark' between you and this movie. Oh well! ⛈️",
    "No way! Did your mood drop, or was the movie just that mediocre? 📉",
    "Let’s not ruin our mood... I guess you didn't quite enjoy this one. 🍋",
    "A waste of good popcorn, this movie didn't deserve it! 🍿🗑️",
    "If the director read this, they’d probably consider retiring... 👴💤",
    "If I had a time machine, I’d send you back 2 hours before you watched this! ⏳",
    "Your eyes didn't deserve this torture, go watch a cat video to recover! 🐱",
    "I bet the best part of the movie was when it ended, right? 😂",
    "Your review is so harsh, the IMDb score might actually drop because of you! 📉💥",
    "Watching paint dry might have been more entertaining than this... 🎨🐢",
    "Your love for this movie is lighter than an ant! 🐜",
    "I hope you fell asleep during this so your time wasn't completely wasted! 😴",
    "Maybe don't pick the movie next time, luck isn't on your side today! 😜"
]

# --- UI Layout ---
st.title("🎬 CineMood AI")
st.write("Type your movie review below in English, and our AI will analyze your inner world (or the movie's quality)! 😎")

# Session state to track if input has changed
if 'last_input' not in st.session_state:
    st.session_state['last_input'] = ""

user_input = st.text_area("Write your review here (English):", placeholder="Example: This movie was absolutely brilliant! I loved every second of it.")

# Check if input has changed since last analysis
button_disabled = (user_input.strip() == "" or user_input == st.session_state['last_input'])

if st.button("Analyze! 🔍", disabled=button_disabled):
    # Update last input
    st.session_state['last_input'] = user_input
    
    try:
        # 1. Load assets
        model, tokenizer = load_assets()
        
        # 2. Preprocess input
        cleaned_text = preprocessor.clean(user_input)
        
        # 3. Vectorize
        seq = tokenizer.texts_to_sequences([cleaned_text])
        padded_seq = pad_sequences(seq, maxlen=config.MAX_LEN)
        
        # 4. Predict
        prediction = model.predict(padded_seq)[0][0]
        
        # 5. Show Results
        st.divider()
        
        if prediction > 0.5:
            # Positive
            st.balloons()
            st.success(f"**RESULT: POSITIVE!** (I'm {prediction*100:.1f}% sure!)")
            st.subheader(random.choice(positive_responses))
        else:
            # Negative
            st.error(f"**RESULT: NEGATIVE!** (I'm {(1-prediction)*100:.1f}% sure!)")
            st.subheader(random.choice(negative_responses))
            
    except Exception as e:
        st.error(f"Something went wrong! (Did you train the model?): {e}")

if button_disabled and user_input.strip() != "":
    st.info("💡 Please change the review for a new analysis.")


# --- Footer ---
st.markdown("---")
st.caption("Yapay Zeka ile Sentiment Analizi Projesi | IMDB Dataset 🎥")
