import config
import pickle
from text_vectorization import DataProvider
from model_builder import build_rnn_model, build_lstm_model

def main():
    # 1. Prepare Data
    provider = DataProvider()
    X_train, X_test, y_train, y_test = provider.prepare_data()

    # Save the tokenizer for later use in Streamlit
    print(f"Saving tokenizer to {config.TOKENIZER_PATH}...")
    with open(config.TOKENIZER_PATH, 'wb') as handle:
        pickle.dump(provider.tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # 2. Train and Evaluate RNN
    # ... (RNN training code remains or can be skipped if only LSTM is needed for production)
    print("\n--- Training Simple RNN ---")
    rnn_model = build_rnn_model()
    rnn_model.fit(X_train, y_train, epochs=config.EPOCHS, batch_size=config.BATCH_SIZE, validation_split=config.VALIDATION_SPLIT, verbose=1)
    rnn_loss, rnn_acc = rnn_model.evaluate(X_test, y_test, verbose=0)
    print(f"RNN Test Accuracy: {rnn_acc:.4f}")

    # 3. Train and Evaluate LSTM
    print("\n--- Training LSTM ---")
    lstm_model = build_lstm_model()
    lstm_model.fit(X_train, y_train, epochs=config.EPOCHS, batch_size=config.BATCH_SIZE, validation_split=config.VALIDATION_SPLIT, verbose=1)
    lstm_loss, lstm_acc = lstm_model.evaluate(X_test, y_test, verbose=0)
    print(f"LSTM Test Accuracy: {acc := lstm_acc:.4f}")

    # Save the best model (LSTM)
    print(f"Saving LSTM model to {config.MODEL_PATH}...")
    lstm_model.save(config.MODEL_PATH)

    # 4. Final Results
    print(f"\nTraining finished. LSTM accuracy: {acc:.4f}")

if __name__ == "__main__":
    main()
