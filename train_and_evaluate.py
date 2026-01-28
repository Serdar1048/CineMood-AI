import config
import pickle
from text_vectorization import DataProvider
from model_builder import build_rnn_model, build_lstm_model, build_gru_model

def main():
    # 1. Prepare Data
    provider = DataProvider()
    X_train, X_test, y_train, y_test = provider.prepare_data()

    # Save the tokenizer for later use in Streamlit
    print(f"Saving tokenizer to {config.TOKENIZER_PATH}...")
    with open(config.TOKENIZER_PATH, 'wb') as handle:
        pickle.dump(provider.tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # 2. Train and Evaluate RNN (Baseline)
    # print("\n--- Training Simple RNN ---")
    # rnn_model = build_rnn_model()
    # rnn_model.fit(X_train, y_train, epochs=config.EPOCHS, batch_size=config.BATCH_SIZE, validation_split=config.VALIDATION_SPLIT, verbose=1)
    # rnn_loss, rnn_acc = rnn_model.evaluate(X_test, y_test, verbose=0)
    # print(f"RNN Test Accuracy: {rnn_acc:.4f}")

    # 3. Train and Evaluate GRU (Faster but powerful)
    print("\n--- Training GRU ---")
    gru_model = build_gru_model()
    gru_model.fit(X_train, y_train, epochs=config.EPOCHS, batch_size=config.BATCH_SIZE, validation_split=config.VALIDATION_SPLIT, verbose=1)
    gru_loss, gru_acc = gru_model.evaluate(X_test, y_test, verbose=0)
    print(f"GRU Test Accuracy: {gru_acc:.4f}")

    # 4. Train and Evaluate LSTM (Full memory)
    # print("\n--- Training LSTM ---")
    # lstm_model = build_lstm_model()
    # lstm_model.fit(X_train, y_train, epochs=config.EPOCHS, batch_size=config.BATCH_SIZE, validation_split=config.VALIDATION_SPLIT, verbose=1)
    # lstm_loss, lstm_acc = lstm_model.evaluate(X_test, y_test, verbose=0)
    # print(f"LSTM Test Accuracy: {lstm_acc:.4f}")

    # Save the best model
    print(f"Saving GRU model (as current best) to {config.MODEL_PATH}...")
    gru_model.save(config.MODEL_PATH)

    # 5. Final Comparison
    print("\n--- Final Results (GRU Only) ---")
    # print(f"Simple RNN Accuracy: {rnn_acc:.4f}")
    print(f"GRU Accuracy:        {gru_acc:.4f}")
    # print(f"LSTM Accuracy:       {lstm_acc:.4f}")

if __name__ == "__main__":
    main()
