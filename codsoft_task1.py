import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Embedding, LSTM, Dense, Bidirectional
from tensorflow.keras.callbacks import ModelCheckpoint
import random

# Load and preprocess the dataset (replace with your dataset path)
data = pd.read_csv('path_to_your_dataset.csv')

# Example: Assuming 'text' column contains the handwritten text
texts = data['text'].tolist()

# Tokenize characters
tokenize = Tokenizer(char_level=True)
tokenize.fit_on_texts(texts)

# Total number of unique characters
number_chars = len(tokenize.word_index) + 1

# Convert text to sequences of integers
sequence = tokenize.texts_to_sequences(texts)

# Pad sequences to ensure uniform length
max_length = max([len(seq) for seq in sequence])
sequence = pad_sequences(sequence, maxlen=max_length, padding='post')

# Prepare input-output pairs
X = sequence[:, :-1]
y = sequence[:, -1]

# Convert y to categorical
y = np.expand_dims(y, axis=-1)

# Define the RNN model
model = Sequential([
    Embedding(number_chars, 50, input_length=max_length-1),
    Bidirectional(LSTM(256, return_sequences=True)),
    Bidirectional(LSTM(256)),
    Dense(number_chars, activation='softmax')
])

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())

# Model checkpoint to save the best model during training
checkpoint_path = "model_checkpoint.h5"
checkpoint = ModelCheckpoint(checkpoint_path, monitor='loss', verbose=1, save_best_only=True, mode='min')

# Train the model
model.fit(X, y, epochs=50, batch_size=128, callbacks=[checkpoint])

# Function to generate text based on the trained model with temperature sampling
def generate_text(seed_text, next_words, model, max_sequence_len, temperature=1.0):
    generated_text = seed_text
    for _ in range(next_words):
        token_list = tokenize.texts_to_sequences([seed_text])[0]
        token_list = pad_sequences([token_list], maxlen=max_sequence_len-1, padding='post')
        predictions = model.predict(token_list, verbose=0)[0]
        
        # Apply temperature sampling to generate diverse outputs
        predictions = np.asarray(predictions).astype('float64')
        predictions = np.log(predictions) / temperature
        exp_preds = np.exp(predictions)
        predictions = exp_preds / np.sum(exp_preds)
        
        predicted_id = np.random.choice(len(predictions), p=predictions)
        
        output_word = ""
        for word, index in tokenize.word_index.items():
            if index == predicted_id:
                output_word = word
                break
        seed_text += output_word
        generated_text += output_word
    return generated_text

# Example of generating new text
generated_text = generate_text("example input ", 100, model, max_length)
print(generated_text)
