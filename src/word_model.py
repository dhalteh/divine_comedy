import numpy as np
import os
from collections import Counter
import re
import tensorflow as tf
from keras.utils import np_utils
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Embedding
import src.config as config
#tf.logging.set_verbosity(tf.logging.ERROR)
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

#########CONFIGS#########
DATA_PATH = config.DATA_PATH

# LSTM general parameters
ACTIVATION=config.ACTIVATION
LOSS=config.LOSS
OPTIMIZER=config.OPTIMIZER

# LSTM word-level parameters
WORD_MEMORY_UNITS=config.WORD_MEMORY_UNITS
WORD_NUM_EPOCHS=config.WORD_NUM_EPOCHS
WORD_OUTPUT_DIMENSION=config.WORD_OUTPUT_DIMENSION
WORD_DROPOUT=config.WORD_DROPOUT
WORD_MODEL_PATH=config.WORD_MODEL_PATH
#########################


def load_data(path):
    if os.path.isfile(path):
        with open(path) as infile:
            text = infile.readlines()
        return text
    else:
        raise FileNotFoundError('File not found!')
        return None


def parse_data_word(raw):
    cleaned_doc = []
    for line in raw:
        cleaned_line = re.sub('[(){}<>"\'?!&$@%*^\-;!.\d\t]', '', line)
        cleaned_doc.append(cleaned_line.strip().lower())
    # Remove all empty strings
    cleaned_doc = list(filter(None, cleaned_doc))
    return cleaned_doc


def create_token_index(cleaned_lines):
    all_tokens = []
    for line in cleaned_lines:
        line = re.sub('[(){}<>"\'?\-;!.\d]', '', line)
        line_tokens = line.split(" ")
        for token in line_tokens:
            all_tokens.append(token)
    token_counts = Counter(all_tokens)
    sorted_tokens = sorted(token_counts.items(), key=lambda x: x[1], reverse=True)

    word_indices = dict()

    for index, pair in enumerate(sorted_tokens):
        word_indices[pair[0]] = index + 1

    return word_indices


def create_tokenized_sequences(cleaned_lines, word_index_dict):
    tokenized_sequences = []
    for line in cleaned_lines:
        line_to_tokens = [word_index_dict[word] for word in line.split(' ')]
        num_tokens = len(line_to_tokens)
        for n in range(1, num_tokens):
            n_grams = line_to_tokens[:n + 1]
            tokenized_sequences.append(n_grams)
    return tokenized_sequences


def pad_sequences(sequences):
    max_sequence_length = np.max([len(sequence) for sequence in sequences])

    padded_sequences = []
    for sequence in sequences:
        sequence_length = len(sequence)
        if sequence_length < max_sequence_length:
            difference = max_sequence_length - sequence_length
            zeros = np.repeat(0, difference)
            padded_sequence = np.concatenate([zeros, sequence])
            padded_sequences.append(padded_sequence)
        else:
            padded_sequences.append(sequence)

    return np.array(padded_sequences)


def get_sequences_labels(padded_sequences, word_indices):
    word_index_length = len(word_indices)

    prepared_sequences = []
    next_words = []
    for sequence in padded_sequences:
        next_word = sequence[-1]
        prepared_sequence = sequence[:-1]
        prepared_sequences.append(prepared_sequence)
        next_words.append(next_word)

    prepared_sequences = np.array(prepared_sequences)
    next_words = np.array(next_words)
    next_words = np_utils.to_categorical(next_words, num_classes=word_index_length + 1)

    return prepared_sequences, next_words


def pad_sequence(sequence, prepared_sequences):
    max_sequence_length = prepared_sequences.shape[1]
    sequence_length = len(sequence)

    padded_sequence = []
    if len(sequence) < max_sequence_length:
        difference = max_sequence_length - sequence_length
        zeros = np.repeat(0, difference)
        sequence = np.array(sequence)
        joined = np.concatenate([zeros, sequence])
        padded_sequence.append(joined)
    else:
        padded_sequence.append(sequence)

    return np.array(padded_sequence)

def create_LSTM_word(word_indices,
                     input_sequences,
                     output_labels,
                     output_dimension=WORD_OUTPUT_DIMENSION,
                     n_memory_units=WORD_MEMORY_UNITS,
                     dropout=WORD_DROPOUT,
                     epochs=WORD_NUM_EPOCHS,
                     activation=ACTIVATION,
                     loss_function=LOSS,
                     optimizer=OPTIMIZER):
    input_sequence_length = input_sequences.shape[1]
    word_input_count = len(word_indices)+1
    model = Sequential()
    model.add(Embedding(input_dim=word_input_count, output_dim=output_dimension, input_length=input_sequence_length))
    model.add(LSTM(n_memory_units))
    model.add(Dropout(dropout))
    model.add(Dense(word_input_count, activation=activation))
    model.compile(loss=loss_function, optimizer=optimizer)
    model.fit(input_sequences, output_labels, epochs=epochs, verbose=1)
    return model

def predict_text(model, seed, output_length, word_indices, prepared_sequences):
    max_sequence_length = prepared_sequences.shape[1]

    for index in range(output_length):
        token_sequences = create_tokenized_sequences([seed], word_indices)[-1]
        token_sequence_length = len(token_sequences)
        if token_sequence_length > max_sequence_length:
            difference = token_sequence_length - max_sequence_length
            token_sequences = token_sequences[difference:]
        padded_tokens = pad_sequence(token_sequences, prepared_sequences)
        prediction_index = model.predict_classes(padded_tokens)

        for token, index in word_indices.items():
            if index == prediction_index:
                predicted = token
                seed += f' {token}'
                break
    return seed

def save_model(model, path):
    model.save(path)

def load_model(path):
    return tf.keras.models.load_model(path)


if __name__ == '__main__':

    # STEP 1: Read in data
    raw_text = load_data(DATA_PATH)

    # STEP 2: Remove white space, headers, and undesired punctuation
    cleaned = parse_data_word(raw_text)

    # STEP 3: Create tokenized index based on word frequency
    word_indices = create_token_index(cleaned)

    # STEP 4: Create tokenized index based on word frequency
    tokenized_sequences = create_tokenized_sequences(cleaned, word_indices)

    # STEP 5: Pad sequences to ensure uniform input size for LSTM
    padded_sequences = pad_sequences(tokenized_sequences)

    # STEP 6: Extract input sequences and output labels (i.e. next words) from padded_sequences
    prepared_sequences, next_words = get_sequences_labels(padded_sequences, word_indices)

    # STEP 7: Train LSTM
    word_model = create_LSTM_word(word_indices=word_indices,
                     input_sequences=prepared_sequences,
                     output_labels=next_words
                     )

    # STEP 8: Save model
    save_model(word_model, WORD_MODEL_PATH)

    # STEP 9: Load LSTM
    model = load_model(WORD_MODEL_PATH)

    # STEP 10: Predict next words!
    seed = 'oh how dark was'
    predict_text(model, seed, 10, word_indices, prepared_sequences)