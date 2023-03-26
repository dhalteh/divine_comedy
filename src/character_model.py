import numpy as np
import os
import tensorflow as tf
from keras.utils import np_utils
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import LSTM
import src.config as config
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


#########CONFIGS#########
DATA_PATH = config.DATA_PATH

# LSTM general parameters
ACTIVATION=config.ACTIVATION
LOSS=config.LOSS
OPTIMIZER=config.OPTIMIZER

# LSTM character-level parameters
MAX_LENGTH = config.MAX_LENGTH
STEP = config.STEP #How many characters to skip until next sentence starts
CHAR_MEMORY_UNITS=config.CHAR_MEMORY_UNITS
CHAR_NUM_EPOCHS=config.CHAR_NUM_EPOCHS
CHAR_BATCH_SIZE=config.CHAR_BATCH_SIZE
CHAR_DROPOUT=config.CHAR_DROPOUT
CHAR_MODEL_PATH=config.CHAR_MODEL_PATH


def load_data(path):
    if os.path.isfile(path):
        with open(path) as infile:
            text = infile.readlines()
        return text
    else:
        raise FileNotFoundError('File not found!')
        return None


def parse_data_char(raw):
    cleaned_doc = []
    for line in raw:
        cleaned_line = line.strip('\(\)-,.!&$@%*^\n').lower().strip()
        if ':' not in cleaned_line:
            cleaned_doc.append(cleaned_line)
    # Remove all empty strings
    cleaned_doc = list(filter(None, cleaned_doc))
    return cleaned_doc


def make_corpus(cleaned_lines):
    joined = '\n'.join(line for line in cleaned_lines)
    return joined


def create_vocabulary(text):
    unique_characters = [i for i in set(text)]
    sorted_characters = sorted(unique_characters)
    character_index_dict = dict((character, index) for index, character in enumerate(sorted_characters))
    index_character_dict = dict((index, character) for character, index in character_index_dict.items())

    return character_index_dict, index_character_dict


def get_sequence_label_pairs(text, max_len=MAX_LENGTH, step_size=STEP):
    sequence_list = []
    true_labels = []

    text_length = len(text)
    remainder = text_length - max_len

    for start_index in range(0, remainder, step_size):
        end_index = start_index + max_len
        sequence = text[start_index: end_index]
        true_label = text[end_index]

        sequence_list.append(sequence)
        true_labels.append(true_label)

    return sequence_list, true_labels


def vectorize_sequence_label_pairs(sequences, labels, char_ind_dict):
    X_list = []
    Y_list = []

    for index, sequence in enumerate(sequences):
        sequence_indices = [char_ind_dict[character] for character in sequence]
        true_label_index = char_ind_dict[labels[index]]
        X_list.append(sequence_indices)
        Y_list.append(true_label_index)
    return X_list, Y_list


def prepare_input_ouput(X_list, Y_list, char_ind_dict, sequences, num_features=1, max_len=MAX_LENGTH):
    vocab_length = len(char_ind_dict)
    sequence_length = len(sequences)

    # Need to reshape X_list in the form of: (samples, timesteps, one. char at a time)
    X = np.reshape(X_list, (sequence_length, max_len, num_features))

    # Normalize X_list
    X = X / (float(vocab_length))

    # One-hot encode next character labels
    Y = np_utils.to_categorical(Y_list)
    print(X.shape, Y.shape)
    return X, Y


def create_LSTM_char(features, labels, n_memory_units, epochs,
                       batch_size, activation='softmax', dropout=0.2):
    # Extract LSTM model dimensions
    sequence_length = features.shape[1]
    feature_length = features.shape[2]
    output_length = labels.shape[1]

    # Initialize model with one layer and dropout
    model = Sequential()
    model.add(LSTM(n_memory_units, input_shape=(sequence_length, feature_length)))
    model.add(Dropout(dropout))

    # Add output layer with softmax activations function
    model.add(Dense(output_length, activation=activation))
    model.compile(loss='categorical_crossentropy', optimizer='adam')

    # Fit the model
    model.fit(features, labels, epochs=epochs, batch_size=batch_size)

    return model


def character_level_generator(model, X_list, CI_dict, IC_dict, text_length, start_seed=None):
    vocab_length = len(CI_dict)
    data_length = len(X_list)

    if start_seed is None:
        start_seed = np.random.randint(0, data_length - 1)
    print(start_seed)

    seed_indices = X_list[start_seed].copy()
    generated_indices = [i for i in seed_indices]

    for i in range(text_length):
        X_sample = np.reshape(seed_indices, (1, len(seed_indices), 1))
        X_sample = X_sample / (float(vocab_length))
        prediction_values = model.predict(X_sample)
        character_index = np.argmax(prediction_values)
        seed_indices.append(character_index)
        generated_indices.append(character_index)
        seed_indices = seed_indices[1:len(seed_indices)]

    print(''.join([IC_dict[index] for index in generated_indices]))


def save_model(model, path):
    model.save(path)


def load_model(path):
    return tf.keras.models.load_model(path)

if __name__ == '__main__':
    # STEP 1: Read in data
    raw_text = load_data(DATA_PATH)

    # STEP 2: Remove white space, headers, and undesired punctuation
    cleaned = parse_data_char(raw_text)

    # STEP 3: Make Corpus
    joined_text = make_corpus(cleaned)

    # STEP 4: Create vocabulary
    CI_dict, IC_dict = create_vocabulary(joined_text)

    # STEP 5: Create sequences and truth labels (i.e. next characters after the sequence)
    sequences, labels = get_sequence_label_pairs(joined_text, max_len=MAX_LENGTH, step_size=STEP)

    # STEP 6: Vectorize sequence and label data
    X_list, Y_list = vectorize_sequence_label_pairs(sequences, labels, CI_dict, IC_dict, max_len=MAX_LENGTH)

    # STEP 7: Prepare sequences and labels for LSTM
    X, Y = prepare_input_ouput(X_list, Y_list, CI_dict, sequences=sequences, num_features=1, max_len=MAX_LENGTH)

    # STEP 8: Fit initial model
    char_model = create_LSTM_char(features=X,
                                labels=Y,
                                n_memory_units=CHAR_MEMORY_UNITS,
                                epochs=CHAR_NUM_EPOCHS,
                                batch_size=CHAR_BATCH_SIZE,
                                dropout=CHAR_DROPOUT)

    # STEP 9: Save model
    save_model(char_model, CHAR_MODEL_PATH)

    # STEP 10: Load model
    char_model = load_model(CHAR_MODEL_PATH)

    # STEP 11: Predict character output based on text seed
    character_level_generator(char_model, X_list, CI_dict, IC_dict, text_length=40, start_seed=None)