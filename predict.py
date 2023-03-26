

## NOTE: In order to predict based on different models, please edit the src/config.py file


from src.word_model import *
import src.config as config

import sys
import logging
import warnings
warnings.filterwarnings("ignore")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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

def predict(input_seed, output_length):
    # STEP 1: Read in data
    raw_text = load_data(DATA_PATH)
    logger.info('Text loaded.')

    # STEP 2: Remove white space, headers, and undesired punctuation
    cleaned = parse_data_word(raw_text)
    logger.info('Text cleaned.')

    # STEP 3/4: Create tokenized index based on word frequency
    word_indices = create_token_index(cleaned)
    tokenized_sequences = create_tokenized_sequences(cleaned, word_indices)
    logger.info('Words tokenized.')

    # STEP 5: Pad sequences to ensure uniform input size for LSTM
    padded_sequences = pad_sequences(tokenized_sequences)
    logger.info('Sequences padded.')

    # STEP 6: Extract input sequences and output labels (i.e. next words) from padded_sequences
    prepared_sequences, next_words = get_sequences_labels(padded_sequences, word_indices)
    logger.info('Sequences prepared.')

    # STEP 7: Load LSTM
    model = load_model(WORD_MODEL_PATH)
    logger.info('Model loaded.')

    # STEP 8: Predict next words!
    logger.info(f'Forming prediction of length {output_length} using seed: {input_seed}')
    prediction = predict_text(model, seed=input_seed, output_length=output_length, word_indices=word_indices,
                              prepared_sequences=prepared_sequences)

    return prediction

if __name__ == '__main__':
    input_seed = str(sys.argv[1]).lower()
    output_length = int(sys.argv[2])

    prediction = predict(input_seed=input_seed, output_length=output_length)
    print(prediction)