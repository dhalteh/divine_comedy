
import os

#########CONFIGS#########
ROOT_PATH = os.path.abspath(os.curdir)
DATA_LOCATION = 'data'
DATA_NAME = 'dante.txt'
DATA_PATH = os.path.join(ROOT_PATH, DATA_LOCATION, DATA_NAME)

# LSTM general parameters
ACTIVATION='softmax'
LOSS='categorical_crossentropy'
OPTIMIZER='adam'

# LSTM word-level parameters
WORD_MEMORY_UNITS=128
WORD_NUM_EPOCHS=50
WORD_OUTPUT_DIMENSION=10
WORD_DROPOUT=0.1
WORD_DROPOUT_KEY = str(WORD_DROPOUT).split('.')[1]
WORD_MODEL_NAME = f'word_model_{WORD_MEMORY_UNITS}_{WORD_DROPOUT_KEY}_{WORD_NUM_EPOCHS}'
MODEL_LOCATION = 'models'
WORD_MODEL_PATH = os.path.join(ROOT_PATH, MODEL_LOCATION, WORD_MODEL_NAME)

# LSTM character-level parameters
MAX_LENGTH = 50
STEP = 1 # How many characters to skip until next sentence starts
CHAR_MEMORY_UNITS=256
CHAR_NUM_EPOCHS=10
CHAR_BATCH_SIZE=128
CHAR_DROPOUT=0.2
CHAR_DROPOUT_KEY = str(CHAR_DROPOUT).split('.')[1]
CHAR_MODEL_NAME = f'char_model_{CHAR_MEMORY_UNITS}_{CHAR_DROPOUT_KEY}_{CHAR_NUM_EPOCHS}'
CHAR_MODEL_PATH=os.path.join(ROOT_PATH, MODEL_LOCATION, CHAR_MODEL_NAME)
