
# User configuration #
DATASET_NAME = 'initial'
LABELS = ('avocado', 'banana')  # ('avocado', 'coconut', 'apple', 'banana', 'blueberry')
WAV_FILES_PER_LABEL = 1
FILES_EXTENSION = 'wav'
DEBUG_MODE = False
PRINT_PREDICTIONS = True

# Other configuration #
DATASET_PATH = 'audios/datasets/' + DATASET_NAME
TRAIN_PATH = DATASET_PATH + '/train'
VALIDATE_PATH = DATASET_PATH + '/validate'

