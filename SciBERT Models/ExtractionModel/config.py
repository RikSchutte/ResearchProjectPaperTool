import transformers

MAX_LEN = 128
TRAIN_BATCH_SIZE = 32
VALID_BATCH_SIZE = 8
EPOCHS = 10
BASE_MODEL_PATH = 'allenai/scibert_scivocab_uncased'
MODEL_PATH = "model.bin"
# TRAINING_FILE = "C:/Users/20202176/Downloads/ner_dataset.csv/ner_dataset.csv"
TRAINING_FILE = "../Projectmap/Data/DataVanNWaardes.xlsx"
TOKENIZER = transformers.BertTokenizer.from_pretrained(
    BASE_MODEL_PATH,
    do_lower_case=True
)