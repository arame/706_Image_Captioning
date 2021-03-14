import torch as T

class Hyper:
    total_epochs = 100
    learning_rate = 1e-6
    embed_size = 256
    hidden_size = 256
    batch_size = 2
    dropout_rate = 0.5

    [staticmethod]   
    def display():
        print("The Hyperparameters")
        print("-------------------")
        print(f"NUmber of epochs = {Hyper.total_epochs}")
        print(f"learning rate = {Hyper.learning_rate}")
        print(f"batch_size = {Hyper.batch_size}")

class Constants:
    device = T.device("cuda" if T.cuda.is_available() else "cpu")
    PAD = "<PAD>"
    SOS = "<SOS>"   # Start Of Sentence
    EOS = "<EOS>"   # End Of Sentence
    UNK = "<UNK>"   # Unknown word
    load_model = False
    save_model = True
    vocab_from_file = True
    vocab_size = 1000
    num_layers = 1
    train_CNN = False
    max_length = 50
    word_threshold = 8
    data_folder = "../mscoco"
    vocab_file = "../mscoco/vocab.pkl"
    data_folder_ann = "../mscoco/annotations"
    images_train_file = "instances_train2017.json"
    images_val_file = "instances_val2017.json"
    captions_train_file = "captions_train2017.json"
    captions_val_file = "captions_val2017.json"