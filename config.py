import torch as T

class Hyper:
    total_epochs = 1
    learning_rate = 1e-6
    embed_size = 256
    hidden_size = 256
    num_layers = 1
    batch_size = 1
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
    backup_model_folder = "../backup"
    backup_model_path = "../backup/model.pth"
    vocab_from_file = True
    
    train_CNN = False
    max_length = 50
    word_threshold = 8
    vocab_file = "../mscoco/vocab.pkl"
    data_folder_ann = "../mscoco/annotations"
    captions_train_file = "captions_train2017.json"
    captions_val_file = "captions_val2017.json"