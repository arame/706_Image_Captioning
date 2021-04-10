import torch as T

class Hyper:
    total_epochs = 2
    learning_rate = 1e-6
    embed_size = 256
    hidden_size = 256
    num_layers = 1
    batch_size = 1
    dropout_rate = 0.5
    is_grayscale = True
    print_freq = 100  # print training/validation stats every __ batches
    #selected_category_names = ['bicycle']
    #selected_category_names = ['horse']
    #selected_category_names = ["train"]
    selected_category_names = ["person"]
    # See the file in the references folder for a list of categories used in Coco. Not all categories have images.

    [staticmethod]   
    def display():
        print("The Hyperparameters")
        print("-------------------")
        categories = ", ".join(Hyper.selected_category_names)
        print(f"Selected category = {categories}")

        print(f"Number of epochs = {Hyper.total_epochs}")
        print(f"Learning rate = {Hyper.learning_rate}")
        print(f"Batch_size = {Hyper.batch_size}")
        print(f"Is grayscale = {Hyper.is_grayscale}")

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
    instances_train_file = "instances_train2017.json"
    instances_val_file = "instances_val2017.json"