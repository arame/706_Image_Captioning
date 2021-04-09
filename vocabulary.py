import torch as T
from config import Hyper, Constants
import spacy
import os
import pickle

# For this line to work, download with the following command in admin mode: 
# python3 -m spacy download en_core_web_sm
# 
spacy_eng = spacy.load("en_core_web_sm")
class Vocabulary:
    def __init__(self, word_threshold):
        # itos = index to string, stoi = string to index
        self.itos = {0:Constants.PAD, 1:Constants.SOS, 2:Constants.EOS, 3:Constants.UNK} 
        self.stoi = {v: k for k, v in self.itos.items()}  
        self.word_threshold = word_threshold

    def __len__(self):
        return len(self.itos)

    def get_vocab(self):
        # There are a lot of captions, so store the vocabulary on a file
        # and use that rather than rebuild every time
        # The user has the option to rebuild the vocabulary and overwrite the file
        # if desired

        with open(Constants.vocab_file, 'rb') as f:
            vocab = pickle.load(f)
            self.itos = vocab.itos
            self.stoi = vocab.stoi
        print('Vocabulary successfully loaded from the vocab.pkl file')

    @staticmethod
    def tokenizer_eng(text):
        output = [tok.text.lower() for tok in spacy_eng.tokenizer(text)]
        return output
        # "I want freedom" --> ["i", "want", "freedom"]

    def build_vocabulary(self, sentence_list):
        frequencies = {}
        idx = len(self.itos)    # idx is set to 4
        i = 0
        no_captions = len(sentence_list)
        for sentence in sentence_list:
            i += 1
            if i % 100000 == 0:
                print(f"[{i}/{no_captions}] Tokenizing captions...")

            for word in self.tokenizer_eng(sentence):
                if word not in frequencies:
                    frequencies[word] = 1
                else:
                    frequencies[word] += 1
            
                if frequencies[word] == self.word_threshold:
                    self.stoi[word] = idx
                    self.itos[idx] = word
                    idx += 1

        # All the tokens are loaded from the captions, now save to the vocab.pkl file
        with open(Constants.vocab_file, 'wb') as f:
            pickle.dump(self, f)

        print(f"{len(self.stoi)} tokens saved to vocab.pkl file from {no_captions} captions")

    def numericalize(self, text):
        tokenized_text = self.tokenizer_eng(text)
        output = [
            self.stoi[token] if token in self.stoi else self.stoi[Constants.UNK]
            for token in tokenized_text
        ]
        return output

if __name__ == "__main__":
    v = Vocabulary(8)
    v.build_vocabulary(["The cat sat on the mat", "so long and thanks for all the fish", "The cat sat on the mat", "so long and thanks for all the fish", "The cat sat on the mat", "so long and thanks for all the fish", "The cat sat on the mat", "so long and thanks for all the fish", "The cat sat on the mat", "so long and thanks for all the fish", "The cat sat on the mat", "so long and thanks for all the fish", "The cat sat on the mat", "so long and thanks for all the fish", "The cat sat on the mat", "so long and thanks for all the fish", "The cat sat on the mat", "so long and thanks for all the fish", "The cat sat on the mat", "so long and thanks for all the fish", "The cat sat on the mat", "so long and thanks for all the fish"])
    output = v.numericalize("The cat likes the long fish")
    print(output)
