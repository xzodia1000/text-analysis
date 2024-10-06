# %%

import re
import string
import contractions
from gensim.models import Word2Vec
import warnings
import pandas as pd
import numpy as np
from tqdm import tqdm

import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk import pos_tag
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer

import pickle

from sklearn.utils.class_weight import compute_class_weight

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
import tensorflow as tf

#!
# from gensim.models import Word2Vec
# import pandas as pd
# import numpy as np

# from tqdm import tqdm

tqdm.pandas()

# from sklearn.utils.class_weight import compute_class_weight

# from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional, SpatialDropout1D
# from tensorflow.keras.regularizers import l2
# from tensorflow.keras.optimizers import Adam
# from tensorflow.keras.utils import to_categorical

# import pickle

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


train = pd.read_csv('data/train.csv')
train = train[['Review', 'overall']]
train = train.dropna(subset=['Review'])

class TextPreprocessor:
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        self.punctuations = str.maketrans(string.punctuation, ' ' * len(string.punctuation))
        # self.spell = SpellChecker()
 
    def lowercase(self, text):
        return text.lower()
   
    def remove_punctuation(self, text):
        return text.translate(self.punctuations)
   
    def tokenize(self, text):
        return word_tokenize(text)
 
    def remove_stop_words(self, tokens):
        stop_words = set(stopwords.words("english"))
        return [word for word in tokens if word not in stop_words]
   
    def remove_numbers(self, tokens):
        return [word for word in tokens if not word.isdigit()]
   
    def POS_tagging(self, tokens):
        return pos_tag(tokens)
   
    #Function to convert nltk pos tags to wordnet pos tags
    def get_wordnet_pos(self, treebank_tag):
        try:
            if treebank_tag.startswith("J"):
                return wordnet.ADJ
            elif treebank_tag.startswith("V"):
                return wordnet.VERB
            elif treebank_tag.startswith("N"):
                return wordnet.NOUN
            elif treebank_tag.startswith("R"):
                return wordnet.ADV
            elif treebank_tag.startswith("P"):
                return wordnet.NOUN
            else:
                return ""
        except:
            print(treebank_tag, f" is not a valid treebank tag")
            return ""
   
    def lemmatize(self, pos_tags):
        lemmatized_tokens = []
        for token, pos_tag in pos_tags:
            wordnet_pos = self.get_wordnet_pos(pos_tag)
            lemma = (
                self.lemmatizer.lemmatize(token, pos=wordnet_pos)
                if wordnet_pos
                else self.lemmatizer.lemmatize(token)
            )
            lemmatized_tokens.append(lemma)
        return lemmatized_tokens
   
    #* function to expand contractions like "I'm" to "I am"
    def expand_contractions(self, text):
        return contractions.fix(text)
   
    #* function to remove words starting with a backslash like "\n" using regex
    def rem_backslash_words(self, text):
        # Replace HTML line breaks with space
        text = re.sub(r'<br\s*/?>', ' ', text)
        # Replace tabs and newlines with a space
        text = re.sub(r'\s+', ' ', text)
        # Optionally, strip leading/trailing whitespace
        text = text.strip()
        return text
 
 
    def preprocess(self, text):
        lower_text = self.lowercase(text)
        expanded_text = self.expand_contractions(lower_text)
        no_punc_text = self.remove_punctuation(expanded_text)
        no_backslash = self.rem_backslash_words(no_punc_text)
        # spell_checked = self.correct_spelling(no_backslash)
        tokens = self.tokenize(no_backslash)
        no_stop = self.remove_stop_words(tokens)
        no_numbers = self.remove_numbers(no_stop)
        pos_tags = self.POS_tagging(no_numbers)
        lemmatized = self.lemmatize(pos_tags)
        return lemmatized
   
pre_processor = TextPreprocessor()
#!---------------------------------------------------------------------------------------
print("starting preprocessing")
 
# train["tokenized"] = train["Review"].progress_apply(lambda x: pre_processor.preprocess(x))
 
# processed_train = train[["tokenized", "overall"]]
# processed_train.loc[:, "tokenized_joined"] = processed_train["tokenized"].progress_apply(lambda x: " ".join(x))
 
# processed_train = processed_train[["tokenized", "tokenized_joined", "overall"]]
 
processed_train = pd.read_pickle("processed_train.pkl")

print("preprocessing done")

# # save the processed_train in a pickle
# with open("processed_train.pkl", "wb") as f:
#     pickle.dump(processed_train, f)


# # %%

print("starting cbow")
tokenized_texts = processed_train['tokenized'].tolist()

cbow = Word2Vec(sentences=tokenized_texts, vector_size=100, window=5, sg=0, min_count=5)
print("cbow done")

# load the skipgram.model
# cbow = Word2Vec.load("cbow.model")

vocab_size = len(cbow.wv.key_to_index) + 1  # Adding 1 to account for padding token
embedding_dim = 100  # Or the vector_size you used for your CBOW model

embedding_matrix = np.zeros((vocab_size, embedding_dim))
for word, i in cbow.wv.key_to_index.items():
    embedding_vector = cbow.wv[word]
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector

# load the embedding matrix pickle
# with open("embedding_matrix.pkl", "rb") as f:
#     embedding_matrix = pickle.load(f)

# vocab_size = embedding_matrix.shape[0]
# %%
# Initialize and fit the tokenizer

tokenizer = Tokenizer(num_words=vocab_size-1)  # vocab_size minus 1 to account for padding token
tokenizer.fit_on_texts(processed_train['tokenized_joined'].tolist())

# Convert texts to sequences of integers
sequences = tokenizer.texts_to_sequences(processed_train['tokenized_joined'].tolist())

# Pad sequences to ensure uniform length
max_length = max(len(seq) for seq in sequences)  # Or choose a max_length that suits your dataset
padded_sequences = pad_sequences(sequences, maxlen=max_length, padding='post')

# %%
model = Sequential()
model.add(Embedding(input_dim=vocab_size, output_dim=embedding_dim, embeddings_initializer=tf.keras.initializers.Constant(embedding_matrix), trainable=False, input_length=max_length))
model.add(LSTM(512, return_sequences=True))
model.add(SpatialDropout1D(0.2))
model.add(LSTM(256, return_sequences=True))
model.add(SpatialDropout1D(0.2))
model.add(LSTM(128))

model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(5, activation='softmax'))  # For 5 classes

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.summary()

y = processed_train['overall'] - 1

classes = np.unique(y)
class_weights = compute_class_weight(class_weight='balanced', classes=classes, y=y)
class_weights_dict = {class_label: weight for class_label, weight in zip(classes, class_weights)}


history = model.fit(padded_sequences, y, epochs=20, batch_size=32, validation_split=0.2, verbose=1)

# save the model and hist in a pickle
with open("model_LSTM.pkl", "wb") as f:
    pickle.dump(model, f)

with open("history_lstm.pkl", "wb") as f:
    pickle.dump(history, f)
