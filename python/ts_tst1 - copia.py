import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
import tensorflow as tf
import ast
import os
import json
import matplotlib.pyplot as plt
from nltk import tokenize
import seaborn as sns
import binascii


dataset_pd = pd.read_csv("C:\\Users\\crist\\Documents\\AES tests\\test3_output.csv")
dataset_pd["ENCRYPTED"] = dataset_pd["ENCRYPTED"].apply(lambda t: binascii.unhexlify(t))


# Removing all pseudonyms that wrote in English.

texts = texts[~texts['author'].isin(['Alexander Search', 'David Merrick', 'Charles Robert Anon', 'I. I. Crosse'])]

texts['text'] = texts['text'].apply(lambda t: ast.literal_eval(t))
texts = texts.reset_index().drop('index', axis=1)
texts = texts['text'].tolist()
texts = np.concatenate(texts)

texts = np.asarray(texts)
texts_p = " ".join(texts)

# we will be truncating large texts soon, so this code only tries to reduce the 
# sequence size by splitting the texts that seem to be significantly larger than 
# the rest. Otherwise, we try to use the structure provided in the data itself

_, ax = plt.subplots(1, 2, figsize=(15, 5))

mylen = np.vectorize(len)

sns.histplot(mylen(texts), bins=50, ax=ax[0])
ax[0].set_title('Histogram of the number of characters in each \nchunk of text BEFORE splitting sentences', fontsize=16)

large_texts = texts[mylen(texts)>350]
large_texts_p = " ".join(large_texts)
large_texts = tokenize.sent_tokenize(large_texts_p)

texts = np.concatenate((texts[~(mylen(texts)>350)], large_texts))

ax[1].set_title('Histogram of the number of characters in each \nchunk of text AFTER splitting sentences', fontsize=16)
sns.histplot(mylen(texts), bins=50, ax=ax[1])

print(f'Length of texts dataset: {len(texts_p)} characters')



vocab = sorted(set(texts_p))
print(f'{len(vocab)} unique characters in texts')















def create_character_tokenizer(list_of_strings):
    tokenizer = Tokenizer(filters=None,
                         char_level=True, 
                          split=None,
                         lower=False)
    tokenizer.fit_on_texts(list_of_strings)
    return tokenizer

tokenizer = create_character_tokenizer(texts)

tokenizer_config = tokenizer.get_config()

word_counts = json.loads(tokenizer_config['word_counts'])
index_word = json.loads(tokenizer_config['index_word'])
word_index = json.loads(tokenizer_config['word_index'])

def strings_to_sequences(tokenizer, list_of_strings):
    sentence_seq = tokenizer.texts_to_sequences(list_of_strings)
    return sentence_seq

seq_texts = strings_to_sequences(tokenizer, texts)

mylen = np.vectorize(len)

print(max(mylen(texts)))
print(np.round(np.mean(mylen(texts))))



def make_padded_dataset(sequences):
    padded_sequence = tf.keras.preprocessing.sequence.pad_sequences(sequences,
                                                 maxlen=300,
                                                 padding='pre',
                                                 truncating='pre',
                                                 value=0)
    return padded_sequence

padded_sequences = make_padded_dataset(seq_texts)


def create_inputs_and_targets(array_of_sequences, batch_size=32):
    input_seq = array_of_sequences[:,:-1]
    target_seq = array_of_sequences[:,1:]
    
    # Prepare the batches and ensure that is ready to be fed to a stateful RNN
    
    num_examples = input_seq.shape[0]

    num_processed_examples = num_examples - (num_examples % batch_size)

    input_seq = input_seq[:num_processed_examples]
    target_seq = target_seq[:num_processed_examples]

    steps = int(num_processed_examples / 32) 

    inx = np.empty((0,), dtype=np.int32)
    for i in range(steps):
        inx = np.concatenate((inx, i + np.arange(0, num_processed_examples, steps)))

    input_seq_stateful = input_seq[inx]
    target_seq_stateful = target_seq[inx]
    
    # Split data between training and validation sets
    
    num_train_examples = int(batch_size * ((0.8 * num_processed_examples) // batch_size))

    input_train = input_seq_stateful[:num_train_examples]
    target_train = target_seq_stateful[:num_train_examples]

    input_valid = input_seq_stateful[num_train_examples:]
    target_valid = target_seq_stateful[num_train_examples:]
    
    # Create datasets objects for training and validation data
    
    dataset_train = tf.data.Dataset.from_tensor_slices((input_train, target_train))
    dataset_train = dataset_train.batch(batch_size, drop_remainder=True)

    dataset_valid = tf.data.Dataset.from_tensor_slices((input_valid, target_valid))
    dataset_valid = dataset_valid.batch(batch_size, drop_remainder=True)
    
    return (dataset_train, dataset_valid)
    

train_data, valid_data = create_inputs_and_targets(padded_sequences)