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


dataset_pd = pd.read_csv("C:\\Users\\crist\\Documents\\AES tests\\output.csv")

dataset_pd["ENCRYPTED"] = dataset_pd["ENCRYPTED"].apply(lambda t: [i for i in binascii.unhexlify(t)])

def create_character_tokenizer(list_of_strings):
    tokenizer = Tokenizer(filters=None,
                         char_level=True, 
                          split=None,
                         lower=False)
    tokenizer.fit_on_texts(list_of_strings)
    return tokenizer


tokenizer = create_character_tokenizer(dataset_pd["TEXT"])

tokenizer_config = tokenizer.get_config()

word_counts = json.loads(tokenizer_config['word_counts'])
index_word = json.loads(tokenizer_config['index_word'])
word_index = json.loads(tokenizer_config['word_index'])

def strings_to_sequences(tokenizer, list_of_strings):
    sentence_seq = tokenizer.texts_to_sequences(list_of_strings)
    return sentence_seq


seq_texts = strings_to_sequences(tokenizer, dataset_pd["TEXT"])
dataset_pd["TEXT"] = seq_texts


x = dataset_pd["TEXT"]
x = [np.asarray(i) for i in x]

y = dataset_pd["ENCRYPTED"]
y = [np.asarray(i) for i in y]

test_pct = .2
batch_size = 64
buffer_size = 10000
embedding_dim = 256
epochs = 50
seq_length = 200
rnn_units = 1024

x_test = x[:int(len(x)*test_pct)]
y_test = y[:int(len(y)*test_pct)]

x_train = x[int(len(x)*test_pct):]
y_train = y[int(len(y)*test_pct):]

tst_full = tf.data.Dataset.from_tensor_slices((x, y))

dataset_train = tf.data.Dataset.from_tensor_slices((x_train,y_train))
train_data = dataset_train.batch(batch_size, drop_remainder=True)


train_data = tf.data.Dataset.from_tensor_slices((x_train,y_train))
valid_data = tf.data.Dataset.from_tensor_slices((x_test,y_test))


model = tf.keras.Sequential()
model.add(tf.keras.layers.Embedding(input_dim=vocab_size, output_dim = 256, mask_zero=True, batch_input_shape=(batch_size, None)))
model.add(tf.keras.layers.LSTM(units=512, return_sequences=True,stateful=True))
model.add(tf.keras.layers.Dropout(0.2))
model.add(tf.keras.layers.LSTM(units=512, return_sequences=True,stateful=True))
model.add(tf.keras.layers.Dense(256))
model.add(tf.keras.layers.Dense(256))
model.add(tf.keras.layers.Dense(256))
model.add(tf.keras.layers.Dense(256))

checkpoint_callback=tf.keras.callbacks.ModelCheckpoint(filepath='.\\models\\ckpt', save_weights_only=True, save_best_only=True)
model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['sparse_categorical_accuracy'])

history = model.fit(x_train, y_train, epochs=30, callbacks=[checkpoint_callback, tf.keras.callbacks.EarlyStopping(patience=2)])


def model_history(history):
    history_dict = dict()
    for k, v in history.history.items():
        history_dict[k] = [float(val) for val in history.history[k]]
    return history_dict


history_dict = model_history(history)

def plot_history(history_dict):
    plt.figure(figsize=(15,5))
    plt.subplot(121)
    plt.plot(history_dict['sparse_categorical_accuracy'])
    plt.title('Accuracy vs. epochs')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.xticks(np.arange(len(history_dict['sparse_categorical_accuracy'])))
    ax = plt.gca()
    ax.set_xticklabels(1 + np.arange(len(history_dict['sparse_categorical_accuracy'])))
    plt.legend(['Training', 'Validation'], loc='lower right')
    plt.subplot(122)
    plt.plot(history_dict['loss'])
    plt.title('Loss vs. epochs')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.xticks(np.arange(len(history_dict['sparse_categorical_accuracy'])))
    ax = plt.gca()
    ax.set_xticklabels(1 + np.arange(len(history_dict['sparse_categorical_accuracy'])))
    plt.legend(['Training', 'Validation'], loc='upper right')
    plt.show() 
    

plot_history(history_dict)
