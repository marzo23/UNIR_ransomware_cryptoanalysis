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


dataset_pd = pd.read_csv("C:\\Users\\crist\\Documents\\AES tests\\tstnew1_output.csv")

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

#dataset_test = tf.data.Dataset.from_tensor_slices((x_test,y_test))
#valid_data = dataset_test.batch(batch_size, drop_remainder=True)

train_data = tf.data.Dataset.from_tensor_slices((x_train,y_train))
valid_data = tf.data.Dataset.from_tensor_slices((x_test,y_test))



model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(256))
model.add(tf.keras.layers.Dense(256))
model.add(tf.keras.layers.Dense(256))
model.add(tf.keras.layers.Dense(256))

model.compile(optimizer='sgd', loss='mse')
history = model.fit(x, y, batch_size=32, epochs=1) #no jala, no hace nada

#https://towardsdatascience.com/generating-text-with-recurrent-neural-networks-based-on-the-work-of-f-pessoa-1e804d88692d
#https://www.tensorflow.org/text/tutorials/text_generation
#https://towardsdatascience.com/generating-text-with-tensorflow-2-0-6a65c7bdc568
#https://towardsdatascience.com/sequence-to-sequence-models-from-rnn-to-transformers-e24097069639





##################################################################################


#NO CORRE:


def get_model(vocab_size, batch_size):
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(input_dim=vocab_size, output_dim = 256, mask_zero=True, batch_input_shape=(batch_size, None)),
        tf.keras.layers.LSTM(units=512, return_sequences=True,stateful=True),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.LSTM(units=512, return_sequences=True,stateful=True),
        tf.keras.layers.Dense(units=vocab_size)
    ])
    return model


model = get_model(len(tokenizer.word_index) + 1, batch_size)


checkpoint_callback=tf.keras.callbacks.ModelCheckpoint(filepath='.\\models\\ckpt',
                                                       save_weights_only=True,
                                                       save_best_only=True)
model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['sparse_categorical_accuracy'])

history = model.fit(train_data, 
                    epochs=30, 
                    validation_data=valid_data,
                    callbacks=[checkpoint_callback, 
                    tf.keras.callbacks.EarlyStopping(patience=2)])




















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
    plt.plot(history_dict['val_sparse_categorical_accuracy'])
    plt.title('Accuracy vs. epochs')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.xticks(np.arange(len(history_dict['sparse_categorical_accuracy'])))
    ax = plt.gca()
    ax.set_xticklabels(1 + np.arange(len(history_dict['sparse_categorical_accuracy'])))
    plt.legend(['Training', 'Validation'], loc='lower right')

    plt.subplot(122)
    plt.plot(history_dict['loss'])
    plt.plot(history_dict['val_loss'])
    plt.title('Loss vs. epochs')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.xticks(np.arange(len(history_dict['sparse_categorical_accuracy'])))
    ax = plt.gca()
    ax.set_xticklabels(1 + np.arange(len(history_dict['sparse_categorical_accuracy'])))
    plt.legend(['Training', 'Validation'], loc='upper right')
    plt.show() 
    
plot_history(history_dict)










def get_model(vocab_size, batch_size):
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(input_dim=vocab_size, output_dim = 256, mask_zero=True, batch_input_shape=(batch_size, None)),
        tf.keras.layers.LSTM(units=512, return_sequences=True,stateful=True),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.LSTM(units=512, return_sequences=True,stateful=True),
        tf.keras.layers.Dense(units=vocab_size)
    ])
    return model


model = get_model(len(tokenizer.word_index) + 1, batch_size)
model.summary()






def build_model(vocab_size, embedding_dim, rnn_units, batch_size):
    model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, embedding_dim, batch_input_shape=[batch_size, None]),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.LSTM(rnn_units,
        return_sequences=True,
        stateful=True,
        recurrent_initializer='glorot_uniform'),
    tf.keras.layers.Dropout(0.2), 
    tf.keras.layers.LSTM(rnn_units,
        return_sequences=True,
        stateful=True,
        recurrent_initializer='glorot_uniform'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(vocab_size)
    ])
    return model


model = build_model(
    vocab_size = len(tokenizer.word_index) + 1,
    embedding_dim=embedding_dim,
    rnn_units=rnn_units,
    batch_size=batch_size)






model = tf.keras.Sequential()
#model.add(tf.keras.layers.Dense(8, input_shape=(64,)))
model.add(tf.keras.layers.Dense(256))
model.add(tf.keras.layers.Dense(256))
model.add(tf.keras.layers.Dense(256))
model.add(tf.keras.layers.Dense(256))
#model.add(tf.keras.layers.Dense(64, input_shape=(64,)))




def get_model(vocab_size, batch_size):
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(input_dim=vocab_size, output_dim = 256, mask_zero=True, batch_input_shape=(batch_size, None)),
        tf.keras.layers.LSTM(units=512, return_sequences=True,stateful=True),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.LSTM(units=512, return_sequences=True,stateful=True),
        tf.keras.layers.Dense(units=vocab_size)
    ])
    return model


model = get_model(len(tokenizer.word_index) + 1, batch_size)




model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(256))
model.add(tf.keras.layers.Dense(256))
model.add(tf.keras.layers.Dense(256))
model.add(tf.keras.layers.Dense(256))

checkpoint_callback=tf.keras.callbacks.ModelCheckpoint(filepath='.\\models\\ckpt',
                                                       save_weights_only=True,
                                                       save_best_only=True)
model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['sparse_categorical_accuracy'])

history = model.fit(tst_full, epochs=30, callbacks=[checkpoint_callback, tf.keras.callbacks.EarlyStopping(patience=2)])






from random import randint
from numpy import array
from numpy import argmax
from numpy import array_equal
from keras.utils import to_categorical
from keras.models import Model
from keras.layers import Input
from keras.layers import LSTM
from keras.layers import Dense
 
# generate a sequence of random integers
def generate_sequence(length, n_unique):
	return [randint(1, n_unique-1) for _ in range(length)]
 
# prepare data for the LSTM
def get_dataset(n_in, n_out, cardinality, n_samples):
	X1, X2, y = list(), list(), list()
	for _ in range(n_samples):
		# generate source sequence
		source = generate_sequence(n_in, cardinality)
		# define padded target sequence
		target = source[:n_out]
		target.reverse()
		# create padded input target sequence
		target_in = [0] + target[:-1]
		# encode
		src_encoded = to_categorical([source], num_classes=cardinality)
		tar_encoded = to_categorical([target], num_classes=cardinality)
		tar2_encoded = to_categorical([target_in], num_classes=cardinality)
		# store
		X1.append(src_encoded)
		X2.append(tar2_encoded)
		y.append(tar_encoded)
	return array(X1), array(X2), array(y)
 
# returns train, inference_encoder and inference_decoder models
def define_models(n_input, n_output, n_units):
	# define training encoder
	encoder_inputs = Input(shape=(None, n_input))
	encoder = LSTM(n_units, return_state=True)
	encoder_outputs, state_h, state_c = encoder(encoder_inputs)
	encoder_states = [state_h, state_c]
	# define training decoder
	decoder_inputs = Input(shape=(None, n_output))
	decoder_lstm = LSTM(n_units, return_sequences=True, return_state=True)
	decoder_outputs, _, _ = decoder_lstm(decoder_inputs, initial_state=encoder_states)
	decoder_dense = Dense(n_output, activation='softmax')
	decoder_outputs = decoder_dense(decoder_outputs)
	model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
	# define inference encoder
	encoder_model = Model(encoder_inputs, encoder_states)
	# define inference decoder
	decoder_state_input_h = Input(shape=(n_units,))
	decoder_state_input_c = Input(shape=(n_units,))
	decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
	decoder_outputs, state_h, state_c = decoder_lstm(decoder_inputs, initial_state=decoder_states_inputs)
	decoder_states = [state_h, state_c]
	decoder_outputs = decoder_dense(decoder_outputs)
	decoder_model = Model([decoder_inputs] + decoder_states_inputs, [decoder_outputs] + decoder_states)
	# return all models
	return model, encoder_model, decoder_model
 
# generate target given source sequence
def predict_sequence(infenc, infdec, source, n_steps, cardinality):
	# encode
	state = infenc.predict(source)
	# start of sequence input
	target_seq = array([0.0 for _ in range(cardinality)]).reshape(1, 1, cardinality)
	# collect predictions
	output = list()
	for t in range(n_steps):
		# predict next char
		yhat, h, c = infdec.predict([target_seq] + state)
		# store prediction
		output.append(yhat[0,0,:])
		# update state
		state = [h, c]
		# update target sequence
		target_seq = yhat
	return array(output)
 
# decode a one hot encoded string
def one_hot_decode(encoded_seq):
	return [argmax(vector) for vector in encoded_seq]
 
# configure problem
n_features = 50 + 1
n_steps_in = 6
n_steps_out = 3
# define model
train, infenc, infdec = define_models(n_features, n_features, 128)
train.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
# generate training dataset
X1, X2, y = get_dataset(n_steps_in, n_steps_out, n_features, 100000)
print(X1.shape,X2.shape,y.shape)
# train model
train.fit([X1, X2], y, epochs=1)
# evaluate LSTM
total, correct = 100, 0
for _ in range(total):
	X1, X2, y = get_dataset(n_steps_in, n_steps_out, n_features, 1)
	target = predict_sequence(infenc, infdec, X1, n_steps_out, n_features)
	if array_equal(one_hot_decode(y[0]), one_hot_decode(target)):
		correct += 1
print('Accuracy: %.2f%%' % (float(correct)/float(total)*100.0))
# spot check some examples
for _ in range(10):
	X1, X2, y = get_dataset(n_steps_in, n_steps_out, n_features, 1)
	target = predict_sequence(infenc, infdec, X1, n_steps_out, n_features)
	print('X=%s y=%s, yhat=%s' % (one_hot_decode(X1[0]), one_hot_decode(y[0]), one_hot_decode(target)))