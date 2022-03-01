import pandas as pd
import numpy as np
from ast import literal_eval
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import *
np.random.seed(seed=0)
import tensorflow
from tensorflow.random import set_seed
set_seed(0)
from tqdm import tqdm
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Model, Input
from keras.layers import GRU, LSTM, Embedding, Dense, TimeDistributed, Dropout, Bidirectional
from keras.callbacks import EarlyStopping
from tensorflow.keras.layers import LayerNormalization
from sklearn.metrics import *
from sklearn.model_selection import train_test_split
from keras.metrics import BinaryAccuracy, Precision, Recall, AUC


class LSTM_TRAINABLE_EMBEDS_MIXED():
  def __init__(self,
               rnn_layer_sizes = [128],
               layer_normalize = [True],
               dropouts = [0.1],
               show_summary=True,
               patience=5,
               epochs=100,
               batch_size=128,
               lr=0.001,
               loss='binary_crossentropy',
               max_seq_len = 128,
               embedding_size = 100,
               metrics = [BinaryAccuracy(name='accuracy'),
                          Precision(name='precision'),
                          Recall(name='recall'),
                          AUC(curve = 'PR', name='auc')
                          ],
               monitor_loss = 'val_loss',
               no_of_ml_features = 6
              ):
        self.lr = lr
        self.batch_size = batch_size
        self.rnn_layer_sizes = rnn_layer_sizes
        self.layer_normalize = layer_normalize
        self.dropouts = dropouts
        self.max_seq_len =  max_seq_len
        self.show_summary = show_summary
        self.patience=patience
        self.epochs = epochs
        self.loss = loss
        self.embedding_size = embedding_size
        self.monitor_loss = monitor_loss
        self.metrics = metrics
        self.earlystop = tf.keras.callbacks.EarlyStopping(monitor=self.monitor_loss,
                                                          patience=self.patience,
                                                          verbose=1,
                                                          restore_best_weights=True,
                                                          mode="min"
                                                         )
        self.unk_token = "[unk]"
        self.pad_token = "[pad]"
        self.no_of_ml_features = no_of_ml_features


  def build(self, vocab_size):
      text_inputs= Input(shape=(self.max_seq_len), name="inputs")
      ml_inputs= Input(shape=(self.no_of_ml_features), name="ml_inputs")
      #x = inputs
      x = Embedding(input_dim=vocab_size, output_dim=self.embedding_size, 
                      input_length=self.max_seq_len, mask_zero=True, trainable=True)(text_inputs)
      for i in range(len(self.rnn_layer_sizes)):
        if self.dropouts[i]:
          x = Dropout(self.dropouts[i])(x)
        if i == len(self.rnn_layer_sizes)-1:
          x = LSTM(self.rnn_layer_sizes[i], return_sequences=False)(x) #last layer
        else: #hidden layer
          x = LSTM(self.rnn_layer_sizes[i], return_sequences=True)(x)
        if self.layer_normalize[i]:
          x = LayerNormalization()(x)
      #concat ML features with the last hidden state of the LSTM
      x = tf.keras.layers.concatenate([x, ml_inputs])
      x = Dense(64, activation = 'relu')(x)
      pred = Dense(1, activation='sigmoid')(x)
      self.model = Model(inputs=[text_inputs, ml_inputs], outputs=pred)
      self.model.compile(loss=self.loss,
                    optimizer=tf.keras.optimizers.Adam(learning_rate=self.lr),
                    metrics=self.metrics)
      if self.show_summary:
          self.model.summary()

  def create_vocab(self, tokenized_actions):
    self.vocab = list(set([subaction for action in tokenized_actions for subaction in action]))
    self.vocab_size = len(self.vocab) + 1
    print("Vocab size: ", self.vocab_size)
    self.w2i = {w: i+2 for i,w in enumerate(self.vocab)}
    self.w2i[self.unk_token] = 1
    self.w2i[self.pad_token] = 0
    self.i2w = {i+2: self.w2i[w] for i,w in enumerate(self.vocab)}
    self.i2w[1] = self.unk_token
    self.i2w[0] = self.pad_token
  
  def to_sequences(self, tokenized_actions):
        x = [[self.w2i[subaction] if subaction in self.w2i else self.w2i[self.unk_token] for subaction in action] 
             for action in tokenized_actions]
        x = pad_sequences(sequences=x, maxlen=self.max_seq_len, padding="post", value=0)  # padding
        return x
  
  def fit(self, tokenized_actions, ml_inputs, y, val_tokenized_actions, val_ml_inputs, val_y):
    # Create vocab and lookup tables
    self.create_vocab(tokenized_actions)
    # turn the tokenized texts and token labels to padded sequences of indices
    X = self.to_sequences(tokenized_actions)
    # build the model and compile it
    self.build(self.vocab_size)
    # start training
    vx = self.to_sequences(val_tokenized_actions)
    history = self.model.fit([X, ml_inputs], y, batch_size=self.batch_size, epochs=self.epochs, validation_data=([vx, val_ml_inputs], val_y),
                             verbose=1, callbacks=[self.earlystop])
    return history
  
  def predict(self, tokenized_actions, ml_inputs):
        predictions = self.model.predict([self.to_sequences(tokenized_actions), ml_inputs])
        return predictions
