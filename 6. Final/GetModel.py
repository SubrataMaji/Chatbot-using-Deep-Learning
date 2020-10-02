#-----------------------------------------------------------------------
# Imports
#-----------------------------------------------------------------------
import numpy as np
import pandas as pd
import re, string
import random as rn
from datetime import datetime
import time
import pickle
import warnings
warnings.filterwarnings("ignore")

import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.keras.layers import Embedding, LSTM, Dense, Softmax
from tensorflow.keras.layers import Bidirectional, Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences



#-----------------------------------------------------------------------
# Encoder
#-----------------------------------------------------------------------
class Encoder(tf.keras.Model):
    ''' Encoder model -- That takes a input sequence
    returns encoder-outputs, encoder_final_state_h, encoder_final_state_c '''
    def __init__(self, inp_vocab_size, embedding_size, lstm_size, input_length, qsn_matrix):
        super().__init__()
        self.vocab_size = inp_vocab_size
        self.embedding_dim = embedding_size
        self.lstm_units = lstm_size
        self.input_length = input_length
        self.qsn_matrix = qsn_matrix
        self.enc_output = self.enc_state_h = self.enc_state_c = 0
        
        #Initialize Embedding layer, output shape: (batch_size, input_length, embedding_dim)
        self.embedding = Embedding(self.vocab_size, self.embedding_dim,
                                   embeddings_initializer=tf.keras.initializers.Constant(self.qsn_matrix), trainable=False,
                                   input_length=self.input_length, mask_zero=True, name="encoder_Embedding")
        
        #Intialize Encoder LSTM layer
        self.lstm = LSTM(units=self.lstm_units, activation='tanh', recurrent_activation='sigmoid',
                         kernel_initializer=tf.keras.initializers.glorot_uniform(seed=26),
                         recurrent_initializer=tf.keras.initializers.orthogonal(seed=54),
                         bias_initializer=tf.keras.initializers.zeros(), return_state=True, return_sequences=True, name="encoder_LSTM")
        
        # Bidirectional layer
        self.bidirectional = Bidirectional(self.lstm)


    def call(self, input_sequence, states):
        '''This function takes a sequence input and the initial states of the encoder.'''
        # Embedding inputs, using pretrained glove vectors
        embedded_input = self.embedding(input_sequence)  # shape: (input_length, glove vector's dimension)
        
        # mask for padding
        mask = self.embedding.compute_mask(input_sequence)
        
        # enc_out shape: (batch_size, input_length, lstm_size) & forward or backward h and c: (batch_size, lstm_size)
        self.enc_out, enc_fw_state_h, enc_bw_state_h, enc_fw_state_c, enc_bw_state_c = self.bidirectional(embedded_input, mask=mask)
        
        # Concatenating forward and backward states
        self.enc_state_h = Concatenate()([enc_fw_state_h, enc_bw_state_h])  # enc_state_h and c shape: (batch_size, 2*lstm_size)
        self.enc_state_c = Concatenate()([enc_fw_state_c, enc_bw_state_c])
        
        return self.enc_out, self.enc_state_h, self.enc_state_c, mask


    def initialize_states(self,batch_size):
      '''Given a batch size it will return intial hidden state and intial cell state.'''
      return (tf.zeros([batch_size, 2*self.lstm_units]), tf.zeros([batch_size, 2*self.lstm_units]))
      
      
      
#-----------------------------------------------------------------------
# Attention
#-----------------------------------------------------------------------
class Attention(tf.keras.layers.Layer):
    ''' Class the calculates score based on the scoring_function using Bahdanu attention mechanism. '''
    
    def __init__(self,scoring_function, att_units):
        super().__init__()
        self.scoring_function = scoring_function
        self.att_units = att_units

        # Initializing for 3 kind of losses
        if self.scoring_function=='dot':
            # Intialize variables needed for Dot score function here
            self.dot = tf.keras.layers.Dot(axes=[2,2])
        if scoring_function == 'general':
            # Intialize variables needed for General score function here
            self.wa = Dense(self.att_units)
        elif scoring_function == 'concat':
            # Intialize variables needed for Concat score function here
            self.wa = Dense(self.att_units, activation='tanh')
            self.va = Dense(1)
  
  
    def call(self,decoder_hidden_state,encoder_output, enc_mask):
        ''' Attention mechanism takes two inputs current step -- decoder_hidden_state and all the encoder_outputs. '''
        
        decoder_hidden_state = tf.expand_dims(decoder_hidden_state, axis=1)
        # mask from encoder
        enc_mask = tf.expand_dims(enc_mask, axis=-1)
        
        # score shape: (batch_size, input_length, 1)
        if self.scoring_function == 'dot':
            # Implementing Dot score function
            score = self.dot([encoder_output, decoder_hidden_state])
        elif self.scoring_function == 'general':
            # Implementing General score function here            
            score = tf.keras.layers.Dot(axes=[2, 2])([self.wa(encoder_output), decoder_hidden_state])
        elif self.scoring_function == 'concat':
            # Implementing General score function here
            decoder_output = tf.tile(decoder_hidden_state, [1, encoder_output.shape[1], 1])
            score = self.va(self.wa(tf.concat((decoder_output, encoder_output), axis=-1)))
            
        score = score + (tf.cast(tf.math.equal(enc_mask, False), score.dtype)*-1e9)
        
        # shape: (batch_size, input_length, 1)
        attention_weights = Softmax(axis=1)(score)
        enc_mask = tf.cast(enc_mask, attention_weights.dtype)
        
        # masking attention weights
        attention_weights = attention_weights * enc_mask

        context_vector = attention_weights * encoder_output
        # shape = (batch_size, dec lstm units)
        context_vector = tf.reduce_sum(context_vector, axis=1)

        return context_vector, attention_weights
        
        
        
#-----------------------------------------------------------------------
# OneStepDecoder
#-----------------------------------------------------------------------
class OneStepDecoder(tf.keras.Model):
    
    def __init__(self,tar_vocab_size, embedding_dim, input_length, dec_units ,score_fun ,att_units, ans_matrix):
        """ In this layer calculate the ooutput for a single timestep """
        
        super().__init__()
        # Initialize decoder embedding layer, LSTM and any other objects needed
        self.vocab_size = tar_vocab_size
        self.embedding_dim = embedding_dim
        self.input_length = input_length
        self.dec_units = dec_units
        self.score_fun = score_fun
        self.att_units = att_units
        self.ans_matrix = ans_matrix

        self.embedding = Embedding(self.vocab_size, self.embedding_dim,
                                    embeddings_initializer=tf.keras.initializers.Constant(self.ans_matrix), trainable=False,
                                    input_length=self.input_length, mask_zero=True, name="Att_Dec_Embedding")

        self.lstm = LSTM(self.dec_units, return_sequences=True, return_state=True, name="Att_Dec_LSTM")
        self.fc = Dense(self.vocab_size)
        # attention layers
        self.attention = Attention(self.score_fun, self.att_units)


    def call(self,input_to_decoder, encoder_output, state_h, state_c, enc_mask):
        ''' Calling this function by passing decoder input for a single timestep, encoder output and encoder final states '''
        
        # shape: (batchsize, input_length, embedding dim)
        embedded_input = self.embedding(input_to_decoder)
        # shape: (batch_size, dec lstm units)
        context_vector, attention_weights = self.attention(state_h, encoder_output, enc_mask)
        # (batch_size, 1, dec lstm units)
        decoder_input = tf.concat([tf.expand_dims(context_vector, 1), embedded_input], axis=-1)
        # output shape: (batch size, input length, lstm units), state shape: (batch size, lstm units)
        decoder_output, dec_state_h, dec_state_c = self.lstm(decoder_input, initial_state=[state_h, state_c])
        # (batch_size, lstm units)
        decoder_output = tf.reshape(decoder_output, (-1, decoder_output.shape[2]))
        # (batch size, vocab size)
        output = self.fc(decoder_output)

        return output, dec_state_h, dec_state_c, attention_weights, context_vector
        
        
        
#-----------------------------------------------------------------------
# Decoder
#-----------------------------------------------------------------------
class Decoder(tf.keras.Model):
    """ Decoder class, takes decoder input, encoder outputs and states and return predicted sentence """
    
    def __init__(self,out_vocab_size, embedding_dim, input_length, dec_units ,score_fun ,att_units, ans_matrix):
        #Intialize necessary variables and create an object from the class onestepdecoder
        super().__init__()
        self.vocab_size = out_vocab_size
        self.embedding_dim = embedding_dim
        self.input_length = input_length
        self.dec_units = dec_units
        self.score_fun = score_fun
        self.att_units = att_units
        self.ans_matrix = ans_matrix
 
        # Initializing onestepdecoder layer
        self.onestepdecoder = OneStepDecoder(self.vocab_size, self.embedding_dim, self.input_length,
                                            self.dec_units, self.score_fun, self.att_units, self.ans_matrix)
 
 
    @tf.function
    def call(self, input_to_decoder,encoder_output,decoder_hidden_state,decoder_cell_state, enc_mask):
 
        #Initializing an empty Tensor array, that will store the outputs at each and every time step    
        all_outputs = tf.TensorArray(tf.float32, size=input_to_decoder.shape[1], name="Output_array")
 
        #Iterate till the length of the decoder input
        for timestep in range(input_to_decoder.shape[1]):
            # Calling onestepdecoder for each token in decoder_input
            output, decoder_hidden_state, decoder_cell_state, attention_weights, context_vector = self.onestepdecoder(
                input_to_decoder[:, timestep:timestep+1], encoder_output, decoder_hidden_state, decoder_cell_state, enc_mask)
            # Store the output in tensorarray
            all_outputs = all_outputs.write(timestep, output)
        
        all_outputs = tf.transpose(all_outputs.stack(), [1,0,2])
        
        # Return the tensor array
        return all_outputs
        
        

#-----------------------------------------------------------------------
# Encoder-Decoder
#-----------------------------------------------------------------------
class Encoder_decoder(tf.keras.Model):
    
    def __init__(self, **params):
        
        super().__init__()
        self.inp_vocab_size = params['inp_vocab_size']
        self.out_vocab_size = params['out_vocab_size']
        self.embedding_size = params['embedding_size']
        self.lstm_size = params['lstm_units']
        self.input_length = params['input_length']
        self.batch_size = params['batch_size']
        self.score_fun = params["score_fun"]
        self.qsn_matrix = params["qsn_matrix"]
        self.ans_matrix = params["ans_matrix"]
        
        #Create encoder object
        self.encoder = Encoder(self.inp_vocab_size+1, embedding_size=self.embedding_size, lstm_size=self.lstm_size, input_length=self.input_length, qsn_matrix=self.qsn_matrix)
 
        #Create decoder object
        self.decoder = Decoder(self.out_vocab_size+1, embedding_dim=self.embedding_size, input_length=self.input_length,
                               dec_units=2*self.lstm_size, score_fun=self.score_fun, att_units=2*self.lstm_size, ans_matrix=self.ans_matrix)
 
 
    def call(self, data):
        ''' Calling the model with ([encoder input, decoder input], decoder outpur) '''
        input, output = data[0], data[1]
 
        enc_initial_states = self.encoder.initialize_states(self.batch_size)
        enc_out, enc_state_h, enc_state_c, enc_mask = self.encoder(input, enc_initial_states)
 
        dec_out = self.decoder(output, enc_out, enc_state_h, enc_state_c, enc_mask)
 
        return dec_out
        
        
        
#-----------------------------------------------------------------------------
# Compiling the Model with custom loss function
#-----------------------------------------------------------------------------
def custom_lossfunction(real, pred):
    # Custom loss function that will not consider the loss for padded zeros.
    # Refer https://www.tensorflow.org/tutorials/text/nmt_with_attention#define_the_optimizer_and_the_loss_function
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')
    
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    loss_ = loss_object(real, pred)
 
    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask
 
    return tf.reduce_mean(loss_)
    

# To get the final compiled model
def get_model(QSN_VOCAB_SIZE, ANS_VOCAB_SIZE, EMBEDDING_SIZE, LSTM_UNITS, MAXLEN, BATCH_SIZE, \
            SCORE_FUN, QSN_EMB, ANS_EMB):
    
    #Create an object of encoder_decoder Model class,
    model = Encoder_decoder(inp_vocab_size=QSN_VOCAB_SIZE,
                        out_vocab_size=ANS_VOCAB_SIZE,
                        embedding_size=EMBEDDING_SIZE,
                        lstm_units=LSTM_UNITS,                        
                        input_length=MAXLEN,
                        batch_size=BATCH_SIZE,
                        score_fun=SCORE_FUN,
                        qsn_matrix=QSN_EMB,
                        ans_matrix=ANS_EMB)
    
    # Optimizer                        
    adam_optimizer = tf.keras.optimizers.Adam()
    optimizer = tfa.optimizers.SWA(adam_optimizer)
    
    # Compile the model and fit the model
    model.compile(optimizer=optimizer, loss=custom_lossfunction, metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])
    
    return model