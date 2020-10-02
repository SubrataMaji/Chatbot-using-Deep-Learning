# Imports
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


#-----------------------------------------------------------------
# Beam Search
#-----------------------------------------------------------------
def beam_predict(input_sentence, model, enc_tokenizer, dec_tokenizer, seq_len, beam_index=3):
    " For a given question this function return a answer "
    # Preparing input data
    input_tokens = enc_tokenizer.texts_to_sequences([input_sentence])
    input_sequence = pad_sequences(input_tokens, maxlen=seq_len, padding='post')
    
    # Getting encoder output and states
    enc_initial_states = model.encoder.initialize_states(len(input_sequence))
    enc_out, enc_state_h, enc_state_c, enc_mask = model.encoder(input_sequence, enc_initial_states)
    state_h, state_c = enc_state_h, enc_state_c
    
    # start and end output tokens
    start_token = dec_tokenizer.word_index['<start>']
    end_token = dec_tokenizer.word_index['<end>']
    # Sending '<start>' as 1st word of decoder
    target_word = np.zeros((1,1))
    target_word[0,0] = start_token

    result = [[[start_token], 0]]
    
    # Predicting upto maximum length
    while len(result[0][0]) < seq_len:
        temp = []
        for sent_token in result:
            # decoder layer, intial states are encoder's final states
            output, dec_state_h, dec_state_c, attention_weights, _ = model.decoder.onestepdecoder(target_word, enc_out, state_h, state_c, enc_mask)
            pred_index = np.argsort(output[-1])[-beam_index:]
            for index in pred_index:
                seq, prob = sent_token[0][:], sent_token[1]
                seq.append(index)
                prob += output[-1][index]
                temp.append([seq, prob])
            result = temp

        # Sorting according to the probabilities
        result = sorted(result, reverse=False, key=lambda l: l[1])
        result = result[-beam_index:]  # Getting the top words

        predicted_list = result[-1][0]
        predicted_index = predicted_list[-1]
        if predicted_index != end_token:
            target_word = np.zeros((1, 1))
            target_word[0,0] = predicted_index
            state_h, state_c = dec_state_h, dec_state_c
        else:
            break
    
    # Converting tokens to words
    result = result[-1][0]
    pred_words = [dec_tokenizer.index_word[i] for i in result]    
    pred_sentence = []
    for word in pred_words:
        if word != '<end>':
            pred_sentence.append(word)
        else:
            break

    return " ".join(pred_sentence[1:])
    
    
    
#-----------------------------------------------------------------
# Greedy Search with batches
#-----------------------------------------------------------------
def predict(input_sentence, model, enc_tokenizer, dec_tokenizer, seq_len):
    " For a given question this function return a answer "
    
    # Preparing input data
    input_tokens = enc_tokenizer.texts_to_sequences(input_sentence)
    input_sequence = pad_sequences(input_tokens, maxlen=seq_len, padding='post')
    batch_size = input_sequence.shape[0]
    
    # Getting encoder output and states
    enc_initial_states = model.encoder.initialize_states(batch_size)
    enc_out, enc_state_h, enc_state_c, enc_mask = model.encoder(input_sequence, enc_initial_states)
    state_h, state_c = enc_state_h, enc_state_c

    # Sending '<start>' as 1st word of decoder
    target_word = np.zeros((batch_size,1))
    target_word[:,0] = dec_tokenizer.word_index['<start>']

    end_token = dec_tokenizer.word_index["<end>"]
    outwords = []
    while np.array(outwords).shape[0] < seq_len:
        # decoder layer, intial states are encoder's final states
        output, dec_state_h, dec_state_c, attention_weights, _ = model.decoder.onestepdecoder(target_word, enc_out, state_h, state_c, enc_mask)
        out_word_index = np.argmax(output, -1)
        outwords.append(out_word_index)
        
        # current output word is input word for next timestamp
        target_word = np.zeros((batch_size,1))
        target_word[:,0] = out_word_index
        
        # current out states are input states for next timestamp
        state_h, state_c = dec_state_h, dec_state_c
        # If last predicted word for all the sentence in batch is end token
        if (np.array(outwords[-1])==end_token).all():
            break
    
    # Converting tokens to words
    sentences = []
    outwords = np.array(outwords)
    for sent_token in outwords.T:
        current_sent = ""
        for ind in sent_token:
            if ind != end_token:
                current_sent += dec_tokenizer.index_word[ind] + " "
            else:
                break
        sentences.append(current_sent.strip())

    return sentences
