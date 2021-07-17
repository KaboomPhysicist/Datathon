from extract_split_data import create_embedding_matrix, plot_history, data_preset, pad, aumentar

import matplotlib.pyplot as plt
import numpy as np

import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.models import load_model
from tensorflow.keras import layers
from tensorflow.keras.backend import clear_session
from tensorflow.keras import optimizers

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping

def create_model(tokenizer, embedding_dim, embedding_path, maxlen):
    #Declaración del modelo de Gravedad

    vocab_size = len(tokenizer.word_index) + 1

    embedding_matrix = create_embedding_matrix(embedding_path,tokenizer.word_index, embedding_dim)

    model = Sequential()
    model.add(layers.Embedding(
        input_dim= vocab_size,
        output_dim= embedding_dim,
        weights = [embedding_matrix],
        input_length= maxlen,
        trainable = True
    ))

    model.add(layers.GlobalMaxPooling1D())
    model.add(layers.Dense(25, activation='sigmoid'))
    model.add(layers.Dense(4, activation='softmax'))

    lr_schedule = optimizers.schedules.ExponentialDecay(initial_learning_rate=1e-2,
                                                        decay_steps=10000,
                                                        decay_rate=0.9)

    opt = optimizers.Adam(learning_rate=lr_schedule, clipnorm = 1, clipvalue = 0.5)

    model.compile(optimizer=opt,
                    loss='sparse_categorical_crossentropy',
                    metrics=['accuracy']
            )
    return model

def create_model2(tokenizer, embedding_dim, embedding_path, maxlen):
    #Declaración del modelo de sesgo
    vocab_size = len(tokenizer.word_index) + 1

    embedding_matrix = create_embedding_matrix(embedding_path,tokenizer.word_index, embedding_dim)

    model = Sequential()
    model.add(layers.Embedding(
        input_dim= vocab_size,
        output_dim= embedding_dim,
        weights = [embedding_matrix],
        input_length= maxlen,
        trainable = True
    ))

    model.add(layers.GlobalMaxPooling1D())
    model.add(layers.Dense(15,activation='relu'))
    model.add(layers.Dense(10, activation='relu'))
    model.add(layers.Dense(3, activation='softmax'))
    model.compile(optimizer='adam',
                    loss='sparse_categorical_crossentropy',
                    metrics=['accuracy']
            )
    return model

def train_neural_basic_preembedding(graph=False, embedding_path = '../embeddings/embeddings-l-model.vec', descarga=False, augmention = False):
    maxlen = 100

    tokenizer, tokenizer2, X_grav_train, X_grav_test, X_ses_train, X_ses_test, grav_train, grav_test, ses_train, ses_test = data_preset(train=True, descarga=descarga, augmention=augmention)
    X_grav_train, X_grav_test, X_ses_train, X_ses_test = pad(X_grav_train, X_grav_test, X_ses_train, X_ses_test, maxlen)
    
    embedding_dim = 150

    model = create_model(tokenizer, embedding_dim, embedding_path, maxlen)
    model2 = create_model2(tokenizer2, embedding_dim, embedding_path, maxlen)

    
    #model.summary()
    #model2.summary()

    clear_session()

    es=EarlyStopping(monitor='val_loss',patience=50, restore_best_weights=True)

    history = model.fit(X_grav_train, grav_train,
                    epochs=200,
                    verbose=False,
                    validation_data=(X_grav_test, grav_test),
                    batch_size=10)

    history2 = model2.fit(X_ses_train, ses_train,
                    epochs=200,
                    verbose=False,
                    validation_data=(X_ses_test, ses_test),
                    batch_size=10)
        
    #history.get_params()

    loss, accuracy = model.evaluate(X_grav_train, grav_train, verbose=False)
    print("Precisión de entrenamiento (Gravedad): {:.4f}".format(accuracy))
    loss, accuracy = model.evaluate(X_grav_test, grav_test, verbose=False)
    print("Precisión de prueba (Gravedad):  {:.4f}".format(accuracy))

    loss, accuracy = model2.evaluate(X_ses_train, ses_train, verbose=False)
    print("Precisión de entrenamiento (Sesgo): {:.4f}".format(accuracy))
    loss, accuracy = model2.evaluate(X_ses_test, ses_test, verbose=False)
    print("Precisión de prueba (Sesgo):  {:.4f}".format(accuracy))

    if graph:
        plot_history(history)
        plot_history(history2)
        plt.show()

    model.save('../models/neural_v3_grav.h5')
    model2.save('../models/neural_v3_ses.h5')

def modelo(pers_test):
    model_grav = load_model('../models/neural_v3_grav.h5')
    model_ses = load_model('../models/neural_v3_ses.h5')

    maxlen = 250

    tokenizer, tokenizer2 = data_preset(maxlen)

    test1 = tokenizer.texts_to_sequences(np.array([pers_test]))
    test2 = tokenizer2.texts_to_sequences(np.array([pers_test]))
    test1 = pad_sequences(test1, padding='post', maxlen= maxlen)
    test2 = pad_sequences(test2, padding='post', maxlen= maxlen)

    print(model_grav.predict(test1),model_ses.predict(test2))

if __name__=="__main__":
    for i in range(10):
        train_neural_basic_preembedding(False, augmention=False)
