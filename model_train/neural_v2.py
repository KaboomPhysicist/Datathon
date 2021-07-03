from extract_split_data import plot_history, data_preset, pad

from keras.models import Sequential
from keras.models import load_model
from keras import layers
from keras.backend import clear_session

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

import matplotlib.pyplot as plt
import numpy as np

def create_model(vocab_size, embedding_dim, maxlen):
    #Declaración del modelo de Gravedad
    model = Sequential()
    model.add(layers.Embedding(
        input_dim= vocab_size,
        output_dim= embedding_dim,
        input_length= maxlen
    ))

    model.add(layers.GlobalMaxPooling1D())
    model.add(layers.Dense(10, activation='relu'))
    model.add(layers.Dense(4, activation='softmax'))
    model.compile(optimizer='adam',
                    loss='sparse_categorical_crossentropy',
                    metrics=['accuracy']
            )
    return model

def create_model2(vocab_size, embedding_dim, maxlen):
    #Declaración del modelo de Gravedad
    model = Sequential()
    model.add(layers.Embedding(
        input_dim= vocab_size,
        output_dim= embedding_dim,
        input_length= maxlen
    ))

    model.add(layers.GlobalMaxPooling1D())
    model.add(layers.Dense(10, activation='relu'))
    model.add(layers.Dense(3, activation='softmax'))
    model.compile(optimizer='adam',
                    loss='sparse_categorical_crossentropy',
                    metrics=['accuracy']
            )
    return model

def train_neural_basic_embedding(graph=False):

    maxlen = 250

    tokenizer, tokenizer2, X_grav_train, X_grav_test, X_ses_train, X_ses_test, grav_train, grav_test, ses_train, ses_test = data_preset(train=True)
    X_grav_train, X_grav_test, X_ses_train, X_ses_test = pad(X_grav_train, X_grav_test, X_ses_train, X_ses_test, maxlen)

    vocab_size = len(tokenizer.word_index) + 1

    embedding_dim = 100
    embedding_dim2 = 100

    model = create_model(vocab_size, embedding_dim, maxlen)
    model2 = create_model2(vocab_size, embedding_dim2, maxlen)

    model.summary()
    model2.summary()

    clear_session()

    history = model.fit(X_grav_train, grav_train,
                    epochs=100,
                    verbose=False,
                    validation_data=(X_grav_test, grav_test),
                    batch_size=10)

    history2 = model2.fit(X_ses_train, ses_train,
                    epochs=100,
                    verbose=False,
                    validation_data=(X_ses_test, ses_test),
                    batch_size=10)
        

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

    model.save('models/neural_v2_grav.h5')
    model2.save('models/neural_v2_ses.h5')


def modelo(pers_test):
    model_grav = load_model('models/neural_v2_grav.h5')
    model_ses = load_model('models/neural_v2_ses.h5')

    maxlen = 250

    tokenizer, tokenizer2 = data_preset(maxlen)

    test1 = tokenizer.texts_to_sequences(np.array([pers_test]))
    test2 = tokenizer2.texts_to_sequences(np.array([pers_test]))
    test1 = pad_sequences(test1, padding='post', maxlen= maxlen)
    test2 = pad_sequences(test2, padding='post', maxlen= maxlen)
    print(model_grav.predict(test1),model_ses.predict(test2))

train_neural_basic_embedding(True)