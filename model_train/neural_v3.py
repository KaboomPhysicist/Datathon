from extract_split_data import sets, create_embedding_matrix, plot_history, data_preset, pad

import matplotlib.pyplot as plt
import numpy as np

import tensorflow as tf
import os, os.path

import pickle

from tensorflow.keras.models import Sequential
from tensorflow.keras.models import load_model
from tensorflow.keras import layers
from tensorflow.keras.backend import clear_session
from tensorflow.keras import optimizers

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

from sklearn.metrics import confusion_matrix, precision_score, recall_score
from mlxtend.plotting import plot_confusion_matrix

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

#    model.add(layers.Conv1D(200, 80, activation='relu'))
    model.add(layers.GlobalMaxPooling1D())
    model.add(layers.Dense(60,activation='tanh',kernel_regularizer=tf.keras.regularizers.l1(0.005),bias_regularizer='l1'))
    model.add(layers.Dense(25, activation='relu'))
    model.add(layers.Dense(4, activation='softmax'))

    lr_schedule = optimizers.schedules.ExponentialDecay(initial_learning_rate=1e-2,
                                                        decay_steps=10000,
                                                        decay_rate=0.9)

    opt = optimizers.Adam(learning_rate=0.001, clipnorm = 1, clipvalue = 0.5)

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
    model.add(layers.Dropout(0.3))
    model.add(layers.Dense(10, activation='relu'))
    model.add(layers.Dense(3, activation='softmax'))
    model.compile(optimizer='adam',
                    loss='sparse_categorical_crossentropy',
                    metrics=['accuracy']
            )
    return model

def train_neural_basic_preembedding(graph=False, embedding_path = '../embeddings/embeddings-l-model.vec', descarga=False, augment = False):
    maxlen = 300

    tokenizer, tokenizer2, X_grav_train, X_grav_test, X_ses_train, X_ses_test, grav_train, grav_test, ses_train, ses_test = data_preset(train=True, descarga=descarga, augment=augment)
    X_grav_train, X_grav_test, X_ses_train, X_ses_test = pad(X_grav_train, X_grav_test, X_ses_train, X_ses_test, maxlen)
    
    embedding_dim = 300

    model = create_model(tokenizer, embedding_dim, embedding_path, maxlen)
    model2 = create_model2(tokenizer2, embedding_dim, embedding_path, maxlen)

   # model.summary()
   # model2.summary()

    clear_session()

    es=EarlyStopping(monitor='val_loss',patience=50, restore_best_weights=True)
    mcp_save = ModelCheckpoint('./checkpoint',save_best_only=True, monitor='val_acc', mode='max')

    history = model.fit(X_grav_train, grav_train,
                    epochs=500,
                    verbose=False,
                    validation_data=(X_grav_test, grav_test),
                    batch_size=128)

    history2 = model2.fit(X_ses_train, ses_train,
                    epochs=500,
                    verbose=False,
                    validation_data=(X_ses_test, ses_test),
                    batch_size=128)
        

    loss, accuracy = model.evaluate(X_grav_train, grav_train, verbose=False)
    print("Precisión de entrenamiento (Gravedad): {:.4f}".format(accuracy))
    loss, accuracy = model.evaluate(X_grav_test, grav_test, verbose=False)
    print("Precisión de prueba (Gravedad):  {:.4f}".format(accuracy))

    loss, accuracy = model2.evaluate(X_ses_train, ses_train, verbose=False)
    print("Precisión de entrenamiento (Sesgo): {:.4f}".format(accuracy))
    loss, accuracy = model2.evaluate(X_ses_test, ses_test, verbose=False)
    print("Precisión de prueba (Sesgo):  {:.4f}".format(accuracy))
    print("--------------------------------------------------------------------")

    DIR = '../models/Modelos gravedad/'
    version = int(len([name for name in os.listdir(DIR) if os.path.isfile(os.path.join(DIR, name))]))

    if graph:
        plot_history(history)
        plt.savefig(f'../performance/accuracy/Gravedad/accuracy_grav_v{version}')
        plot_history(history2)
        plt.savefig(f'../performance/accuracy/Sesgo/accuracy_ses_v{version}')

    

    model.save(f'../models/Modelos gravedad/neural_v3_grav_r{version}.h5')
    model2.save(f'../models/Modelos sesgo/neural_v3_ses_r{version}.h5')

    with open(f'../models/tokenizers/Gravedad/tokenizer_r{version}.pickle','wb') as handle1:
        pickle.dump(tokenizer, handle1, protocol=pickle.HIGHEST_PROTOCOL)

    with open(f'../models/tokenizers/Sesgo/tokenizer2_r{version}.pickle','wb') as handle:
        pickle.dump(tokenizer2, handle, protocol=pickle.HIGHEST_PROTOCOL)

def modelo(pers_test, version):
    model_grav = load_model(f'../models/Modelos gravedad/neural_v3_grav_r{version}.h5')
    model_ses = load_model(f'../models/Modelos sesgo/neural_v3_ses_r{version}.h5')

    maxlen = 300

    with open(f'../models/tokenizers/Gravedad/tokenizer_r{version}.pickle','rb') as handle1:
        tokenizer = pickle.load(handle1)

    with open(f'../models/tokenizers/Sesgo/tokenizer2_r{version}.pickle','rb') as handle:
        tokenizer2 = pickle.load(handle)

    test1 = tokenizer.texts_to_sequences(np.array([pers_test]))
    test2 = tokenizer2.texts_to_sequences(np.array([pers_test]))
    test1 = pad_sequences(test1, padding='post', maxlen= maxlen)
    test2 = pad_sequences(test2, padding='post', maxlen= maxlen)

    print(f"{pers_test}\nGravedad: {model_grav.predict(test1)}\nSesgo: {model_ses.predict(test2)}")

def cm(y_true,y_pred):
    #print(confusion_matrix(y_true, y_pred))
    a = confusion_matrix(y_true,y_pred)
    sum_first_diagonal = sum(a[i][i] for i in range(len(a)))

    fig, ax = plot_confusion_matrix(confusion_matrix(y_true,y_pred), cmap='Reds')
    ax.set(title=(str(np.sum(a) - sum_first_diagonal)))
    ax.text(-0.5,-0.5,str(precision_score(y_true,y_pred, average='macro')))


def metricas(maxlen,version=int(len([name for name in os.listdir('../models') if os.path.isfile(os.path.join('../models', name))]))-1):
    
    model_grav = load_model(f'../models/Modelos gravedad/neural_v3_grav_r{version}.h5')
    model_ses = load_model(f'../models/Modelos sesgo/neural_v3_ses_r{version}.h5')

    with open(f'../models/tokenizers/Gravedad/tokenizer_r{version}.pickle','rb') as handle1:
        tokenizer = pickle.load(handle1)

    with open(f'../models/tokenizers/Sesgo/tokenizer2_r{version}.pickle','rb') as handle:
        tokenizer2 = pickle.load(handle)
    
    sentences, grav_true, ses_true = sets(split=False)
    ses_true+=1

    data = tokenizer.texts_to_sequences(sentences)
    data2 = tokenizer2.texts_to_sequences(sentences)

    data = pad_sequences(data, padding='post', maxlen= maxlen)
    data2 = pad_sequences(data2, padding='post', maxlen= maxlen)
    
    grav_pred = model_grav.predict(data)
    ses_pred = model_ses.predict(data2)

    grav_val = np.round(grav_pred).argmax(axis=1)
    ses_val= np.round(ses_pred).argmax(axis=1)

    cm(ses_true, ses_val)
    plt.savefig(f'../performance/confussion_matrix/Sesgo/Confussion_matrix_sesgo_v{version}')
    cm(grav_true, grav_val)
    plt.savefig(f'../performance/confussion_matrix/Gravedad/Confussion_matrix_gravedad_v{version}')


if __name__=="__main__":
    #train_neural_basic_preembedding(True, embedding_path='../embeddings/fasttext-sbwc.vec', descarga=False, augment=False)
    #metricas(300,i)
    modelo("Ingrese la frase a ser clasificada: Las calles y muros de varias ciudades colombianas son por estos días los lienzos de las manifestaciones que se vienen dando en el paro nacional que ya cumple 27 jornadas. Los artistas que intervienen estos espacios son conscientes de que los mensajes que plasman, muy posiblemente, serán borrados en poco tiempo. Algunos académicos, sin embargo, resaltan en el arte de calle el poder de generar diálogo y su potencial como acción política.",23)