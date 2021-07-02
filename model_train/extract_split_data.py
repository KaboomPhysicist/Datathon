import pandas as pd
import numpy as np
from scipy.stats import mode
import matplotlib.pyplot as plt

import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from drive.quickstart_drive import data_download

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences


#Función extractora de datos desde el servidor
def sets(tipo='moda', descarga=False, join=False, graph=False):
        if descarga:
                data_download()

        filepath = 'data/clasificacion.csv'
        df=pd.read_csv(filepath)

        #Elimina las filas con la columna de Item vacío
        df.dropna(subset=["Item (Texto)"],inplace=True)

        #Extrae los párrafos. El método values hace que la extracción se haga en una lista en lugar de un DataFrame
        sentences = df["Item (Texto)"].values

        #saca la lista a partir del string separando por comas, convierte los elementos de la lista en enteros y los lleva a una nueva lista para aplicarle el promedio y redondearlo.
        grav = [int(round(np.mean(list(map(int,i.split(',')))),0)) for i in df['Gravedad']]
        sesgo = [int(round(np.mean(list(map(int,i.split(',')))),0)) for i in df['Sesgo']]

        grav_moda = [mode(list(map(int,i.split(','))))[0][0] for i in df['Gravedad']]
        sesgo_moda = [mode(list(map(int,i.split(','))))[0][0] for i in df['Sesgo']]

        #Gráficas de la distribución de los datos

        if graph:
                fig, axs = plt.subplots(2,2)
                axs[0,0].hist(grav,[0,1,2,3,4], align= 'mid', rwidth=0.8, color='skyblue')
                axs[0,0].set_title('Gravedad con media')
                axs[0,1].hist(sesgo,[-1,0,1,2], align= 'mid', rwidth=0.8, color='red')
                axs[0,1].set_title('Sesgo con media')
                axs[1,0].hist(grav_moda,[0,1,2,3,4], align= 'mid', rwidth=0.8, color='violet')
                axs[1,0].set_title('Gravedad con moda')
                axs[1,1].hist(sesgo_moda,[-1,0,1,2], align= 'mid', rwidth=0.8, color= 'salmon')
                axs[1,1].set_title('Sesgo con moda')

                for ax in axs.flat:
                        ax.label_outer()

                plt.grid()
                plt.show()

        #Para devolver los párrafos junto a una tupla bidimensional de gravedad y sesgo

        if join:
                if tipo=='moda':
                        sentences_train, sentences_test, y_train, y__test = train_test_split(
                                np.array(sentences), np.array([(grav_moda[i],sesgo_moda[i]) for i in range(len(grav_moda))]), test_size=0.25, random_state=1000)
                else:
                        sentences_train, sentences_test, y_train, y_test = train_test_split(
                                np.array(sentences), np.array([(grav[i],sesgo[i]) for i in range(len(grav_moda))]) , test_size=0.25, random_state=1000)

                return sentences_train, sentences_test, y_train, y__test
        
        else:
                if tipo=='moda':
                        sentences_grav_train, sentences_grav_test, grav_train, grav_test = train_test_split(
                                np.array(sentences), np.array(grav_moda), test_size=0.25, random_state=1000)

                        sentences_ses_train, sentences_ses_test, ses_train, ses_test = train_test_split(
                                np.array(sentences), np.array(sesgo_moda), test_size=0.25, random_state=1000)
                else:
                        sentences_grav_train, sentences_grav_test, grav_train, grav_test = train_test_split(
                                np.array(sentences), np.array(grav), test_size=0.25, random_state=1000)

                        sentences_ses_train, sentences_ses_test, ses_train, ses_test = train_test_split(
                                np.array(sentences), np.array(sesgo), test_size=0.25, random_state=1000)

                return sentences_grav_train, sentences_grav_test, grav_train, grav_test, sentences_ses_train, sentences_ses_test, ses_train, ses_test

#Función graficadora de la precisón y el error de los modelos.
def plot_history(history):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    x = range(1, len(acc) + 1)

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(x, acc, 'b', label='Training acc')
    plt.plot(x, val_acc, 'r', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(x, loss, 'b', label='Training loss')
    plt.plot(x, val_loss, 'r', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()

#Función que devuelve los conjuntos de datos vectorizados (CountVectorizer)
def vectorized_set(only_vectorizer=False):
        sentences_grav_train, sentences_grav_test, grav_train, grav_test, sentences_ses_train, sentences_ses_test, ses_train, ses_test=sets()

        vectorizer = CountVectorizer()
        vectorizer.fit(sentences_grav_train)

        if only_vectorizer:
                return vectorizer
        
        else:
                X_grav_train = vectorizer.transform(sentences_grav_train)
                X_grav_test = vectorizer.transform(sentences_grav_test)

                X_ses_train = vectorizer.transform(sentences_ses_train)
                X_ses_test = vectorizer.transform(sentences_ses_test)
        
                return vectorizer, X_grav_train, X_grav_test, X_ses_train, X_ses_test, grav_train, grav_test, ses_train, ses_test

#Función que devuelve los conjuntos de datos tokenizados (para Embedding)
def data_preset(maxlen, train = False):
    sentences_grav_train, sentences_grav_test, grav_train, grav_test, sentences_ses_train, sentences_ses_test, ses_train, ses_test=sets()
    ses_test+=1
    ses_train+=1

    tokenizer = Tokenizer(num_words=5000)
    tokenizer.fit_on_texts(sentences_grav_train)

    tokenizer2 = Tokenizer(num_words=5000)
    tokenizer2.fit_on_texts(sentences_ses_train)

    if train:
        X_grav_train = tokenizer.texts_to_sequences(sentences_grav_train)
        X_grav_test  = tokenizer.texts_to_sequences(sentences_grav_test)

        X_ses_train = tokenizer2.texts_to_sequences(sentences_ses_train)
        X_ses_test = tokenizer2.texts_to_sequences(sentences_ses_test)

        X_grav_train = pad_sequences(X_grav_train, padding='post', maxlen=maxlen)
        X_grav_test = pad_sequences(X_grav_test, padding='post', maxlen=maxlen)

        X_ses_train = pad_sequences(X_ses_train, padding='post',maxlen=maxlen)
        X_ses_test = pad_sequences(X_ses_test, padding='post',maxlen=maxlen)

        return tokenizer, tokenizer2, X_grav_train, X_grav_test, X_ses_train, X_ses_test, grav_train, grav_test, ses_train, ses_test

    else:
        return tokenizer, tokenizer2 

#Crea la matriz de embedding a partir de los datos de preembedding
def create_embedding_matrix(filepath, word_index, embedding_dim):
    vocab_size = len(word_index) + 1
    embedding_matrix = np.zeros((vocab_size, embedding_dim))

    with open(filepath) as f:
        for line in f:
            word, *vector = line.split()
            if word in word_index:
                idx = word_index[word]
                embedding_matrix[idx] = np.array(vector, dtype=np.float32)[:embedding_dim]

    return embedding_matrix
