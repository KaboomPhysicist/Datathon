import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model_train.extract_split_data import plot_history

from drive.quickstart_drive import data_download
from data_augmentation.back_translation import maind
from data_augmentation.synaug import main

import nltk

from tensorflow import keras
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv1D, Dropout, GlobalMaxPool1D, Embedding, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix, precision_score, recall_score

#data_download()

#nltk.download('averaged_perceptron_tagger')
#nltk.download('wordnet')
#nltk.download('omw')

#main()
#print('nlpaug done')
#maind()
#print('back_translation done')
CSV_Path1 ="../data_augmentation/Train-google-en_grav.csv"
CSV_Path2 ="../data_augmentation/Test-google-en_grav.csv"

df_concat = pd.read_csv(CSV_Path1, header = 0)
df3 = pd.read_csv(CSV_Path2, header = 0)

X_train = df_concat['Item (Texto)'].values
y_train = df_concat['GravedadMode'].values

X_test = df3['Item (Texto)'].values
y_test = df3['GravedadMode'].values

t = Tokenizer()
t.fit_on_texts(X_train)
t.fit_on_texts(X_test)

vocab_size = len(t.word_index) + 1

sequences1 = t.texts_to_sequences(X_train)
sequences2 = t.texts_to_sequences(X_test)

def max_news(seq):
    for i in range(1, len(seq)):
        max_length = len(seq[0])
        if len(seq[i]) > max_length:
            max_length = len(seq[i])
    return max_length

news_num1 = max_news(seq = sequences1)

news_num2 = max_news(seq = sequences2)

padded_x_train = pad_sequences(sequences1, padding='pre', maxlen=max(news_num1, news_num2))
padded_x_test = pad_sequences(sequences2, padding='pre', maxlen=max(news_num1, news_num2))

labels_train = to_categorical(np.asarray(y_train))
labels_test = to_categorical(np.asarray(y_test))

X_train, X_test, y_train, y_test = padded_x_train, padded_x_test, labels_train, labels_test

X_val = X_test[-int(y_test.shape[0]/2):]
y_val = y_test[-int(y_test.shape[0]/2):]
X_test = X_test[:-int(y_test.shape[0]/2)]
y_test = y_test[:-int(y_test.shape[0]/2)]

print('X_train size:', X_train.shape)
print('y_train size:', y_train.shape)
print('X_test size:', X_test.shape)
print('y_test size:', y_test.shape)
print('X_val size:', X_val.shape)
print('y_val size:', y_val.shape)

embedding = '../embeddings/embeddings-l-model.vec'

embeddings_index = dict()
f = open(embedding)
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()
print('Loaded %s word vectors.' % len(embeddings_index))

embedding_matrix = np.zeros((vocab_size, 300))

# relleno de la matriz
for word, i in t.word_index.items():  # diccionario
    embedding_vector = embeddings_index.get(word) # obtención de los vectores embedded de la palabra en GloVe.
    if embedding_vector is not None:
        # adición en la matriz
        embedding_matrix[i] = embedding_vector # cada fila de la matriz.

keras.backend.clear_session()

# Creación de la capa embedding esando la matriz embedding predefinida.
# la entrada será vocab_size, y la salida 300
# para cargar los pesos de la matriz embedding hacemos trainable = False
embedding_layer = Embedding(input_dim=vocab_size, output_dim=300, weights=[embedding_matrix],input_length = max(news_num1, news_num2), trainable=True)

es=EarlyStopping(monitor='val_loss',patience=30, restore_best_weights=True)

def create_model(neurons=20, momentum=0.9):
  mod = Sequential()
  mod.add(embedding_layer)
  mod.add(Conv1D(450, 4, activation='relu'))
  mod.add(GlobalMaxPool1D())
  mod.add(Dense(neurons, activation='relu', kernel_initializer='he_uniform', kernel_regularizer=keras.regularizers.l1(0.005),bias_regularizer='l1'))
  mod.add(Dropout(0.2))
  mod.add(Dense(neurons, activation='relu', kernel_initializer='he_uniform', kernel_regularizer=keras.regularizers.l1(0.005),bias_regularizer='l1'))
  mod.add(Dropout(0.2))
  mod.add(Dense(neurons, activation='relu', kernel_initializer='he_uniform', kernel_regularizer=keras.regularizers.l1(0.005),bias_regularizer='l1'))
  mod.add(Dropout(0.2))
  mod.add(BatchNormalization())
  mod.add(Dense(neurons, activation='relu', kernel_initializer='he_uniform', kernel_regularizer=keras.regularizers.l1(0.005),bias_regularizer='l1'))
  mod.add(Dropout(0.2))
  mod.add(Dense(neurons, activation='relu', kernel_initializer='he_uniform', kernel_regularizer=keras.regularizers.l1(0.005),bias_regularizer='l1'))
  mod.add(Dropout(0.2))
  mod.add(Dense(neurons, activation='relu', kernel_initializer='he_uniform', kernel_regularizer=keras.regularizers.l1(0.005),bias_regularizer='l1'))
  mod.add(Dense(4, activation='softmax'))
  opt = keras.optimizers.SGD(learning_rate=0.01, momentum=momentum, nesterov=True, clipnorm = 1, clipvalue = 0.5)
  mod.compile(optimizer=opt,
                loss='categorical_crossentropy',
                metrics=['accuracy'])
  mod.summary()
  return mod

model = KerasClassifier(build_fn=create_model,epochs=500,batch_size=256, callbacks=[es])


#momentum = [0.1, 0.3, 0.5, 0.7, 0.9, 1]
#param_grid = dict(momentum = momentum )

#grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1, cv=3)
#grid_result = grid.fit(X_train, y_train,validation_data=(X_test, y_test))


#print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
#means = grid_result.cv_results_['mean_test_score']
#stds = grid_result.cv_results_['std_test_score']
#params = grid_result.cv_results_['params']
#for mean, stdev, param in zip(means, stds, params):
#    print("%f (%f) with: %r" % (mean, stdev, param))

plt.style.use('ggplot')

keras.backend.clear_session()
mod = create_model(neurons=60, momentum=0.9)

es=EarlyStopping(monitor='val_loss',patience=30, restore_best_weights=True)
mcp_save = ModelCheckpoint('./checkpoint',save_best_only=True, monitor='val_accuracy', mode='max')

history = mod.fit(X_train, y_train,
                            batch_size=256,
                            epochs=700,
                            validation_data=(X_val, y_val),
                            callbacks=[es,mcp_save])
loss, accuracy = mod.evaluate(X_train, y_train, verbose=False)
print("Training Accuracy: {:.4f}".format(accuracy))
loss, accuracy = mod.evaluate(X_test, y_test, verbose=False)
print("Testing Accuracy:  {:.4f}".format(accuracy))
plot_history(history)

mod.load_weights('./checkpoint')


loss, accuracy = mod.evaluate(X_train, y_train, verbose=False)
print("Training Accuracy: {:.4f}".format(accuracy))
loss, accuracy = mod.evaluate(X_test, y_test, verbose=False)
print("Testing Accuracy:  {:.4f}".format(accuracy))

mod.save('..models/testing/prueba_gravedad.h5')
#y_true = np.concatenate((y_train, y_test), axis=0)
#X_true = np.concatenate((X_train, X_test), axis=0)

#y_pred = mod.predict(X_true)
#y_pred = np.round(y_pred)

#true = np.zeros(len(y_true))
#for i in range(len(y_true)):
#  true[i] = np.argmax(y_true[i])

#pred = np.zeros(len(y_pred))
#for i in range(len(y_pred)):
#  pred[i] = np.argmax(y_pred[i])

#from mlxtend.plotting import plot_confusion_matrix

#def cm(y_true,y_pred):
#  return plot_confusion_matrix(confusion_matrix(y_true,y_pred), cmap='Reds')

#cm(true, pred)

#precision_score(true,pred, average='micro')
#recall_score(true,pred, average='macro')

#comentario = ['El presidente en alocución oficial pronunció un discurso de odio y terror para toda la población, incitando al asesinato de todo aquel de la oposición y declarando que la única postura válida en el país era la ultraderecha.']

#comentario = t.texts_to_sequences(comentario)


#comentario = pad_sequences(comentario, padding = 'pre',maxlen=max(news_num1, news_num2))

#np.argmax(mod.predict(comentario), axis=-1)
