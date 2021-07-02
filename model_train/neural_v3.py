from extract_split_data import create_embedding_matrix, plot_history, data_preset

import matplotlib.pyplot as plt
import numpy as np

from keras.models import Sequential
from keras import layers
from keras.backend import clear_session

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

maxlen = 250

tokenizer, tokenizer2, X_grav_train, X_grav_test, X_ses_train, X_ses_test, grav_train, grav_test, ses_train, ses_test = data_preset(maxlen, train=True)

vocab_size = len(tokenizer.word_index) + 1

embedding_dim = 100
embedding_matrix = create_embedding_matrix('embeddings/embeddings-l-model.vec', tokenizer.word_index, embedding_dim)

model = Sequential()
model2 = Sequential()

model.add(layers.Embedding(
    input_dim = vocab_size,
    output_dim = embedding_dim,
    weights =[embedding_matrix],
    input_length= maxlen,
    trainable = True
))

model.add(layers.GlobalMaxPooling1D())
model.add(layers.Dense(10, activation='relu'))
model.add(layers.Dense(4, activation='softmax'))

model2.add(layers.Embedding(
    input_dim = vocab_size,
    output_dim = embedding_dim,
    weights =[embedding_matrix],
    input_length= maxlen,
    trainable = True
))
model2.add(layers.GlobalMaxPooling1D())
model2.add(layers.Dense(10, activation='relu'))
model2.add(layers.Dense(3, activation='softmax'))

model.compile(optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy']
        )
model2.compile(optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy']
                )

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


plot_history(history)
plot_history(history2)
plt.show()

model.save('models/neural_v3_grav.h5')
model2.save('models/neural_v3_ses.h5')