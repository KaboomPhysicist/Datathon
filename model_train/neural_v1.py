from keras.models import Sequential
from keras.models import load_model
from keras import layers
from keras.backend import clear_session

import matplotlib.pyplot as plt
import numpy as np


from extract_split_data import vectorized_set, plot_history

#Modelos de redes neuronales basados en CountVectorizer con sesgo y gravedad separados

def create_model(input_dim):
    #Declaración del modelo de Gravedad
    model = Sequential()
    model.add(layers.Dense(10, input_dim = input_dim, activation='relu'))
    model.add(layers.Dense(4, activation='softmax'))

    #sparse_categorical_crossentropy se usa para las clasificaciones de varias categorías
    model.compile(loss='sparse_categorical_crossentropy',
                    optimizer='adam',
                    metrics=['accuracy'])
    return model

def create_model2(input_dim):
    #Declaración del modelo de Gravedad
    model = Sequential()
    model.add(layers.Dense(10, input_dim = input_dim, activation='relu'))
    model.add(layers.Dense(3, activation='softmax'))

    #sparse_categorical_crossentropy se usa para las clasificaciones de varias categorías
    model.compile(loss='sparse_categorical_crossentropy',
                    optimizer='adam',
                    metrics=['accuracy'])
    return model


def train_neural_vectorizer(graph=False):

    vectorizer, X_grav_train, X_grav_test, X_ses_train, X_ses_test, grav_train, grav_test, ses_train, ses_test = vectorized_set()

    input_dim = X_grav_train.shape[1]

    #Declaración del modelo de Gravedad
    model = create_model(input_dim)

    #Declaración del modelo de Sesgo
    model2 = create_model2(input_dim)

    clear_session()

    #Entrenamiento de los modelos. En el caso del modelo para el sesgo, se le suma 1 a los valores de sesgo para que el rango sea [0,1,2] en lugar de [-1,0,1].
    #Keras no admite este último rango.

    history = model.fit(X_grav_train, grav_train,
                            epochs=50,
                            verbose=False,
                            validation_data=(X_grav_test, grav_test),
                            batch_size=10
        )

    history2 = model2.fit(X_ses_train, ses_train+1,
                            epochs=50,
                            verbose=False,
                            validation_data=(X_ses_test, ses_test+1),
                            batch_size=10
        )

    loss, accuracy = model.evaluate(X_grav_train, grav_train, verbose=False)
    print("Precisión de entrenamiento (Gravedad): {:.4f}".format(accuracy))
    loss, accuracy = model.evaluate(X_grav_test, grav_test, verbose=False)
    print("Precisión de prueba (Gravedad):  {:.4f}".format(accuracy))

    loss, accuracy = model2.evaluate(X_ses_train, ses_train+1, verbose=False)
    print("Precisión de entrenamiento (Sesgo): {:.4f}".format(accuracy))
    loss, accuracy = model2.evaluate(X_ses_test, ses_test+1, verbose=False)
    print("Precisión de prueba (Sesgo):  {:.4f}".format(accuracy))

    if graph:
        plot_history(history)
        plot_history(history2)
        plt.show()
    
    model.save('models/neural_v1_grav.h5')
    model2.save('models/neural_v1_ses.h5')

def modelo(pers_test):
    model_grav = load_model('models/neural_v1_grav.h5')
    model_ses = load_model('models/neural_v1_ses.h5')

    vectorizer = vectorized_set(True)

    test = vectorizer.transform(np.array([pers_test]))
    print(model_grav.predict(test),model_ses.predict(test))
