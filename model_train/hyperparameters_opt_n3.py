from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV

import neural_v3 as n3

from extract_split_data import data_preset, pad

#param_grid : tokenizer, embedding_dim, embedding_path, maxlen

#Función para optimizar los hiperparámetros del modelo de red neuronal de neural_v3.py
def hyperoptimization(epochs, param_grid, type = 'random'):
    #Archivo donde se guardaran los resultados de parámetros óptimos
    output_file = 'performance/output_neural_v3.txt'

    maxlen = 250

    tokenizer, tokenizer2, X_grav_train, X_grav_test, X_ses_train, X_ses_test, grav_train, grav_test, ses_train, ses_test = data_preset(train=True, descarga=False)
    X_grav_train, X_grav_test, X_ses_train, X_ses_test = pad(X_grav_train, X_grav_test, X_ses_train, X_ses_test, maxlen)

    #A la matriz de parámetros a probar se le añade el tokenizer y el maxlen, los cuales son únicos bajo este código y se establecen dentro del mismo.
    param_grid.update(dict(tokenizer = [tokenizer],maxlen = [maxlen]))
    
    #Declaración del modelo para gravedad
    model = KerasClassifier(build_fn = n3.create_model, epochs = epochs,
                            batch_size = 64, verbose=True)

    #Elección del método para buscar los hiperparámetros
    if type == 'random':
        grid = RandomizedSearchCV(estimator=model, param_distributions=param_grid, cv=4, verbose=False, n_iter=5, n_jobs=-1)
    else:
        grid = GridSearchCV(estimator=model, param_grid=param_grid, cv=4, verbose=False,n_jobs=2)


    grid_result = grid.fit(X_grav_train, grav_train)
    test_acc = grid.score(X_grav_test, grav_test)

    # Save and evaluate results
    with open(output_file, 'a') as f:
        f.write("Matriz de parámetros:{}\n".format(param_grid))
        f.write("Tipo:{}\n\n".format(type))
        s = ('Modelo (Gravedad): {}\nBest Accuracy : '
            '{:.4f}\n{}\nTest Accuracy : {:.4f}\n\n')
        output_string = s.format(
            'Red Neuronal V3',
            grid_result.best_score_,
            grid_result.best_params_,
            test_acc)
        print(output_string)
        f.write(output_string)

    model = KerasClassifier(build_fn = n3.create_model2, epochs = epochs,
                            batch_size = 64, verbose=False)

    #Cambia el tokenizer (se supone que es diferente para sesgo y para gravedad)
    param_grid.update(dict(tokenizer = [tokenizer2]))

    if type == 'random':
        grid = RandomizedSearchCV(estimator=model, param_distributions=param_grid, cv=4, verbose=False, n_iter=5, n_jobs=-1)
    else:
        grid = GridSearchCV(estimator=model, param_grid=param_grid, cv=4, verbose=False, n_jobs=2)


    grid_result = grid.fit(X_ses_train, ses_train)
    test_acc = grid.score(X_ses_test, ses_test)

    # Save and evaluate results
    with open(output_file, 'a') as f:
        s = ('Modelo (Sesgo): {}\nBest Accuracy : '
            '{:.4f}\n{}\nTest Accuracy : {:.4f}\n')
        output_string = s.format(
            'Red Neuronal V3',
            grid_result.best_score_,
            grid_result.best_params_,
            test_acc)
        print(output_string)
        f.write(output_string)
        f.write('\n------------------------------------------------------------------------------------------------------\n')

if __name__=='__main__':
    #'../embeddings/embeddings-l-model.vec','../embeddings/fasttext-sbwc.vec','../embeddings/SBW-vectors-300-min5.txt','../embeddings/glove-sbwc.i25.vec'
    hyperoptimization(150, dict(embedding_path = ['../embeddings/cbow.vec'],embedding_dim = [200,300,400]),type='random')
