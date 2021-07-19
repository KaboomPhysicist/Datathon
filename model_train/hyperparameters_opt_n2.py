from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV

import neural_v2 as n2

from extract_split_data import data_preset, pad

def hyperoptimization(epochs,param_grid,type='random'):
    output_file = 'performance/output_neural_v2.txt'

    #param_grid = dict(vocab_size=[1000, 2000, 4000,5000], embedding_dim=[80,100,200,250], maxlen=[250])
        
    tokenizer, tokenizer2, X_grav_train, X_grav_test, X_ses_train, X_ses_test, grav_train, grav_test, ses_train, ses_test = data_preset(True)
    X_grav_train, X_grav_test, X_ses_train, X_ses_test = pad(X_grav_train, X_grav_test, X_ses_train, X_ses_test,250)

    model = KerasClassifier(build_fn = n2.create_model, epochs = epochs,
                            batch_size = 10, verbose=False)

    if type=='random':        
        grid = RandomizedSearchCV(estimator=model, param_distributions=param_grid, cv=4, verbose=False, n_iter=5)
    else:
        grid = GridSearchCV(estimator=model, param_grid=param_grid, cv=4, verbose=1, n_jobs=-1)

    grid_result = grid.fit(X_grav_train, grav_train)
    test_acc = grid.score(X_grav_test, grav_test)

    # Save and evaluate results
    with open(output_file, 'a') as f:
        f.write("Matriz de parámetros:{}".format(param_grid))
        f.write("Tipo:{}".format(type))
        s = ('Modelo (Gravedad): {}\nBest Accuracy : '
            '{:.4f}\n{}\nTest Accuracy : {:.4f}\n\n')
        output_string = s.format(
            'Red Neuronal V2',
            grid_result.best_score_,
            grid_result.best_params_,
            test_acc)
        print(output_string)
        f.write(output_string)

    model = KerasClassifier(build_fn = n2.create_model2, epochs = epochs,
                            batch_size = 10, verbose=False)

    if type=='random':        
        grid = RandomizedSearchCV(estimator=model, param_distributions=param_grid, cv=4, verbose=False, n_iter=5, n_jobs=-1)
    else:
        grid = GridSearchCV(estimator=model, param_grid=param_grid, cv=4, verbose=1, n_jobs=-1)

    grid_result = grid.fit(X_ses_train, ses_train)
    test_acc = grid.score(X_ses_test, ses_test)

    # Save and evaluate results
    with open(output_file, 'a') as f:
    
        s = ('Modelo (Sesgo): {}\nBest Accuracy : '
            '{:.4f}\n{}\nTest Accuracy : {:.4f}\n\n')
        output_string = s.format(
            'Red Neuronal V2',
            grid_result.best_score_,
            grid_result.best_params_,
            test_acc)
        print(output_string)
        f.write(output_string)
        f.write('\n------------------------------------------------------------------------------------------------------\n')

#create_model(vocab_size, embedding_dim, maxlen)
#hyperoptimization(100,dict(vocab_size=[1000, 2000, 4000,5000], embedding_dim=[80,100,200,250], maxlen=[250]))