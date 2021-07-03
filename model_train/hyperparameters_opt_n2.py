from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV

import neural_v2 as n2

from extract_split_data import data_preset

def hyperoptimization(epochs,param_grid,type='random'):
    output_file = 'model_train/performance/output_neural_v2.txt'

    #param_grid = dict(vocab_size=[1000, 2000, 4000,5000], embedding_dim=[80,100,200,250], maxlen=[250])
        
    tokenizer, tokenizer2, X_grav_train, X_grav_test, X_ses_train, X_ses_test, grav_train, grav_test, ses_train, ses_test = data_preset(250, True)

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
        s = ('Modelo: {}\nBest Accuracy : '
            '{:.4f}\n{}\nTest Accuracy : {:.4f}\n\n')
        output_string = s.format(
            'Red Neuronal V2',
            grid_result.best_score_,
            grid_result.best_params_,
            test_acc)
        print(output_string)
        f.write(output_string)
