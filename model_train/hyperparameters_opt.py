from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import RandomizedSearchCV

import neural_v1 as n1
import neural_v2 as n2
import neural_v3 as n3

from extract_split_data import vectorized_set, data_preset


modelos = {'neural_v2' : (n2,data_preset,dict(vocab_size=[5000], embedding_dim=[80,100,200], maxlen=[250])),
           'neural_v3' : (n3,data_preset,dict(vocab_size=[5000], embedding_dim=[80,100,200], maxlen=[100,200,250]))}

epochs = 100
output_file = 'model_train/performance/output.txt'

for net in modelos:
    print("Modelo usado:", net)
    param_grid = modelos[net][2]
    if net=='neural_v2':
        
        tokenizer, tokenizer2, X_grav_train, X_grav_test, X_ses_train, X_ses_test, grav_train, grav_test, ses_train, ses_test = data_preset(250, True)

        model = KerasClassifier(build_fn = modelos[net][0].create_model, epochs = epochs,
                                batch_size = 10, verbose=False)
                
        grid = RandomizedSearchCV(estimator=model, param_distributions=param_grid, cv=4, verbose=False, n_iter=5)

        grid_result = grid.fit(X_grav_train, grav_train)
        test_acc = grid.score(X_grav_test, grav_test)

        # Save and evaluate results
        with open(output_file, 'a') as f:
            s = ('Modelo: {}\nBest Accuracy : '
                '{:.4f}\n{}\nTest Accuracy : {:.4f}\n\n')
            output_string = s.format(
                net,
                grid_result.best_score_,
                grid_result.best_params_,
                test_acc)
            print(output_string)
            f.write(output_string)
