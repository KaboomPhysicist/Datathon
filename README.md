# Datathon

## English

This project is the proposed solution to a small competitiion named Datathon. The main objective is creating a Deep Learning Model for text classification, specifically, the model must give a value for **political bias** and **seriousness** given a paragraph news.
The problem is originally born from Colombian strike in 2021, so the news are in spanish and focused to the demonstrations and statements from public forms in the media about this topic.

The original dataset used for training is saved in data/clasificacion.csv. This dataset has been made by all the integrants of the competition. Each group gave 50 paragraph news and scored 100 for bias (Sesgo) and seriousness (Gravedad). 

There are three possible values for bias:
* -1 for negative bias. Against the government.
* 0 for impartiaity.
* 1 for positive bias. In favor of government.

There are four possible values for seriousness:
* 0 for no serious.
* 1 for lightly serious.
* 2 for serious.
* 3 for very serious.

If you only want to use the final trained model that has been chosen for evaluating a paragraph, you can download the final_model/ folder which contains a Notebook with instructions to calculate the confussion matrix and some functions to make predictions. You can too execute the file neural_v3.py and it will require you a paragraph for testing. This file has other options and functions, but they must be called inside the file if you want to use them.

The main files in this repository are in model_train, these are neural_v3.py and Augmented Em-pre_DNN.ipynb. The files gravedad.py and sesgo.py are limited no-notebook versions of Augmented Em-pre_DNN.ipynb.
The file neural_v3.py contains a model for seriousness and a model for bias, besides it has functions for training these models and for loading other models. Every generated model with this script is saved in models/ (Gravedad/ for seriousness and Sesgo/ for bias).
All the models use tensorflow and keras for creating the Neural Network and includes a embedding layer for using a pretrained embedding.

The data/ folder contains the main dataset and some variations of this one. The variations has been done by data augmentation, using libraries like nlpaug or BackTranslation. The scripts in data_augmentation/ folder generated these datasets, but they were not used, since the resulting data were poor and the models trained with these had a worse performance.

The models/ folder contains all the generated models from neural_v3.py and Augmented Em-pre_DNN.ipynb. There are over 100 models and they can be classified in groups according to the parameters set in the script. The information about these parameters has not been saved, but you can check the performance of the model by the confussion matrix and the accuracy and loss graphs in the performance/ folder.

The drive/ folder contains a file for autodownloading the dataset. This was useful when the dataset was changing, but the dataset doesn't change anymore, so you can ignore this file. However, if you want to use this file for downloading another dataset, you must change the id in the file and provide a client_secrets.json in the same folder (for more information about client_secrets.json you can check: https://developers.google.com/identity/protocols/oauth2


### Important advice
If you want to use any of the main files for training with any dataset you must make a new directory in the main folder named embeddings/ and you must place at least one spanish embedding (and modify the path in the scripts). We have used the embeddings from this repository: https://github.com/dccuchile/spanish-word-embeddings

