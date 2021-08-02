# Datathon

## Español

Este proyecto es la solución propuesta a una pequeña competencia interna llamada Datathon. El principal objetivo era crear un modelo de Deep Learning para clasificación de texto, más específicamente, para dar valores de sesgo (político) y gravedad a párrafos noticiosos.
La competencia nace en medio del Paro Nacional de Colombia en 2021, por lo tanto las noticias a trabajar están enfocadas en los hechos ocurridos en las manifestaciones y en las declaraciones de figuras públicas en los medios.

El conjunto de datos original usado está en el documento clasificacion.csv en la carpeta data/. Este conjunto de datos fue construido con la ayuda de cada uno de los participantes de la competencia, donde cada grupo introdujo 50 párrafos de noticias y los calificó, además de calificar otros 50 de otros equipos. Para seleccionar el valor de gravedad y sesgo se acordó escoger la moda entre los datos dados.

Hay 3 posibles valores para sesgo:
* -1 para sesgo negativo. En contra del gobierno.
* 0 para imparcialidad.
* 1 para sesgo positivo. A favor del gobierno.

Hay 4 posibles valores para gravedad:
* 0 para Sin Gravedad
* 1 para Levemente Grave
* 2 para Grave
* 3 para Muy Grave

Si solo desea evaluar una noticia con el modelo final entrenado que hemos escogido para la competencia, puede descargar la carpeta final_model/, la cual contiene un Notebook de Jupyter con instrucciones para calcular la matriz de confusión dado un conjunto de datos, y una función para hacer las predicciones con tan solo ingresar el párrafo noticioso. Alternativamente, puede descargar el archivo neural_v3.py de la carpeta model_train/, el cual solicitará directamente la noticia a evaluar. Para usar este último es necesario descargar la carpeta models/. El archivo contiene otras opciones y funciones, pero estas deben de ser cambiadas o invocadas dentro del propio archivo para poder ser usadas.

Los principales archivos de este repositorio se encuentran en model_train. Estos son neural_v3.py y Augmented Em-pre_DNN.ipynb. Los archivos gravedad.py y sesgo.py son versiones en forma de script del archivo Augmented Em-pre_DNN.ipynb. Ambos archivos son los usados para el entrenamiento de las redes neuronales. Los modelos contenidos en models/ corresponden a los generados por neural_v3.py a lo largo de las pruebas.

El archivo neural_v3.py contiene un modelo para gravedad y uno para sesgo. Adicionalmente, contiene funciones para generar gráficas del rendimiento y de la matriz de confusión, y también hay funciones para la implementación del modelo como se mencionó previamente.
Todos los modelos (tanto de neural_v3.py como de Augmented Em-pre_DNN.ipynb) usan tensorflow y keras para la red neural, e incluyen una capa de embedding para usar un embedding preentrenado.

La carpeta data/ contiene el conjunto de datos original, el conjunto de datos usado (con la moda identificada), y algunas modificaciones del original. Las variaciones fueron hechas con data augmentation, haciendo uso de librerías como nlpaug y BackTranslation. Los scripts en la carpeta data_augmentation contienen las funciones encargadas de generar estos conjuntos de datos modificados. Lastimosamente, estos conjuntos de datos no fueron usados para la generación de los modelos, puesto que estos estaban teniendo peor rendimiento.

La carpeta models/ contiene todos los modelos generados con el archivo neural_v3.py, y contiene algunas pocas pruebas de Augmented Em-pre_DNN.ipynb. Hay más de 100 modelos y se pueden dividir en grupos dependiendo de los parámetros que se establecieron para la generación en el script. Estos parámetros no están guardados en ninguna parte (el mensaje del commit original contenía algo de información al respecto), sin embargo, se puede verificar el rendimiento del modelo mediante la matriz de confusión y los gráficos de precisión y pérdida, guardados en la carpeta performance/.

La carpeta drive/ contiene un script con una función para descargar automáticamente el conjunto de datos original. La función fue usada cuando el conjunto de datos estaba creciendo, pero como la competencia acabó, este conjunto de datos ya no aumenta más, por lo que el archivo ya no tiene utilidad. No obstante, si usted desea usar la función para descargar automáticamente otro conjunto de datos, debe ubicar un archivo llamado client_secrets.json con una id válida de autenticación de Google. El archivo usado originalmente ha sido eliminado y la id ha sido cambiada por seguridad. Para más información acerca de este archivo puede revisar: https://developers.google.com/identity/protocols/oauth2

### Anuncio importante
Si desea usar cualquiera de los archivos principales para entrenar con cualquier conjunto de datos, debe crear una nueva carpeta en el directorio central del repositorio llamada embeddings, y debe descargar en esta carpeta al menos un embedding en español (y modificar la ruta en las funciones correspondientes, usualmente referido como embedding_path). Para este proyecto hemos usado los embeddings que se presentan en este repositorio: https://github.com/dccuchile/spanish-word-embeddings



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
