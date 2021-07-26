from matplotlib import pyplot as plt

from extract_split_data import create_embedding_matrix, data_preset, pad, plot_history

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.backend import clear_session


class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super(TransformerBlock, self).__init__()
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = keras.Sequential(
            [layers.Dense(ff_dim, activation="relu"), layers.Dense(embed_dim),]
        )
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def call(self, inputs, training):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)


def transformer_model(tokenizer, neurons = 60, embedding_path = '../embeddings/embeddings-l-model.vec'):
    
    vocab_size = 5000  # Only consider the top 5k words
    maxlen = 300  # Only consider the first 200 words of each movie review

    embed_dim = 300  # Embedding size for each token
    num_heads = 10  # Number of attention heads
    ff_dim = 40 # Hidden layer size in feed forward network inside transformer

    vocab_size = len(tokenizer.word_index) + 1
    embedding_matrix = create_embedding_matrix(embedding_path,tokenizer.word_index, embed_dim)
    embedding_layer = layers.Embedding(input_dim=vocab_size, output_dim=embed_dim, weights=[embedding_matrix],input_length = maxlen, trainable=True)

    inputs = layers.Input(shape=(maxlen,))
    x = embedding_layer(inputs)
    transformer_block = TransformerBlock(embed_dim, num_heads, ff_dim)
    x = transformer_block(x)
    x = layers.Conv1D(128, 5, activation='relu')(x)
    x = layers.GlobalMaxPool1D()(x)
    x = layers.Dense(neurons, activation='relu', kernel_initializer='he_uniform', kernel_regularizer=tf.keras.regularizers.l1(0.005),bias_regularizer='l1')(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(neurons, activation='relu', kernel_initializer='he_uniform', kernel_regularizer=tf.keras.regularizers.l1(0.005),bias_regularizer='l1')(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(neurons, activation='relu', kernel_initializer='he_uniform', kernel_regularizer=tf.keras.regularizers.l1(0.005),bias_regularizer='l1')(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(neurons, activation='relu', kernel_initializer='he_uniform', kernel_regularizer=tf.keras.regularizers.l1(0.005),bias_regularizer='l1')(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(neurons, activation='relu', kernel_initializer='he_uniform', kernel_regularizer=tf.keras.regularizers.l1(0.005),bias_regularizer='l1')(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(neurons, activation='relu', kernel_initializer='he_uniform', kernel_regularizer=tf.keras.regularizers.l1(0.005),bias_regularizer='l1')(x)
    outputs = layers.Dense(4, activation='softmax')(x)


    model = keras.Model(inputs=inputs, outputs=outputs)

    model.compile("adam", "sparse_categorical_crossentropy", metrics=["accuracy"])
    model.summary()

    return model

maxlen = 200

tokenizer, tokenizer2, X_grav_train, X_grav_test, X_ses_train, X_ses_test, grav_train, grav_test, ses_train, ses_test = data_preset(train=True, augment=False)
X_grav_train, X_grav_test, X_ses_train, X_ses_test = pad(X_grav_train, X_grav_test, X_ses_train, X_ses_test, maxlen)


es=EarlyStopping(monitor='val_loss', patience=50, restore_best_weights=True)

clear_session()

model = transformer_model(neurons = i, tokenizer = tokenizer)
history = model.fit(X_grav_train, grav_train,
                    batch_size=128,
                    epochs=1000,
                    validation_data=(X_grav_test, grav_test),
                    callbacks=[es],
                    verbose = False)

loss, accuracy = model.evaluate(X_grav_train, grav_train, verbose=False)
print("Precision de entrenamiento (Gravedad): {:.4f}".format(accuracy))
loss, accuracy = model.evaluate(X_grav_test, grav_test, verbose=False)
print("Precisión de prueba (Gravedad):  {:.4f}".format(accuracy))

plot_history(history)
plt.show()
