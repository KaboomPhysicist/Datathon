from matplotlib import pyplot as plt

from extract_split_data import create_embedding_matrix, data_preset, pad, plot_history

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping
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

class TokenAndPositionEmbedding(layers.Layer):
    def __init__(self, maxlen, vocab_size, embed_dim):
        super(TokenAndPositionEmbedding, self).__init__()
        self.token_emb = layers.Embedding(input_dim=vocab_size, output_dim=embed_dim)
        self.pos_emb = layers.Embedding(input_dim=maxlen, output_dim=embed_dim)

    def call(self, x):
        maxlen = tf.shape(x)[-1]
        positions = tf.range(start=0, limit=maxlen, delta=1)
        positions = self.pos_emb(positions)
        x = self.token_emb(x)
        return x + positions

vocab_size = 5000  # Only consider the top 20k words
maxlen = 200  # Only consider the first 200 words of each movie review

tokenizer, tokenizer2, X_grav_train, X_grav_test, X_ses_train, X_ses_test, grav_train, grav_test, ses_train, ses_test = data_preset(train=True)
X_grav_train, X_grav_test, X_ses_train, X_ses_test = pad(X_grav_train, X_grav_test, X_ses_train, X_ses_test, maxlen)


embed_dim = 300  # Embedding size for each token
num_heads = 2  # Number of attention heads
ff_dim = 32  # Hidden layer size in feed forward network inside transformer


#embedding_layer = TokenAndPositionEmbedding(maxlen, vocab_size, embed_dim)

embedding_path = '../embeddings/embeddings-l-model.vec'

vocab_size = len(tokenizer.word_index) + 1
embedding_matrix = create_embedding_matrix(embedding_path,tokenizer.word_index, embed_dim)
embedding_layer = layers.Embedding(input_dim=vocab_size, output_dim=embed_dim, weights=[embedding_matrix],input_length = maxlen, trainable=True)

neurons = 40

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

es=EarlyStopping(monitor='val_loss',patience=50, restore_best_weights=True)

history = model.fit(X_grav_train, grav_train,
                    batch_size=128,
                    epochs=500,
                    validation_data=(X_grav_test, grav_test),
                    callbacks=[es])

loss, accuracy = model.evaluate(X_grav_train, grav_train, verbose=False)
print("Precisión de entrenamiento (Gravedad): {:.4f}".format(accuracy))
loss, accuracy = model.evaluate(X_grav_test, grav_test, verbose=False)
print("Precisión de prueba (Gravedad):  {:.4f}".format(accuracy))

plot_history(history)
plt.show()