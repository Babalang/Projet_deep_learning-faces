import os

# Réduire les logs
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# TF XLA pose problème si ptxas absent → on le désactive
os.environ.pop("TF_XLA_FLAGS", None)

import tensorflow as tf

# Activer le memory growth pour éviter les grosses allocs qui plantent
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"[GPU OK] {len(gpus)} GPU détecté(s).")
    except Exception as e:
        print("Erreur config GPU :", e)
else:
    print("[CPU] Aucun GPU détecté par TensorFlow.")

import numpy as np
import pandas as pd

from keras.layers import Input, Conv2D, LeakyReLU, Dropout, BatchNormalization, Flatten, Dense, Embedding, Reshape, Activation, Concatenate, Conv2DTranspose, MaxPooling2D, UpSampling2D
from keras.models import Model
from keras.optimizers import Adam
from numpy.random import randn, randint
from matplotlib import pyplot as plt
from keras.models import save_model
from math import sqrt
from numpy import asarray
from numpy.random import randn
from keras.models import load_model

emotion_labels = {
        0: "Angry", 1: "Disgust", 2: "Fear", 3: "Happy",
        4: "Sad", 5: "Surprise", 6: "Neutral"
    }

def load_fer2013_dataset():
    data = pd.read_csv('./test_deepFace/modules/fer2013.csv')
    pixels = data['pixels'].values
    emotions = data['emotion'].values
    X, y = [], []
    for pixel_sequence, emotion in zip(pixels, emotions):
        pv = pixel_sequence.split()
        if len(pv) != 2304:
            continue
        arr = np.array(pv, dtype=np.uint8).reshape(48, 48, 1).astype('float32') / 255.0
        arr = arr * 2.0 - 1.0              # [-1,1] pour tanh
        X.append(arr)
        y.append(emotion)
    X = np.array(X, dtype='float32')
    y = np.array(y, dtype='int32')
    print("Images:", X.shape, "Labels:", y.shape)
    print("Min pixel:", X.min(), "Max pixel:", X.max())  # <-- Ajout
    print("Unique labels:", np.unique(y))                # <-- Ajout
    return (X, y)

def define_discriminator(input_shape=(48, 48, 1), n_classes=7):
    input_image = Input(shape=input_shape)
    x = Conv2D(256, 3, padding='same')(input_image)
    x = LeakyReLU(alpha=0.2)(x)
    x = Conv2D(256, 4, strides=2, padding='same')(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = Conv2D(256, 4, strides=2, padding='same')(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = Conv2D(256, 4, strides=2, padding='same')(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = Conv2D(256, 4, strides=2, padding='same')(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = Flatten()(x)
    x = Dropout(0.4)(x)
#    x = Dense(128, activation='relu')(x)
    out1 = Dense(1, activation='sigmoid', name="fake_real")(x)
    out2 = Dense(n_classes, activation='softmax', name="emotion")(x)
    model = Model(input_image, [out1, out2])
    opt = Adam(learning_rate=0.0003, beta_1=0.5)
    model.compile(
        loss=['binary_crossentropy', 'sparse_categorical_crossentropy'],
        optimizer=opt,
        metrics=['accuracy', 'accuracy']
    )
    return model
d=define_discriminator()
d.summary()

def define_generator(latent_dim, n_classes=7):
    input_label = Input(shape=(1,))
    li = Embedding(n_classes, 50)(input_label)
    n_nodes = 12 * 12
    li = Dense(n_nodes)(li)
    li = Reshape((12, 12, 1))(li)
    input_lat = Input(shape=(latent_dim,))
    n_nodes = 384 * 12 * 12
    gen = Dense(n_nodes)(input_lat)
    gen = Activation('relu')(gen)
    gen = Reshape((12, 12, 384))(gen)
    merge = Concatenate()([gen, li])
    # Première couche upsampling
    gen = Conv2DTranspose(192, (5,5), strides=(2,2), padding='same')(merge)
    gen = BatchNormalization()(gen)
    gen = Activation('relu')(gen)
    # Deuxième couche upsampling
    gen = Conv2DTranspose(96, (5,5), strides=(2,2), padding='same')(gen)
    gen = BatchNormalization()(gen)
    gen = Activation('relu')(gen)
    # Troisième couche upsampling (ajoutée)
    gen = Conv2DTranspose(48, (5,5), strides=(1,1), padding='same')(gen)
    gen = BatchNormalization()(gen)
    gen = Activation('relu')(gen)
    # Couche de sortie
    out_layer = Conv2DTranspose(1, (5,5), strides=(1,1), padding='same', activation='tanh')(gen)
    model = Model([input_lat, input_label], out_layer)
    return model

# Remplace define_generator par une version UNet
def define_unet_generator(latent_dim, n_classes=7):
    input_label = Input(shape=(1,))
    li = Embedding(n_classes, 50)(input_label)
    li = Dense(12 * 12)(li)
    li = Reshape((12, 12, 1))(li)

    input_lat = Input(shape=(latent_dim,))
    x = Dense(128 * 12 * 12)(input_lat)
    x = LeakyReLU(alpha=0.2)(x)
    x = Reshape((12, 12, 128))(x)

    merge = Concatenate()([x, li])

    # Ajout de plusieurs couches d'upsampling
    x = Conv2D(256, 5, padding='same')(merge)
    x = LeakyReLU(alpha=0.2)(x)
    x = Conv2DTranspose(256, 4, strides=2, padding='same')(x)   # 12x12 -> 24x24
    x = LeakyReLU(alpha=0.2)(x)
    x = Conv2D(128, 5, padding='same')(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = Conv2DTranspose(128, 4, strides=2, padding='same')(x)   # 24x24 -> 48x48
    x = LeakyReLU(alpha=0.2)(x)
    x = Conv2D(64, 5, padding='same')(x)
    x = LeakyReLU(alpha=0.2)(x)
    out_layer = Conv2D(1, 7, activation='tanh', padding='same')(x)  # 48x48x1

    model = Model([input_lat, input_label], out_layer)
    return model
g=define_unet_generator(latent_dim=100)
g.summary()

def define_gan(g_model, d_model):
    d_model.trainable = False
    gan_output = d_model(g_model.output)
    model = Model(g_model.input, gan_output)
    opt = Adam(learning_rate=0.0002, beta_1=0.5)
    model.compile(loss=['binary_crossentropy', 'sparse_categorical_crossentropy'], optimizer=opt)
    return model

# use the generator to generate n fake examples, with class labels
def generate_fake_samples(generator, latent_dim, n_samples, n_classes=7):
    z_input = np.random.randn(latent_dim * n_samples)
    z_input = z_input.reshape(n_samples, latent_dim)
    labels = np.random.randint(0, n_classes, n_samples).reshape(-1, 1)
    X_fake = generator.predict([z_input, labels], verbose=0)
    return X_fake, labels


def train(g_model, d_model, gan_model, dataset, latent_dim, n_epochs, n_batch):
    X_train, y_train = dataset
    #bat_per_epo = int(X_train.shape[0] / n_batch)
    bat_per_epo = 256
    for epoch in range(1, n_epochs + 1):
        print(f"Epoch {epoch}/{n_epochs}")

        for batch in range(bat_per_epo):
            d_model.trainable = True
            ix = np.random.randint(0, X_train.shape[0], n_batch)
            X_real, labels_real = X_train[ix], y_train[ix]
            y_real = np.ones((n_batch, 1))
            d_loss_real = d_model.train_on_batch(X_real, [y_real, labels_real])

            X_fake, labels_fake = generate_fake_samples(g_model, latent_dim, n_batch)
            y_fake = np.zeros((n_batch, 1))
            d_loss_fake = d_model.train_on_batch(X_fake, [y_fake, labels_fake])

            d_model.trainable = False

            z_input = np.random.randn(n_batch * latent_dim)
            z_input = z_input.reshape(n_batch, latent_dim)
            z_labels = np.random.randint(0, 7, n_batch).reshape(-1, 1)
            y_gan = np.ones((n_batch, 1))
            g_loss = gan_model.train_on_batch([z_input, z_labels], [y_gan, z_labels])

            print(
                f"  Batch {batch+1}/{bat_per_epo} | "
                f"D_real: loss={d_loss_real[0]:.3f}, "
                f"FAKE/REAL_acc={d_loss_real[1]:.3f}, EMO_acc={d_loss_real[4]:.3f} | "
                f"D_fake: loss={d_loss_fake[0]:.3f}, "
                f"FAKE/REAL_acc={d_loss_fake[1]:.3f}, EMO_acc={d_loss_fake[4]:.3f} | "
                f"G_loss={g_loss[0]:.3f}"
            )

        # Save generated images every few epochs
        if epoch % 10 == 0:
            examples = 7  # une image par label
            latent_dim = 100
            latent_points = np.random.randn(examples * latent_dim).reshape(examples, latent_dim)
            labels = np.arange(7).reshape(-1, 1)  # 0 à 6, une fois chacun
            X_fake = g_model.predict([latent_points, labels])
            plt.figure(figsize=(14, 2))
            for i in range(examples):
                plt.subplot(1, examples, i + 1)
                plt.imshow(X_fake[i].reshape(48, 48), cmap='gray')
                plt.title(emotion_labels[labels[i][0]], fontsize=10)
                plt.axis('off')
            plt.tight_layout()
            filename = f'generated_epoch2_{epoch:04d}.png'
            plt.savefig(filename, bbox_inches='tight')
            plt.close()
     

def run_cgan():
    dataset = load_fer2013_dataset()
    X_train, y_train = dataset
    d_model = define_discriminator(input_shape=(48, 48, 1))
    g_model = define_unet_generator(latent_dim=100)
    gan_model = define_gan(g_model, d_model)
    train(g_model, d_model, gan_model, (X_train, y_train), latent_dim=100, n_epochs=100, n_batch=64)
    save_model(d_model, 'discriminator_model.h5')  # Save discriminator model
    # Save generator model
    save_model(g_model, 'generator_model.h5')  # Save generator model
    # Save GAN model (including both generator and discriminator)
    save_model(gan_model, 'acgan_model.h5')  # Save CGAN model

def generate_emotion_image(latent_vector, emotion_label):
    g_model = load_model('generator_model.h5', compile=False)
    latent_vector = np.expand_dims(latent_vector, axis=0)
    emotion_label = np.array([[emotion_label]])
    generated_image = g_model.predict([latent_vector, emotion_label])
    generated_image = (generated_image + 1.0) / 2.0  # Remettre en [0,1]
    return generated_image[0]
if __name__ == "__main__":
    print("=== ACGAN ===")
    print("Voulez-vous entraîner le modèle ACGAN ou générer une image ?")
    choice = input("Entrez 'train' pour entraîner ou 'generate' pour générer une image: ").strip().lower()
    if choice == 'train':
        run_cgan()
    elif choice == 'generate':
        latent_vector = np.random.randn(100)
        emotion_label = int(input("Entrez une étiquette d'émotion (0 = colère, 1 = dégoût, 2 = peur, 3 = bonheur, 4 = tristesse, 5 = surprise, 6 = neutre): "))
        generated_image = generate_emotion_image(latent_vector, emotion_label)
        import matplotlib.pyplot as plt
        plt.imshow(generated_image.reshape(48,48), cmap='gray')
        plt.axis('off')
        plt.title(f'Émotion: {emotion_label}')
        plt.show()
     