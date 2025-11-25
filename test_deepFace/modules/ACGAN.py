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

from keras.layers import Input, Conv2D, LeakyReLU, Dropout, BatchNormalization, Flatten, Dense, Embedding, Reshape, Activation, Concatenate, Conv2DTranspose
from keras.models import Model
from keras.optimizers import Adam
from numpy.random import randn, randint
from matplotlib import pyplot as plt
from keras.models import save_model
from math import sqrt
from numpy import asarray
from numpy.random import randn
from keras.models import load_model

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
    fe = Conv2D(32, (3,3), strides=(2,2), padding='same')(input_image)
    fe = LeakyReLU(alpha=0.2)(fe)
    fe = Dropout(0.2)(fe)
    fe = Conv2D(64, (3,3),strides=(2,2), padding='same')(fe)
    fe = BatchNormalization()(fe)
    fe = LeakyReLU(alpha=0.2)(fe)
    fe = Dropout(0.2)(fe)
    fe = Conv2D(128, (3,3), strides=(2,2), padding='same')(fe)
    fe = BatchNormalization()(fe)
    fe = LeakyReLU(alpha=0.2)(fe)
    fe = Dropout(0.2)(fe)
    fe = Conv2D(256, (3,3), padding='same')(fe)
    fe = BatchNormalization()(fe)
    fe = LeakyReLU(alpha=0.2)(fe)
    fe = Dropout(0.3)(fe)
    fe = Flatten()(fe)
    out1 = Dense(1, activation='sigmoid')(fe)
    out2 = Dense(n_classes, activation='softmax')(fe)
    model = Model(input_image, [out1, out2])
    opt = Adam(learning_rate=0.00005, beta_1=0.5)
    model.compile(loss=['binary_crossentropy', 'sparse_categorical_crossentropy'], optimizer=opt)
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
    gen = Conv2DTranspose(256, (5,5), strides=(2,2), padding='same')(merge)
    gen = BatchNormalization()(gen)
    gen = Activation('relu')(gen)
    gen = Conv2DTranspose(128, (5,5), strides=(2,2), padding='same')(gen)
    gen = BatchNormalization()(gen)
    gen = Activation('relu')(gen)
    gen = Conv2DTranspose(64, (5,5), strides=(1,1), padding='same')(gen)
    gen = BatchNormalization()(gen)
    gen = Activation('relu')(gen)
    gen = Conv2DTranspose(1, (5,5), strides=(1,1), padding='same')(gen)
    out_layer = Activation('tanh')(gen)
    model = Model([input_lat, input_label], out_layer)
    return model
g = define_generator(latent_dim=100)
g.summary()

def define_gan(g_model, d_model):
    for layer in d_model.layers:
        if not isinstance(layer, BatchNormalization):
            layer.trainable = False
    gan_output = d_model(g_model.output)
    model = Model(g_model.input, gan_output)
    opt = Adam(learning_rate=0.0001, beta_1=0.5)
    model.compile(loss=['binary_crossentropy', 'sparse_categorical_crossentropy'], optimizer=opt)
    return model

# use the generator to generate n fake examples, with class labels
def generate_fake_samples(generator, latent_dim, n_samples, n_classes=7):
    z_input = np.random.randn(latent_dim * n_samples)
    z_input = z_input.reshape(n_samples, latent_dim)
    labels = np.random.randint(0, n_classes, n_samples).reshape(-1, 1)
    assert labels.min() >= 0 and labels.max() <= 6
    X_fake = generator.predict([z_input, labels])
    return X_fake, labels


def train(g_model, d_model, gan_model, dataset, latent_dim, n_epochs, n_batch):
    X_train, y_train = dataset
    bat_per_epo = max(1, X_train.shape[0] // n_batch)
    print(f"Batches/epoch: {bat_per_epo}")
    
    history = {'d_real': [], 'd_fake': [], 'g_loss': []}
    
    for epoch in range(1, n_epochs + 1):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch}/{n_epochs}")
        print(f"{'='*60}")
        
        d_r1_sum, d_r2_sum = 0, 0
        d_f_sum, d_f2_sum = 0, 0
        g_1_sum, g_2_sum = 0, 0
        
        for batch in range(bat_per_epo):
            ix = np.random.randint(0, X_train.shape[0], n_batch)
            X_real, labels_real = X_train[ix], y_train[ix]
            # Ajout de bruit
            X_real += np.random.normal(0, 0.05, X_real.shape)
            X_real = np.clip(X_real, -1.0, 1.0)
            y_real = np.random.uniform(0.8, 1.0, (n_batch, 1))
            _, d_r1, d_r2 = d_model.train_on_batch(X_real, [y_real, labels_real])

            X_fake, labels_fake = generate_fake_samples(g_model, latent_dim, n_batch)
            y_fake = np.random.uniform(0.0, 0.2, (n_batch, 1))
            _, d_f, d_f2 = d_model.train_on_batch(X_fake, [y_fake, labels_fake])

            z_input = np.random.randn(n_batch, latent_dim)
            z_labels = np.random.randint(0, 7, n_batch).reshape(-1, 1)
            y_gan = np.ones((n_batch, 1))
            _, g_1, g_2 = gan_model.train_on_batch([z_input, z_labels], [y_gan, z_labels])

            d_r1_sum += d_r1
            d_r2_sum += d_r2
            d_f_sum += d_f
            d_f2_sum += d_f2
            g_1_sum += g_1
            g_2_sum += g_2
        
        avg_d_real = d_r1_sum / bat_per_epo
        avg_d_fake = d_f_sum / bat_per_epo
        avg_g = g_1_sum / bat_per_epo
        
        history['d_real'].append(avg_d_real)
        history['d_fake'].append(avg_d_fake)
        history['g_loss'].append(avg_g)
        
        print(f"\n[Summary] D_real: {avg_d_real:.4f} | D_fake: {avg_d_fake:.4f} | G: {avg_g:.4f}")

        if epoch % 10 == 0:
            examples = 7
            latent_points = np.random.randn(examples, latent_dim)
            labels = np.arange(0, 7).reshape(-1, 1)
            X_fake = g_model.predict([latent_points, labels])
            imgs = (X_fake + 1.0) / 2.0
            
            plt.figure(figsize=(14, 2))
            for i in range(examples):
                plt.subplot(1, examples, i+1)
                plt.imshow(imgs[i].reshape(48, 48), cmap='gray')
                plt.axis('off')
                plt.title(f'Emo {int(labels[i])}', fontsize=10)
            plt.tight_layout()
            plt.savefig(f"generated_epoch_{epoch:04d}.png", dpi=100)
            plt.close()
            
            save_model(g_model, f'checkpoints/generator_epoch_{epoch}.h5')
            save_model(d_model, f'checkpoints/discriminator_epoch_{epoch}.h5')
            print(f"✓ Checkpoint sauvegardé: epoch {epoch}")
        
        if epoch % 50 == 0:
            plt.figure(figsize=(10, 4))
            plt.plot(history['d_real'], label='D_real', alpha=0.7)
            plt.plot(history['d_fake'], label='D_fake', alpha=0.7)
            plt.plot(history['g_loss'], label='G_loss', alpha=0.7)
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.title('Training History')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.savefig(f'training_history_epoch_{epoch}.png', dpi=100)
            plt.close()
            print(f"✓ Graphe sauvegardé: epoch {epoch}")
     

def run_cgan():
    os.makedirs('checkpoints', exist_ok=True)
    dataset = load_fer2013_dataset()
    X_train, y_train = dataset
    d_model = define_discriminator(input_shape=(48, 48, 1))
    g_model = define_generator(latent_dim=100)
    gan_model = define_gan(g_model, d_model)
    train(g_model, d_model, gan_model, (X_train, y_train), latent_dim=100, n_epochs=200, n_batch=16)
    save_model(d_model, 'discriminator_model.h5')
    save_model(g_model, 'generator_model.h5')
    save_model(gan_model, 'acgan_model.h5')
    print("\n✓ Entraînement terminé. Modèles finaux sauvegardés.")

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
     