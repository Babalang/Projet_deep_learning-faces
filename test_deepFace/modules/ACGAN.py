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
    opt = Adam(learning_rate=0.0001, beta_1=0.5)
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
    g_model = define_unet_generator(latent_dim=128)
    gan_model = define_gan(g_model, d_model)
    train(g_model, d_model, gan_model, (X_train, y_train), latent_dim=128, n_epochs=100, n_batch=64)
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

# Ajoute à la fin de ACGAN.py
def modify_emotion_with_acgan(img_path, new_emotion_label, noise_scale=0.5):
    import cv2
    import matplotlib.pyplot as plt
    import os
    
    encoder_path = os.path.join(os.path.dirname(__file__), '..', 'encoder_model.h5')
    encoder = load_model(encoder_path, compile=False)

    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (48, 48))
    img = img.astype(np.float32) / 255.0
    img = np.expand_dims(img, axis=-1)
    img = np.expand_dims(img, axis=0)

    mu, logvar = encoder.predict(img)
    z_vae = mu[0]
    z_gan = z_vae[:128]
    
    # Ajoute du bruit pour forcer des variations
    noise = np.random.randn(128) * noise_scale
    z_gan = z_gan + noise

    modified_img = generate_emotion_image(z_gan, new_emotion_label)

    # Affiche original et modifié côte à côte
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    axes[0].imshow(img.squeeze(), cmap='gray')
    axes[0].set_title('Original')
    axes[0].axis('off')
    axes[1].imshow(modified_img.squeeze(), cmap='gray')
    axes[1].set_title(f'Modified: {emotion_labels[new_emotion_label]}')
    axes[1].axis('off')
    plt.tight_layout()
    plt.show()

# ————————————————
# Solution 2: adaptateur VAE -> GAN
# ————————————————

def train_vae2gan_adapter(epochs=30, batch_size=64, vae_latent_dim=256, gan_latent_dim=128):
    """
    Entraîne un adaptateur mu(VAE)->z(GAN) pour que G(z, label)=image (label identique).
    Utilise Huber + perceptual loss via le discriminateur gelé. Clip des gradients.
    """
    import os
    from keras.layers import Input, Dense
    from keras.models import Model
    from keras.losses import Huber

    # Dataset GAN: images en [-1,1], labels int [0..6]
    X_train, y_train = load_fer2013_dataset()
    y_train = y_train.reshape(-1, 1)

    # Charge modèles
    encoder_path = os.path.join(os.path.dirname(__file__), '..', 'encoder_model.h5')
    encoder = load_model(encoder_path, compile=False)
    g_model = load_model('generator_model.h5', compile=False)
    d_model = define_discriminator(input_shape=(48,48,1))  # même archi que le tien
    d_model.trainable = False

    # Gèle encoder/G
    encoder.trainable = False
    g_model.trainable = False

    # Adaptateur
    adapter_in = Input(shape=(vae_latent_dim,), name='adapter_in')
    a = Dense(512, activation='relu')(adapter_in)
    a = Dense(256, activation='relu')(a)
    adapter_out = Dense(gan_latent_dim, name='adapter_out')(a)
    adapter = Model(adapter_in, adapter_out, name='vae2gan_adapter')

    huber = Huber()
    opt = tf.keras.optimizers.Adam(1e-4, clipnorm=1.0)

    @tf.function
    def train_step(x_gan, labels):
        # x_gan in [-1,1]; encoder attend [0,1]
        x_enc = (x_gan + 1.0) / 2.0
        with tf.GradientTape() as tape:
            mu, logvar = encoder(x_enc, training=False)
            z_gan = adapter(mu, training=True)
            x_pred = g_model([z_gan, labels], training=False)   # [-1,1]

            # Reconstruction (Huber) en [-1,1]
            recon = huber(x_gan, x_pred)

            # Perceptual loss: features du discri (après flatten)
            # On prend la sortie "emotion" logits comme features
            _, feat_real = d_model(x_gan, training=False)
            _, feat_pred = d_model(x_pred, training=False)
            perceptual = tf.reduce_mean(tf.abs(feat_real - feat_pred))

            # Régularisation latente
            reg_z = 1e-3 * tf.reduce_mean(tf.square(z_gan))

            loss = recon + 0.1 * perceptual + reg_z

        grads = tape.gradient(loss, adapter.trainable_variables)
        opt.apply_gradients(zip(grads, adapter.trainable_variables))
        return loss

    ds = tf.data.Dataset.from_tensor_slices((X_train, y_train)).shuffle(8192).batch(batch_size).prefetch(tf.data.AUTOTUNE)

    for epoch in range(1, epochs + 1):
        losses = []
        for x_b, y_b in ds:
            loss = train_step(x_b, y_b)
            losses.append(loss.numpy())
        print(f"[Adapter] Epoch {epoch}/{epochs} - loss: {np.mean(losses):.4f}")

    adapter_path = os.path.join(os.path.dirname(__file__), 'vae2gan_adapter.h5')
    adapter.save(adapter_path)
    print(f"[OK] Adaptateur sauvé: {adapter_path}")


def modify_emotion_with_acgan_using_adapter(img_path, new_emotion_label, vae_latent_dim=256):
    """
    Inférence: Image -> mu(VAE) -> Adapter -> z(GAN) -> G(z, new_label).
    """
    import cv2
    import matplotlib.pyplot as plt
    import os

    encoder_path = os.path.join(os.path.dirname(__file__), '..', 'encoder_model.h5')
    adapter_path = os.path.join(os.path.dirname(__file__), 'vae2gan_adapter.h5')
    encoder = load_model(encoder_path, compile=False)
    adapter = load_model(adapter_path, compile=False)
    g_model = load_model('generator_model.h5', compile=False)

    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Image introuvable: {img_path}")
    img = cv2.resize(img, (48, 48))
    img_enc = img.astype(np.float32) / 255.0            # [0,1] pour encoder
    x_enc = np.expand_dims(img_enc[..., None], 0)

    mu, logvar = encoder.predict(x_enc, verbose=0)
    z_gan = adapter.predict(mu, verbose=0)              # (1, 128)
    label = np.array([[int(new_emotion_label)]], dtype=np.int32)

    modified = g_model.predict([z_gan, label], verbose=0)  # [-1,1]
    modified_disp = (modified[0].squeeze() + 1.0) / 2.0    # [0,1]

    # Affichage
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    axes[0].imshow(img_enc, cmap='gray')
    axes[0].set_title('Original')
    axes[0].axis('off')
    axes[1].imshow(modified_disp, cmap='gray')
    axes[1].set_title(f'Modified: {emotion_labels[int(new_emotion_label)]}')
    axes[1].axis('off')
    plt.tight_layout()
    plt.show()


def _disc_feature_extractor(d_model):
    # Récupère la dernière couche Conv2D comme “perceptual features”
    conv_layers = [l for l in d_model.layers if isinstance(l, Conv2D)]
    feat_layer = conv_layers[-1].output if conv_layers else d_model.layers[-3].output  # fallback
    return Model(d_model.input, feat_layer, name="disc_feat")

def _edge_map(x01):
    # x01: [0,1], shape (B,H,W,1) ou (H,W,1). Retourne magnitude des gradients Sobel.
    x01 = tf.convert_to_tensor(x01, dtype=tf.float32)
    if x01.shape.rank == 3:
        x01 = tf.expand_dims(x01, axis=0)  # (1,H,W,1)
    gxgy = tf.image.sobel_edges(x01)  # (B,H,W,1,2)
    gx, gy = gxgy[..., 0], gxgy[..., 1]
    mag = tf.sqrt(gx * gx + gy * gy + 1e-8)
    return mag

def invert_latent_acgan(img_path, latent_dim=128, steps=800, lr=0.02, n_starts=5):
    import cv2
    g_model = load_model('generator_model.h5', compile=False)
    d_model = load_model('discriminator_model.h5', compile=False)
    feat_model = _disc_feature_extractor(d_model)

    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Image introuvable: {img_path}")
    img = cv2.resize(img, (48, 48))
    img01 = img.astype(np.float32) / 255.0
    x_gan = np.expand_dims((img01 * 2.0 - 1.0)[..., None], 0)  # [-1,1]
    x01 = np.expand_dims(img01[..., None], 0)                  # [0,1]

    # Label d'origine via D
    _, emo_logits = d_model.predict(x_gan, verbose=0)
    y0 = int(np.argmax(emo_logits, axis=1)[0])
    y0_t = tf.convert_to_tensor([[y0]], dtype=tf.int32)

    best_z, best_loss = None, 1e9
    for _ in range(n_starts):
        z = tf.Variable(tf.random.normal([1, latent_dim], stddev=1.0), trainable=True)
        opt = tf.keras.optimizers.Adam(lr, clipnorm=1.0)
        for _it in range(steps):
            with tf.GradientTape() as tape:
                x_pred = g_model([z, y0_t], training=False)      # [-1,1]
                x_pred01 = (x_pred + 1.0) / 2.0                  # [0,1]

                # Pertes: L1 + SSIM + perceptual + edges + TV + prior
                l1 = tf.reduce_mean(tf.abs(x_pred - x_gan))
                ssim_loss = 1.0 - tf.reduce_mean(tf.image.ssim(x_pred01, x01, max_val=1.0))
                f_real = feat_model(x_gan, training=False)
                f_pred = feat_model(x_pred, training=False)
                percept = tf.reduce_mean(tf.abs(f_real - f_pred))
                edges_real = _edge_map(x01)
                edges_pred = _edge_map(x_pred01)
                edge_loss = tf.reduce_mean(tf.abs(edges_real - edges_pred))
                tv = tf.reduce_mean(tf.image.total_variation(x_pred01))
                prior = tf.reduce_mean(tf.square(z))  # N(0,1)

                loss = (
                    1.0 * l1
                    + 0.5 * ssim_loss
                    + 0.2 * percept
                    + 0.3 * edge_loss
                    + 1e-4 * tv
                    + 1e-3 * prior
                )
            grads = tape.gradient(loss, [z])
            opt.apply_gradients(zip(grads, [z]))
        cur_loss = float(loss.numpy())
        if cur_loss < best_loss:
            best_loss, best_z = cur_loss, z.numpy()
    return best_z, y0, img01

def _detect_and_crop_face(img_bgr, target_size=(48, 48)):
    """
    Détecte la plus grande face et retourne:
    - face_crop_gray (48x48)
    - bbox (x, y, w, h) dans l'image originale
    - img_gray (pour overlay)
    """
    import cv2
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    # Haar cascade (assure-toi que le fichier est présent)
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    face_cascade = cv2.CascadeClassifier(cascade_path)
    faces = face_cascade.detectMultiScale(img_gray, scaleFactor=1.2, minNeighbors=5, minSize=(30, 30))
    if len(faces) == 0:
        # fallback: prend tout, redimensionne
        h, w = img_gray.shape
        x, y, w0, h0 = 0, 0, w, h
    else:
        # prend la plus grande face
        x, y, w0, h0 = sorted(faces, key=lambda b: b[2]*b[3], reverse=True)[0]
        # petite marge autour du visage
        pad = int(0.15 * max(w0, h0))
        x = max(0, x - pad); y = max(0, y - pad)
        w0 = min(img_gray.shape[1] - x, w0 + 2*pad)
        h0 = min(img_gray.shape[0] - y, h0 + 2*pad)
    face = img_gray[y:y+h0, x:x+w0]
    face_resized = cv2.resize(face, target_size, interpolation=cv2.INTER_AREA)
    return face_resized, (x, y, w0, h0), img_gray

def _match_histogram(src_face_gray, dst_region_gray):
    import numpy as np
    # Normalise l’histogramme du visage modifié sur la région cible (mean/std)
    s = src_face_gray.astype(np.float32)
    d = dst_region_gray.astype(np.float32)
    s_mean, s_std = s.mean(), s.std() + 1e-6
    d_mean, d_std = d.mean(), d.std() + 1e-6
    matched = (s - s_mean) / s_std * d_std + d_mean
    return np.clip(matched, 0, 255).astype(np.uint8)

def _laplacian_blend(src_patch_bgr, dst_patch_bgr, mask_gray, levels=3):
    """
    Laplacian pyramid blending local (src dans dst avec masque doux).
    src_patch_bgr, dst_patch_bgr: (h,w,3), uint8
    mask_gray: (h,w), uint8 in [0,255]
    """
    import cv2, numpy as np
    h, w = mask_gray.shape
    # Normalise masque [0,1]
    mask = (mask_gray.astype(np.float32) / 255.0)
    mask = cv2.GaussianBlur(mask, (0,0), sigmaX=max(h,w)/40.0)  # feathering fort

    def build_pyr(img):
        g = img.astype(np.float32) / 255.0
        gp = [g]
        for _ in range(levels):
            g = cv2.pyrDown(g)
            gp.append(g)
        lp = [gp[-1]]
        for i in range(levels, 0, -1):
            ge = cv2.pyrUp(gp[i])
            ge = cv2.resize(ge, (gp[i-1].shape[1], gp[i-1].shape[0]))
            l = gp[i-1] - ge
            lp.append(l)
        return gp, lp

    def build_mask_pyr(m):
        gp = [m]
        for _ in range(levels):
            m = cv2.pyrDown(m)
            gp.append(m)
        return gp

    gpS, lpS = build_pyr(src_patch_bgr)
    gpD, lpD = build_pyr(dst_patch_bgr)
    gpM = build_mask_pyr(mask)

    LS = []
    for lS, lD, m in zip(lpS, lpD, reversed(gpM)):
        m3 = np.repeat(m[..., None], 3, axis=2)
        LS.append(lS * m3 + lD * (1.0 - m3))

    # Reconstruction
    img = LS[0]
    for i in range(1, len(LS)):
        img = cv2.pyrUp(img)
        img = cv2.resize(img, (LS[i].shape[1], LS[i].shape[0]))
        img = img + LS[i]
    img = np.clip(img * 255.0, 0, 255).astype(np.uint8)
    return img

def _paste_face(img_gray, face48, bbox, scale=0.88, mask_blur=41, blend_mode="mixed"):
    """
    Intègre le visage modifié avec:
    - shrink (scale<1) pour limiter débordements
    - mask feathering fort
    - histogram matching
    - Laplacian pyramid blending local pour éviter voir l’ancien visage
    - (fallback) seamlessClone MIXED/NORMAL
    """
    import cv2, numpy as np
    x, y, w0, h0 = bbox

    # Shrink léger
    w1 = max(1, int(w0 * scale))
    h1 = max(1, int(h0 * scale))
    face_resized = cv2.resize(face48, (w1, h1), interpolation=cv2.INTER_CUBIC)

    # Inset centré
    x_inset = x + (w0 - w1) // 2
    y_inset = y + (h0 - h1) // 2

    # Patch destination + harmonisation
    dst_patch = img_gray[y_inset:y_inset+h1, x_inset:x_inset+w1]
    face_harmonized = _match_histogram(face_resized, dst_patch)

    # Conversions
    src = cv2.cvtColor(face_harmonized, cv2.COLOR_GRAY2BGR)
    dst = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)

    # Masque ellipsoïdal doux
    mask = np.zeros((h1, w1), dtype=np.uint8)
    center_local = (w1 // 2, h1 // 2)
    axes = (int(w1 * 0.44), int(h1 * 0.44))
    cv2.ellipse(mask, center_local, axes, 0, 0, 360, 255, -1)
    mask = cv2.GaussianBlur(mask, (mask_blur, mask_blur), 0)

    # Laplacian blending local (sur le patch)
    blended_patch = _laplacian_blend(src, dst[y_inset:y_inset+h1, x_inset:x_inset+w1].copy(), mask, levels=3)
    dst[y_inset:y_inset+h1, x_inset:x_inset+w1] = blended_patch

    # Fallback Poisson blending (applique peu, masque déjà doux)
    center_global = (x_inset + w1 // 2, y_inset + h1 // 2)
    mode = cv2.MIXED_CLONE if blend_mode == "mixed" else cv2.NORMAL_CLONE
    try:
        dst = cv2.seamlessClone(src, dst, mask, center_global, mode)
    except Exception:
        pass

    # Post-lissage local (bilateral pour atténuer démarcation)
    patch = dst[y:y+h0, x:x+w0]
    patch_smooth = cv2.bilateralFilter(patch, d=7, sigmaColor=35, sigmaSpace=35)
    dst[y:y+h0, x:x+w0] = patch_smooth

    out_gray = cv2.cvtColor(dst, cv2.COLOR_BGR2GRAY)
    return out_gray

def modify_emotion_on_cropped_face(img_path, new_emotion_label,
                                   inv_steps=800, inv_lr=0.02,
                                   edit_steps=600, edit_lr=0.005):
    """
    Pipeline:
    1) Détecte et crop le visage (face_extraite) -> 48x48
    2) Applique modify_emotion_with_acgan_inversion sur le crop
    3) Replace le visage modifié dans l'image originale
    """
    import cv2
    # 1) Charge image et crop
    img_bgr = cv2.imread(img_path)
    if img_bgr is None:
        raise ValueError(f"Image introuvable: {img_path}")
    face48, bbox, img_gray = _detect_and_crop_face(img_bgr, target_size=(48, 48))

    # 2) Sauvegarde temporaire du crop et édition
    tmp_path = os.path.join(os.path.dirname(__file__), "tmp_face_crop.png")
    cv2.imwrite(tmp_path, face48)

    # Utilise la même inversion avec des hyperparams plus doux (car crop déjà centré)
    # Note: modify_emotion_with_acgan_inversion affiche; on veut juste le résultat → factorisons la génération
    # On reprend le cœur de la fonction pour obtenir x_final sans plots:
    g_model = load_model('generator_model.h5', compile=False)
    d_model = load_model('discriminator_model.h5', compile=False)
    encoder_path = os.path.join(os.path.dirname(__file__), '..', 'encoder_model.h5')
    encoder = load_model(encoder_path, compile=False)

    # Inversion sur le crop
    z_star, y0, img01 = invert_latent_acgan(tmp_path, latent_dim=128, steps=inv_steps, lr=inv_lr, n_starts=3)

    # Édition légère (plus conservatrice)
    x_gan = np.expand_dims((img01 * 2.0 - 1.0)[..., None], 0).astype('float32')
    x01 = np.expand_dims(img01[..., None], 0).astype('float32')
    mu_ref, _ = encoder.predict(x01, verbose=0)

    delta = tf.Variable(tf.zeros_like(z_star), trainable=True)
    y_new_t = tf.convert_to_tensor([[int(new_emotion_label)]], dtype=tf.int32)
    opt = tf.keras.optimizers.Adam(edit_lr, clipnorm=1.0)
    ce = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)

    for _ in range(edit_steps):
        with tf.GradientTape() as tape:
            z = z_star + delta
            x_pred = g_model([z, y_new_t], training=False)
            x_pred01 = (x_pred + 1.0) / 2.0

            # Identité et structure plus fortes sur crop
            mu_pred, _ = encoder(x_pred01, training=False)
            id_loss = tf.reduce_mean(tf.abs(mu_pred - mu_ref))
            _, emo_prob = d_model([x_pred], training=False)
            emo_ce = ce(y_new_t, emo_prob)
            ssim_loss = 1.0 - tf.reduce_mean(tf.image.ssim(x_pred01, x01, max_val=1.0))
            pix = tf.reduce_mean(tf.abs(x_pred - x_gan))
            delta_reg = tf.reduce_mean(tf.square(delta))

            loss = (1.5 * emo_ce + 1.0 * id_loss + 0.3 * pix + 0.3 * ssim_loss + 0.6 * delta_reg)

        grads = tape.gradient(loss, [delta])
        opt.apply_gradients(zip(grads, [delta]))
        delta.assign(tf.clip_by_norm(delta, 1.0))

    x_final = g_model.predict([z_star + delta.numpy(), np.array([[int(new_emotion_label)]])], verbose=0)
    face_mod = (x_final[0].squeeze() + 1.0) / 2.0  # [0,1]

    # 3) Remet le visage modifié dans l'image originale
    face_mod_uint8 = np.clip(face_mod * 255.0, 0, 255).astype(np.uint8)
    out_gray = _paste_face(img_gray, face_mod_uint8, bbox, scale=0.88, mask_blur=41, blend_mode="mixed")

    # Affiche
    plt.figure(figsize=(10,4))
    plt.subplot(1,2,1); plt.imshow(img_gray, cmap='gray'); plt.title('Original'); plt.axis('off')
    plt.subplot(1,2,2); plt.imshow(out_gray, cmap='gray'); plt.title(f'Modified: {emotion_labels[int(new_emotion_label)]}'); plt.axis('off')
    plt.tight_layout(); plt.show()

    # Nettoyage
    try:
        os.remove(tmp_path)
    except Exception:
        pass

if __name__ == "__main__":
    print("=== ACGAN ===")
    print("Voulez-vous entraîner le modèle ACGAN ou générer une image ?")
    choice = input("Entrez 'train' pour entraîner, 'generate' pour générer une image, 'modify' (sans adaptateur) ou 'adapt' / 'modify_adapter' / 'modify_invert': ").strip().lower()
    if choice == 'train':
        run_cgan()
    elif choice == 'generate':
        latent_vector = np.random.randn(128)
        emotion_label = int(input("Entrez une étiquette d'émotion (0-6): "))
        generated_image = generate_emotion_image(latent_vector, emotion_label)
        import matplotlib.pyplot as plt
        plt.imshow(generated_image.reshape(48,48), cmap='gray'); plt.axis('off'); plt.title(f'Émotion: {emotion_label}'); plt.show()
    elif choice == 'modify':
        img_path = input("Entrez le chemin de l'image à modifier: ").strip()
        new_emotion_label = int(input("Entrez la nouvelle étiquette d'émotion (0-6): "))
        modify_emotion_with_acgan(img_path, new_emotion_label)
    elif choice == 'adapt':
        train_vae2gan_adapter(epochs=30, batch_size=32, vae_latent_dim=256, gan_latent_dim=128)
    elif choice == 'modify_adapter':
        img_path = input("Entrez le chemin de l'image à modifier: ").strip()
        new_emotion_label = int(input("Entrez la nouvelle étiquette d'émotion (0-6): "))
        modify_emotion_with_acgan_using_adapter(img_path, new_emotion_label, vae_latent_dim=256)
    elif choice == 'modify_invert':
        img_path = input("Entrez le chemin de l'image à modifier: ").strip()
        new_emotion_label = int(input("Entrez la nouvelle étiquette d'émotion (0-6): "))
        # Applique sur face_extraite
        modify_emotion_on_cropped_face(img_path, new_emotion_label,
                                       inv_steps=800, inv_lr=0.02,
                                       edit_steps=600, edit_lr=0.005)
    else:
        print("Choix invalide. Veuillez entrer 'train', 'generate', 'modify', 'adapt', 'modify_adapter' ou 'modify_invert'.")
     