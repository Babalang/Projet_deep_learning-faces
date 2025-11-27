from typing import List, Union
import numpy as np
import cv2
import os
from tqdm import tqdm
from models.Demography import Demography
from commons import package_utils, weight_utils
import tensorflow as tf
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator


tf_major_version = package_utils.get_tf_major_version()
if tf_major_version == 1:
    from keras.models import Model,Sequential
    from keras.layers import Conv2D, Flatten, Activation, MaxPooling2D, AveragePooling2D, Dropout, Dense
else:
    from tensorflow.keras.models import Model,Sequential
    from tensorflow.keras.layers import Conv2D, Flatten, Activation, MaxPooling2D, AveragePooling2D, Dropout, Dense, Input, Reshape, Concatenate, Embedding, Conv2DTranspose, LeakyReLU, BatchNormalization

labels = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']

WEIGHTS_URL = "https://github.com/serengil/deepface_models/releases/download/v1.0/facial_expression_model_weights.h5"

class EmotionClient(Demography):
    def __init__(self):
        self.model = load_model()
        self.model_name = "Emotion"

    def _preprocess_image(self, img:np.ndarray) -> np.ndarray:
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_gray = cv2.resize(img_gray, (48, 48))
        return img_gray

    def predict(self,img:Union[np.ndarray, List[np.ndarray]]) -> np.ndarray:
        imgs = self._preprocess_batch_or_single_input(img)
        processed_imgs = np.expand_dims(np.array([self._preprocess_image(img) for img in imgs]), axis=-1)
        predictions = self._predict_internal(processed_imgs)
        return predictions
    
def load_model(
    url=WEIGHTS_URL,
    load_weights: bool = True,
) -> Sequential:
    """
    Consruct emotion model, download and load weights
    """

    num_classes = 7

    model = Sequential()

    # 1st convolution layer
    model.add(Conv2D(64, (5, 5), activation="relu", input_shape=(48, 48, 1)))
    model.add(MaxPooling2D(pool_size=(5, 5), strides=(2, 2)))

    # 2nd convolution layer
    model.add(Conv2D(64, (3, 3), activation="relu"))
    model.add(Conv2D(64, (3, 3), activation="relu"))
    model.add(AveragePooling2D(pool_size=(3, 3), strides=(2, 2)))

    # 3rd convolution layer
    model.add(Conv2D(128, (3, 3), activation="relu"))
    model.add(Conv2D(128, (3, 3), activation="relu"))
    model.add(AveragePooling2D(pool_size=(3, 3), strides=(2, 2)))

    model.add(Flatten())

    # fully connected neural networks
    model.add(Dense(1024, activation="relu"))
    model.add(Dropout(0.3))
    model.add(Dense(1024, activation="relu"))
    model.add(Dropout(0.3))

    model.add(Dense(num_classes, activation="softmax"))

    # ----------------------------
    if load_weights:
        model = weight_utils.load_model_weights(model=model, weights_file="emotion_model.h5")

    return model

def load_model_latent(latent_dim=128):
    img_input = Input(shape=(48, 48, 1), name="input")
    x = Conv2D(64, (5, 5), activation="relu")(img_input)
    x = MaxPooling2D(pool_size=(5, 5), strides=(2, 2))(x)
    x = Conv2D(64, (3, 3), activation="relu")(x)
    x = Conv2D(64, (3, 3), activation="relu")(x)
    x = AveragePooling2D(pool_size=(3, 3), strides=(2, 2))(x)
    x = Conv2D(128, (3, 3), activation="relu")(x)
    x = Conv2D(128, (3, 3), activation="relu")(x)
    x = AveragePooling2D(pool_size=(3, 3), strides=(2, 2))(x)
    x = Flatten()(x)
    x = Dense(1024, activation="relu")(x)
    x = Dropout(0.3)(x)
    x = Dense(1024, activation="relu")(x)
    x = Dropout(0.3)(x)
    mu = Dense(latent_dim, name="mu")(x)
    logvar = Dense(latent_dim, name="logvar")(x)
    model = Model(inputs=img_input, outputs=[mu, logvar], name="vae_encoder")
    return model

def load_model_decoder(latent_dim=256):
    latent_input = Input(shape=(latent_dim,), name="latent")
    label_input = Input(shape=(1,), name="label")
    label_emb = Embedding(len(labels), 64)(label_input)
    label_vec = Dense(latent_dim, activation="relu")(label_emb)
    label_vec = Reshape((latent_dim,))(label_vec)
    merged = Concatenate()([latent_input, label_vec])

    # UNet-style decoder
    x = Dense(12 * 12 * 256, activation="relu")(merged)
    x = Reshape((12, 12, 256))(x)

    # Downsampling (encoder part)
    d1 = Conv2D(256, 3, padding='same')(x)
    d1 = LeakyReLU(0.2)(d1)
    d1 = BatchNormalization()(d1)
    d2 = Conv2D(128, 3, strides=2, padding='same')(d1)  # 12x12 -> 6x6
    d2 = LeakyReLU(0.2)(d2)
    d2 = BatchNormalization()(d2)
    d3 = Conv2D(64, 3, strides=2, padding='same')(d2)   # 6x6 -> 3x3
    d3 = LeakyReLU(0.2)(d3)
    d3 = BatchNormalization()(d3)

    # Upsampling (decoder part) + skip connections
    u1 = Conv2DTranspose(64, 3, strides=2, padding='same')(d3)  # 3x3 -> 6x6
    u1 = LeakyReLU(0.2)(u1)
    u1 = BatchNormalization()(u1)
    u1 = Concatenate()([u1, d2])  # Skip connection

    u2 = Conv2DTranspose(128, 3, strides=2, padding='same')(u1) # 6x6 -> 12x12
    u2 = LeakyReLU(0.2)(u2)
    u2 = BatchNormalization()(u2)
    u2 = Concatenate()([u2, d1])  # Skip connection

    u3 = Conv2DTranspose(256, 4, strides=2, padding='same')(u2) # 12x12 -> 24x24
    u3 = LeakyReLU(0.2)(u3)
    u3 = BatchNormalization()(u3)

    u4 = Conv2DTranspose(128, 4, strides=2, padding='same')(u3) # 24x24 -> 48x48
    u4 = LeakyReLU(0.2)(u4)
    u4 = BatchNormalization()(u4)

    out_layer = Conv2D(1, 7, activation='sigmoid', padding='same')(u4)  # 48x48x1

    decoder = Model([latent_input, label_input], out_layer)
    return decoder




def load_fer2013_dataset(data_dir,labels):
    X,y = [],[]
    for idx,label in enumerate(labels):
        folder = os.path.join(data_dir,label)
        if not os.path.isdir(folder):
            continue
        files = [f for f in os.listdir(folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        for fname in tqdm(files, desc=f"Loading {label}"):
            fpath = os.path.join(folder,fname)
            img = cv2.imread(fpath,cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            img = cv2.resize(img,(48,48))
            X.append(img)
            y.append(idx)
    X = np.array(X,dtype=np.float32)/255.0
    X = np.expand_dims(X,axis=-1)
    y = np.array(y)
    return X,y

def load_faces_dataset_with_labels(folder, labels):
    # Charge toutes les images, resize en (48,48), normalise, retourne X et y
    import os, cv2
    X, y = [], []
    for idx, label in enumerate(labels):
        label_folder = os.path.join(folder, label)
        if not os.path.isdir(label_folder):
            continue
        for fname in os.listdir(label_folder):
            if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                img = cv2.imread(os.path.join(label_folder, fname), cv2.IMREAD_GRAYSCALE)
                if img is None: continue
                img = cv2.resize(img, (48, 48))
                img = img.astype(np.float32) / 255.0
                X.append(img)
                y.append(idx)
    X = np.array(X)
    X = np.expand_dims(X, axis=-1)  # (N, 48, 48, 1)
    y = np.array(y).reshape(-1, 1)  # (N, 1)
    return X, y

def train_emotion_model():
    print("Loading FER2013 dataset...")
    X_train,y_train = load_fer2013_dataset("imgs_db/train",labels)
    print("Loading validation dataset...")
    X_test,y_test = load_fer2013_dataset("imgs_db/test",labels)

    y_train_oneshot = tf.keras.utils.to_categorical(y_train,num_classes=len(labels))
    y_test_oneshot = tf.keras.utils.to_categorical(y_test,num_classes=len(labels))

    datagen = ImageDataGenerator(
        rotation_range=15,
        zoom_range=0.1,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True
    )
    datagen.fit(X_train)
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5)
    # 1. Entraînement initial sans class_weight
    model = load_model(load_weights=False)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(datagen.flow(X_train, y_train_oneshot, batch_size=64), epochs=30, validation_data=(X_test, y_test_oneshot), callbacks=[early_stopping, reduce_lr])
    model.save("emotion_pretrained.h5")

    # 2. Fine-tuning avec class_weight
    model = load_model(load_weights=False)
    model.load_weights("emotion_pretrained.h5")
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(y_train),
        y=y_train
    )
    class_weight_dict = dict(enumerate(class_weights))
    model.fit(datagen.flow(X_train, y_train_oneshot, batch_size=64), epochs=20, validation_data=(X_test, y_test_oneshot), class_weight=class_weight_dict, callbacks=[early_stopping, reduce_lr])
    model.save("emotion_model.h5")


def vae_loss(y_true, y_pred, mu, logvar):
    # Utilise uniquement BCE pour une meilleure correspondance pixel à pixel
    bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
    recon_loss = tf.reduce_sum(bce, axis=[1,2])
    kl_loss = -0.5 * tf.reduce_sum(1 + logvar - tf.square(mu) - tf.exp(logvar), axis=1)
    # KL très faible pour favoriser la reconstruction et l'émotion
    return tf.reduce_mean(recon_loss + 0.001 * kl_loss)

def vae_train():
    X_train, y_train = load_faces_dataset_with_labels("imgs_db/train", labels)
    print(f"Shape de X_train : {X_train.shape}")
    print(f"Dtype de X_train : {X_train.dtype}")
    latent_dim = 256  # Plus grand latent
    n_classes = len(labels)

    encoder = load_model_latent(latent_dim)
    decoder = load_model_decoder(latent_dim)

    img_input = Input(shape=(48,48,1))
    label_input = Input(shape=(1,))
    mu, logvar = encoder(img_input)
    class Sampling(tf.keras.layers.Layer):
        def call(self, inputs):
            mu, logvar = inputs
            epsilon = tf.random.normal(shape=tf.shape(mu))
            return mu + tf.exp(0.5 * logvar) * epsilon
    z = Sampling()([mu, logvar])
    reconstructed_img = decoder([z, label_input])
    vae = Model(inputs=[img_input, label_input], outputs=reconstructed_img)

    class VAE(Model):
        def __init__(self, encoder, decoder):
            super().__init__()
            self.encoder = encoder
            self.decoder = decoder
        def train_step(self, data):
            x, labels = data
            with tf.GradientTape() as tape:
                mu, logvar = self.encoder(x)
                z = Sampling()([mu, logvar])
                y_pred = self.decoder([z, labels])
                loss = vae_loss(x, y_pred, mu, logvar)
            grads = tape.gradient(loss, self.trainable_weights)
            self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
            return {"loss": loss}

    dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    dataset = dataset.shuffle(buffer_size=1024).batch(64)

    vae = VAE(encoder, decoder)
    vae.compile(optimizer=tf.keras.optimizers.Adam(1e-4))
    vae.fit(dataset, epochs=50)  # Plus d'epochs pour une meilleure incrustation
    encoder.save("encoder_model.h5")
    decoder.save("decoder_model.h5")
