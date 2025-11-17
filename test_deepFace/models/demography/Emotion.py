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
    from tensorflow.keras.layers import Conv2D, Flatten, Activation, MaxPooling2D, AveragePooling2D, Dropout, Dense, Input, Reshape, Concatenate

labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

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

def load_model_latent(latent_dim=500):
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

def load_model_decoder(latent_dim=500, num_classes=7) -> Model:
    from tensorflow.keras.layers import Conv2DTranspose, BatchNormalization
    
    latent_input = Input(shape=(latent_dim,), name="latent")
    emotion_input = Input(shape=(num_classes,), name="emotion")

    x = Concatenate()([latent_input, emotion_input])
    x = Dense(3 * 3 * 128, activation="relu")(x)
    x = Reshape((3, 3, 128))(x)
    
    x = Conv2DTranspose(128, (3, 3), strides=2, padding="same", activation="relu")(x)  # 6x6
    x = BatchNormalization()(x)
    x = Conv2DTranspose(64, (3, 3), strides=2, padding="same", activation="relu")(x)   # 12x12
    x = BatchNormalization()(x)
    x = Conv2DTranspose(64, (3, 3), strides=2, padding="same", activation="relu")(x)   # 24x24
    x = BatchNormalization()(x)
    x = Conv2DTranspose(32, (3, 3), strides=2, padding="same", activation="relu")(x)   # 48x48
    x = Conv2DTranspose(1, (3, 3), padding="same", activation="sigmoid")(x)            # 48x48x1

    decoder = Model(inputs=[latent_input, emotion_input], outputs=x, name="decoder")
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

def load_faces_dataset(folder):
    # Charge toutes les images, resize en (48,48), normalise
    import os, cv2
    X = []
    for fname in os.listdir(folder):
        for ffname in os.listdir(os.path.join(folder,fname)):
            if ffname.lower().endswith(('.jpg', '.jpeg', '.png')):
                img = cv2.imread(os.path.join(os.path.join(folder, fname), ffname), cv2.IMREAD_GRAYSCALE)
                if img is None: continue
                img = cv2.resize(img, (48, 48))
                img = img.astype(np.float32) / 255.0
                X.append(img)
    X = np.array(X)
    X = np.expand_dims(X, axis=-1)  # (N, 48, 48, 1)
    return X

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


if __name__ == "__main__":
    X_train = load_faces_dataset("imgs_db/train")
    print(f"Shape de X_train : {X_train.shape}")
    print(f"Dtype de X_train : {X_train.dtype}")
    latent_dim = 500
    encoder = load_model_latent(latent_dim)
    decoder = load_model_decoder(latent_dim, num_classes=len(labels))

    class Sampling(tf.keras.layers.Layer):
        def call(self, inputs):
            mu, logvar = inputs
            epsilon = tf.keras.backend.random_normal(shape=tf.shape(mu))
            return mu + tf.exp(0.5 * logvar) * epsilon
    img_input = Input(shape=(48, 48, 1))
    emotion_input = Input(shape=(len(labels),))
    mu, logvar = encoder(img_input)
    z = Sampling()([mu, logvar])
    reconstructed_img = decoder([z, emotion_input])
    vae = Model(inputs=[img_input, emotion_input], outputs=reconstructed_img)

    def vae_loss(y_true, y_pred, mu, logvar):
        recon_loss = tf.reduce_mean(tf.keras.losses.binary_crossentropy(y_true, y_pred))
        kl_loss = -0.5 * tf.reduce_mean(1 + logvar - tf.square(mu) - tf.exp(logvar))
        return recon_loss + kl_loss

    class VAETrainer(Model):
        def __init__(self, vae, encoder):
            super().__init__()
            self.vae = vae
            self.encoder = encoder

        def train_step(self, data):
            x = data
            emotion = tf.zeros((tf.shape(x)[0], 7))
            with tf.GradientTape() as tape:
                mu, logvar = self.encoder(x)
                z = Sampling()([mu, logvar])
                y_pred = self.vae([x, emotion])
                loss = vae_loss(x, y_pred, mu, logvar)
            grads = tape.gradient(loss, self.vae.trainable_weights)
            self.optimizer.apply_gradients(zip(grads, self.vae.trainable_weights))
            return {"loss": loss}
    
    dataset = tf.data.Dataset.from_tensor_slices(X_train)
    dataset = dataset.shuffle(buffer_size=1024).batch(64)
    
    vae_trainer = VAETrainer(vae, encoder)
    vae_trainer.compile(optimizer="adam")
    vae_trainer.fit(dataset, epochs=50)

    vae.save("vae_face_model.h5")
    decoder.save("decoder_model.h5")

