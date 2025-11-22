import os
from typing import Tuple, Optional, List
import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import (
    Input, Conv2D, MaxPooling2D, UpSampling2D, Dense,
    Concatenate, Layer, GlobalAveragePooling2D
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

EMOTION_LABELS = ['angry','disgust','fear','happy','sad','surprise','neutral']
LATENT_CHANNELS = 8  # profondeur latente
IMAGE_SIZE = 96
DEFAULT_IMAGE_SHAPE = (IMAGE_SIZE, IMAGE_SIZE, 1)

@tf.keras.utils.register_keras_serializable()
class FiLM(Layer):
    def __init__(self, channels, **kwargs):
        super().__init__(**kwargs)
        self.channels = channels

    def build(self, input_shape):
        self.gamma_dense = Dense(self.channels)
        self.beta_dense = Dense(self.channels)

    def call(self, x, cond):
        gamma = self.gamma_dense(cond)
        beta = self.beta_dense(cond)
        gamma = tf.reshape(gamma, (-1, 1, 1, self.channels))
        beta = tf.reshape(beta, (-1, 1, 1, self.channels))
        return x * (1 + gamma) + beta

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"channels": self.channels})
        return cfg


def build_unet_encoder(input_shape=DEFAULT_IMAGE_SHAPE, latent_channels=LATENT_CHANNELS):
    inp = Input(shape=input_shape)
    c1 = Conv2D(16,3,activation='relu',padding='same')(inp)
    c1 = Conv2D(16,3,activation='relu',padding='same')(c1)
    p1 = MaxPooling2D(2)(c1)   # 48

    c2 = Conv2D(32,3,activation='relu',padding='same')(p1)
    c2 = Conv2D(32,3,activation='relu',padding='same')(c2)
    p2 = MaxPooling2D(2)(c2)   # 24

    c3 = Conv2D(64,3,activation='relu',padding='same')(p2)
    c3 = Conv2D(64,3,activation='relu',padding='same')(c3)
    p3 = MaxPooling2D(2)(c3)   # 12

    c4 = Conv2D(128,3,activation='relu',padding='same')(p3)
    c4 = Conv2D(128,3,activation='relu',padding='same')(c4)
    p4 = MaxPooling2D(2)(c4)   # 6

    bott = Conv2D(256,3,activation='relu',padding='same')(p4)
    bott = Conv2D(256,3,activation='relu',padding='same')(bott)

    mu = Conv2D(latent_channels,1,activation=None,name='mu_map')(bott)
    logvar = Conv2D(latent_channels,1,activation=None,name='logvar_map')(bott)
    return Model(inp, [mu, logvar, c1, c2, c3, c4], name="unet_encoder")

def build_unet_decoder(n_emotions: int, latent_channels=LATENT_CHANNELS):
    z_in = Input(shape=(6,6,latent_channels))
    emo_in = Input(shape=(n_emotions,))
    s1 = Input(shape=(IMAGE_SIZE,IMAGE_SIZE,16))
    s2 = Input(shape=(IMAGE_SIZE//2,IMAGE_SIZE//2,32))
    s3 = Input(shape=(IMAGE_SIZE//4,IMAGE_SIZE//4,64))
    s4 = Input(shape=(IMAGE_SIZE//8,IMAGE_SIZE//8,128))
    emo_emb = Dense(64,activation='relu')(emo_in)

    u4 = UpSampling2D()(z_in)  # 12
    u4b = Conv2D(128,3,activation='relu',padding='same')(u4)
    u4 = FiLM(128)(u4b, emo_emb)
    u4 = Concatenate()([u4, s4])

    u3 = UpSampling2D()(u4)    # 24
    u3b = Conv2D(64,3,activation='relu',padding='same')(u3)
    u3 = FiLM(64)(u3b, emo_emb)
    u3 = Concatenate()([u3, s3])

    u2 = UpSampling2D()(u3)    # 48
    u2b = Conv2D(32,3,activation='relu',padding='same')(u2)
    u2 = FiLM(32)(u2b, emo_emb)
    u2 = Concatenate()([u2, s2])

    u1 = UpSampling2D()(u2)    # 96
    u1b = Conv2D(16,3,activation='relu',padding='same')(u1)
    u1 = FiLM(16)(u1b, emo_emb)
    u1 = Concatenate()([u1, s1])

    out = Conv2D(1,1,activation='sigmoid')(u1)
    cls_head = GlobalAveragePooling2D()(u3)
    cls_head = Dense(64,activation='relu')(cls_head)
    cls_out = Dense(n_emotions,activation='softmax',name='emotion_pred')(cls_head)
    return Model([z_in, emo_in, s1, s2, s3, s4], [out, cls_out], name="unet_decoder")

def sample_spatial(mu, logvar):
    eps = tf.random.normal(shape=tf.shape(mu))
    return mu + tf.exp(0.5 * logvar) * eps


def build_identity_embed():
    inp = Input(shape=DEFAULT_IMAGE_SHAPE)
    x = Conv2D(32, 3, activation='relu', padding='same')(inp)
    x = MaxPooling2D(2)(x)
    x = Conv2D(64, 3, activation='relu', padding='same')(x)
    x = MaxPooling2D(2)(x)
    x = Conv2D(128, 3, activation='relu', padding='same')(x)
    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation='relu')(x)
    x = Dense(64, activation=None)(x)
    return Model(inp, x, name="id_embed")


class CVAE(Model):
    def __init__(self, encoder, decoder, beta=1.0, free_bits=0.5, emo_weight=1.0, edit_prob=1.0):
        super().__init__()
        self.encoder=encoder; self.decoder=decoder
        self.beta=beta; self.free_bits=free_bits; self.emo_weight=emo_weight
        self.recon_loss_tracker=tf.keras.metrics.Mean("recon_loss")
        self.id_loss_tracker=tf.keras.metrics.Mean("id_loss")
        self.emo_loss_tracker=tf.keras.metrics.Mean("emo_loss")
        self.kl_loss_tracker=tf.keras.metrics.Mean("kl_loss")
        self.total_loss_tracker=tf.keras.metrics.Mean("loss")
        self.id_net=build_identity_embed()
        self.edit_prob=edit_prob

    def call(self, inputs, training=False):
        imgs, emotions = inputs
        mu_map, logvar_map, s1, s2, s3, s4 = self.encoder(imgs, training=training)
        z = sample_spatial(mu_map, logvar_map) if training else mu_map
        recon, _ = self.decoder([z, emotions, s1, s2, s3, s4], training=training)
        return recon

    @property
    def metrics(self):
        return [self.total_loss_tracker,self.recon_loss_tracker,self.id_loss_tracker,self.emo_loss_tracker,self.kl_loss_tracker]

    def train_step(self,data):
        (imgs, emotions), _ = data
        num_emotions = len(EMOTION_LABELS)
        with tf.GradientTape() as tape:
            mu_map, logvar_map, s1,s2,s3,s4 = self.encoder(imgs,training=True)
            z = sample_spatial(mu_map,logvar_map)

            # ORIGINAL reconstruction path
            recon_orig, emo_pred_orig = self.decoder([z, emotions, s1,s2,s3,s4], training=True)

            # Emotion editing path: force different target emotion
            orig_idx = tf.argmax(emotions, axis=1, output_type=tf.int32)
            rand_idx = tf.random.uniform(shape=tf.shape(orig_idx), minval=0, maxval=num_emotions, dtype=tf.int32)
            # assure différence
            same = tf.equal(rand_idx, orig_idx)
            rand_idx = tf.where(same, (rand_idx + 1) % num_emotions, rand_idx)
            edited_emotions = tf.one_hot(rand_idx, num_emotions)

            # Optionnel: sous-ensemble (edit_prob)
            if self.edit_prob < 1.0:
                mask = tf.cast(tf.random.uniform(tf.shape(orig_idx),0,1) < self.edit_prob, tf.bool)
                edited_emotions = tf.where(mask[:,None], edited_emotions, emotions)
            edited_recon, emo_pred_edit = self.decoder([z, edited_emotions, s1,s2,s3,s4], training=True)

            # Reconstruction + identité uniquement sur original
            bce = tf.keras.losses.binary_crossentropy(imgs,recon_orig)
            recon_loss = tf.reduce_mean(tf.reduce_sum(bce,axis=[1,2])/(IMAGE_SIZE*IMAGE_SIZE))

            emb_r = tf.math.l2_normalize(self.id_net(imgs,training=True),axis=-1)
            emb_x = tf.math.l2_normalize(self.id_net(recon_orig,training=True),axis=-1)
            id_loss = tf.reduce_mean(1 - tf.reduce_sum(emb_r*emb_x,axis=-1))

            # Emotion classification loss sur version éditée
            emo_loss = tf.reduce_mean(tf.keras.losses.categorical_crossentropy(edited_emotions, emo_pred_edit))

            # KL
            kl_cell = -0.5*(1+logvar_map-tf.square(mu_map)-tf.exp(logvar_map))
            kl = tf.reduce_mean(tf.reduce_sum(tf.maximum(kl_cell,self.free_bits),axis=[1,2,3]))

            loss = recon_loss + 0.3*id_loss + self.emo_weight*emo_loss + self.beta*kl

        grads = tape.gradient(loss,self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads,self.trainable_variables))
        self.recon_loss_tracker.update_state(recon_loss)
        self.id_loss_tracker.update_state(id_loss)
        self.emo_loss_tracker.update_state(emo_loss)
        self.kl_loss_tracker.update_state(kl)
        self.total_loss_tracker.update_state(loss)
        return {"loss":self.total_loss_tracker.result(),
                "recon_loss":self.recon_loss_tracker.result(),
                "id_loss":self.id_loss_tracker.result(),
                "emo_loss":self.emo_loss_tracker.result(),
                "kl_loss":self.kl_loss_tracker.result()}


class BetaAnneal(tf.keras.callbacks.Callback):
    def __init__(self, cvae: CVAE, beta_final=1.0, warmup=8):
        super().__init__()
        self.cvae = cvae
        self.beta_final = beta_final
        self.warmup = warmup

    def on_epoch_begin(self, epoch, logs=None):
        if epoch < self.warmup:
            self.cvae.beta = self.beta_final * (epoch + 1) / self.warmup
        else:
            self.cvae.beta = self.beta_final


def load_images_from_folders(data_dir: str) -> Tuple[np.ndarray, np.ndarray]:
    import cv2
    from pathlib import Path
    X, y = [], []
    for idx, label in enumerate(EMOTION_LABELS):
        folder = Path(data_dir) / label
        if not folder.is_dir():
            continue
        for f in folder.glob("*"):
            if f.suffix.lower() not in ('.jpg', '.jpeg', '.png'):
                continue
            img = cv2.imread(str(f), cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE)).astype(np.float32) / 255.0
            X.append(img)
            y.append(idx)
    if not X:
        raise ValueError("Dataset vide.")
    X = np.array(X)[..., None]
    y = np.array(y)
    return X, y


def train_cvae(
    data_dir: str,
    batch_size: int = 64,
    epochs: int = 40,
    lr: float = 1e-3,
    beta: float = 0.5,
    save_dir: Optional[str] = "./models/vae",
):
    print("Chargement données...")
    X, y = load_images_from_folders(data_dir)
    y_oh = to_categorical(y, len(EMOTION_LABELS))
    ds = tf.data.Dataset.from_tensor_slices(((X, y_oh), X)).shuffle(10000).batch(batch_size).prefetch(tf.data.AUTOTUNE)

    print("Construction modèle...")
    encoder = build_unet_encoder()
    decoder = build_unet_decoder(len(EMOTION_LABELS))
    cvae = CVAE(encoder, decoder, beta=0.0, free_bits=0.5, emo_weight=1.0, edit_prob=1.0)
    cvae.compile(optimizer=Adam(lr))
    _ = cvae((tf.zeros((1, IMAGE_SIZE, IMAGE_SIZE, 1)), tf.zeros((1, len(EMOTION_LABELS)))))

    callbacks = [
        BetaAnneal(cvae, beta_final=beta, warmup=8),
        EarlyStopping(monitor='recon_loss', mode='min', patience=6, restore_best_weights=True),
        ReduceLROnPlateau(monitor='recon_loss', mode='min', factor=0.5, patience=3, min_lr=5e-5),
    ]

    cvae.fit(ds, epochs=epochs, callbacks=callbacks)
    os.makedirs(save_dir, exist_ok=True)
    encoder.save(os.path.join(save_dir, "cvae_encoder.h5"))
    decoder.save(os.path.join(save_dir, "cvae_decoder.h5"))
    cvae.save_weights(os.path.join(save_dir, "cvae_final.weights.h5"))
    print("✓ Sauvegardé.")
    return cvae, encoder, decoder


def preprocess_face_for_vae(path_or_img):
    import cv2
    if isinstance(path_or_img,str):
        img=cv2.imread(path_or_img,cv2.IMREAD_GRAYSCALE)
        if img is None: raise FileNotFoundError(path_or_img)
    else:
        img=path_or_img
        if img.ndim==3:
            import cv2
            img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    img=cv2.resize(img,(IMAGE_SIZE,IMAGE_SIZE)).astype(np.float32)/255.0
    return np.expand_dims(img,axis=(0,-1))


def reconstruct_with_emotion(img: np.ndarray, encoder: Model, decoder: Model, target_emotion_idx: int):
    mu_map, logvar_map, s1, s2, s3, s4 = encoder.predict(img, verbose=0)
    emo = to_categorical([target_emotion_idx], len(EMOTION_LABELS))
    recon, _ = decoder.predict([mu_map, emo, s1, s2, s3, s4], verbose=0)
    return (recon * 255).astype(np.uint8).squeeze()


def reconstruct_all_emotions(img: np.ndarray, encoder: Model, decoder: Model) -> List[np.ndarray]:
    return [reconstruct_with_emotion(img, encoder, decoder, i) for i in range(len(EMOTION_LABELS))]





