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
    from tensorflow.keras.layers import Conv2D, Flatten, Activation, MaxPooling2D, AveragePooling2D, Dropout, Dense

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