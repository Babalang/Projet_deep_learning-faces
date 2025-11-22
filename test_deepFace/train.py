import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
# Désactiver XLA pour éviter grosses allocations initiales
os.environ.pop("TF_XLA_FLAGS", None)

import tensorflow as tf
for gpu in tf.config.list_physical_devices('GPU'):
    try:
        tf.config.experimental.set_memory_growth(gpu, True)
    except:
        pass

from modules.VAE import train_cvae

def main():
    print("============================================================")
    print("ENTRAÎNEMENT CVAE POUR CHANGEMENT D'ÉMOTION")
    print("============================================================")
    data_dir = "./imgs_db/train"
    save_dir = "./models/vae"
    # batch réduit, epochs moins grands pour test
    train_cvae(
        data_dir=data_dir,
        batch_size=16,
        epochs=30,
        lr=1e-3,
        beta=0.4,
        save_dir=save_dir
    )

if __name__ == "__main__":
    main()