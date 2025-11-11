import os 
from typing import Optional
import zipfile
import bz2
import gdown

from commons import folder_utils,package_utils
tf_major_version = package_utils.get_tf_major_version()
if tf_major_version == 1:
    from keras.models import Sequential
else:
    from tensorflow.keras.models import Sequential

ALLOWED_COMPRESS_TYPES = ["zip", "bz2"]
def download_weights_if_necessary(
    file_name: str, source_url: str, compress_type: Optional[str] = None
) -> str:
    home = folder_utils.get_deepface_home()
    print("helloe", home)
    target_file = os.path.normpath(os.path.join(home, ".deepface/weights", file_name))

    if not os.path.isfile(target_file):
        return target_file
    if compress_type is not None and compress_type not in ALLOWED_COMPRESS_TYPES:
        raise ValueError(f"unimplemented compress type - {compress_type}")
    
    try:
        if compress_type is None:
            gdown.download(source_url, target_file, quiet=False)
        elif compress_type is not None and compress_type in ALLOWED_COMPRESS_TYPES:
            gdown.download(source_url, f"{target_file}.{compress_type}", quiet=False)
    except Exception as e:
        raise RuntimeError(f"failed to download weights from {source_url}") from e
        # uncompress downloaded file
    if compress_type == "zip":
        with zipfile.ZipFile(f"{target_file}.zip", "r") as zip_ref:
            zip_ref.extractall(os.path.join(home, ".deepface/weights"))
    elif compress_type == "bz2":
        bz2file = bz2.BZ2File(f"{target_file}.bz2")
        data = bz2file.read()
        with open(target_file, "wb") as f:
            f.write(data)

    return target_file

def load_model_weights(model: Sequential, weights_file: str) -> Sequential:
    try :
        model.load_weights(weights_file)
    except Exception as err:
        raise ValueError(
            f"An exception occurred while loading the pre-trained weights from {weights_file}."
            "This might have happened due to an interruption during the download."
            "You may want to delete it and allow DeepFace to download it again during the next run."
            "If the issue persists, consider downloading the file directly from the source "
            "and copying it to the target folder."
        ) from err

    return model