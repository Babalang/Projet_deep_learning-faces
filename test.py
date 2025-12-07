# exécute depuis la racine du repo
import h5py, json
p = "Code/decoder_model.h5"
with h5py.File(p,'r') as f:
    print("attrs keys:", list(f.attrs.keys()))
    print("keras_version:", f.attrs.get("keras_version"))
    raw = f.attrs.get("model_config")
    if raw:
        cfg = json.loads(raw.decode() if isinstance(raw,(bytes,bytearray)) else raw)
        s = json.dumps(cfg)
        print("'groups' in config:", "'groups'" in s)