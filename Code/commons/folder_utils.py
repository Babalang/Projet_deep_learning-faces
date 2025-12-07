import os

def initialize_folder()->None:
    home = get_deepface_home()
    deepface_folder_path = os.path.join(home,".deepface")
    weights_path = os.path.join(deepface_folder_path,"weights")

    if not os.path.exists(deepface_folder_path):
        os.makedirs(deepface_folder_path, exist_ok=True)
    
    if not os.path.exists(weights_path):
        os.makedirs(weights_path, exist_ok=True)

def get_deepface_home()->str:
    return str(os.getenv("DEEPFACE_HOME", default=os.path.expanduser("~")))