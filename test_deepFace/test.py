from deepface import DeepFace
import matplotlib.pyplot as plt
import cv2

models = [
    "VGG-Face", "Facenet", "Facenet512", "OpenFace", "DeepFace",
    "DeepID", "ArcFace", "Dlib", "SFace", "GhostFaceNet",
    "Buffalo_L",
]
img_path= "./imgs_db/degout.jpg"
objs = DeepFace.analyze(
  img_path = img_path, actions = ['age', 'gender', 'race', 'emotion']
)

result = DeepFace.verify(img1_path = "./imgs_db/verif1.jpg", img2_path = "verif2.jpg")

embedding_objs = DeepFace.represent(img_path = img_path)

dfs = DeepFace.find(img_path = "verif2.jpg", db_path = "imgs_db")



# --- Print full results ---
print("\n[DeepFace Analysis Result]")
img = cv2.imread(img_path)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.imshow(img)
plt.axis('off')
plt.title(f"Predicted Emotion: {objs[0]['dominant_emotion']}")
plt.show()
print("\n[Verification Result]")
print(result)
print("\n[Facial Recognition Result]")
if(dfs != []):
  img_query = cv2.imread("./verif2.jpg")
  img_query = cv2.cvtColor(img_query, cv2.COLOR_BGR2RGB)

  img_match = cv2.imread(dfs[0]['identity'][0])
  img_match = cv2.cvtColor(img_match, cv2.COLOR_BGR2RGB)

  plt.figure(figsize=(10, 5))
  plt.subplot(1, 2, 1)
  plt.imshow(img_query)
  plt.axis("off")
  plt.title("Query Image (verif2.jpg)")

  plt.subplot(1, 2, 2)
  plt.imshow(img_match)
  plt.axis("off")
  plt.title(f"Best Match: {dfs[0]['identity'][0]}")

  plt.suptitle("DeepFace Recognition Result", fontsize=14)
  plt.tight_layout()
  plt.show()

# la peur est mal détectée (suprprise) et tristesse aussi (fear)