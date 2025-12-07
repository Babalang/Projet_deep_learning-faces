# Projet_deep_learning-faces

Résumé rapide
- Projet de génération et modification d'expressions faciales basé sur ACGAN + VAE.
- Contient pipeline d'entraînement/génération/édition.

Arborescence clé
- Code/  
  - Face.py — CLI principal (classifier / vae / gan / analyze)  
  - modules/ACGAN.py — ACGAN, entraînement, génération, inversion, adaptateur  
  - data/fer2013.csv & data/imgs_db/ — dataset et images
- Code/Tentatives_infructueuses/ — scripts expérimentaux

Prérequis
- Linux (tests développés sur Linux)
- Python 3.9 (recommandé)
- venv
- GPU recommandé (TensorFlow + CUDA) mais CPU fonctionne

Installation rapide
1. Creez et activez un environnement virtuel
   ```
   python3 -m venv .venv
   source .venv/bin/activate
   ```
2. Installez les dépendances
   ```
   pip install -r Code/requirements.txt
   ```
   Si vous rencontrez des erreurs de désérialisation de .h5, essayez une version TF compatible (ex. tensorflow==2.9.1) :
   ```
   pip install "tensorflow==2.9.1"
   ```

Utilisation — CLI principal (Code/Face.py)
- Lancer le CLI :
  ```
  cd Code
  python Face.py
  ```
- Choix disponibles :
  - `classifier` : chargement du modèle `emotion_pretrained.h5` ou `emotion_model.h5`, prédiction sur une image.
  - `vae` : reconstruction avec `encoder_model.h5` + `decoder_model.h5`. Permet de reconstruire pour chaque label.
  - `gan` : menu ACGAN (train / generate / modify / modify_crop / adapt / modify_adapter / modify_invert).
Exemples rapides
- Classifier
  ```
  Component: 1
  Image path: Code/data/imgs_db/degout.jpg
  ```
- VAE reconstruct (affiche toutes les émotions générées)
  ```
  Component: 2
  Action: reconstruct
  Image path: data/imgs_db/degout.jpg
  ```
- GAN generate
  ```
  Component: 3
  Action: generate
  Emotion label: 3
  ```

# Authors
Jalbaud Lucas et Langouet Bastian
