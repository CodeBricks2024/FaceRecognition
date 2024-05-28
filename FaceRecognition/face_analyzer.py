from deepface import DeepFace
import matplotlib.pyplot as plt
import cv2
import json
import pandas as pd


backends = ["opencv", "ssd", "dlib", "retinaface", "mtcnn", "mediapipe"]
#
# fig, axs = plt.subplots(3, 2, figsize=(15, 10))
# axs = axs.flatten()
# for i, b in enumerate(backends):
#     try:
#         face = DeepFace.extract_faces("person3.jpg", target_size=(224, 224), detector_backend=b)
#         axs[i].imshow(face)
#         axs[i].set_title(b)
#         axs[i].axis('off')
#         print(face.shape)
#     except:
#         pass
# plt.show()


models = ["VGG_Face", "Facenet", "Facenet512", "OpenFace", "DeepFace", "DeepID", "ArcFace", "Dlib", "SFace"]

#face_verification
for model in models:

    result = DeepFace.verify(img1_path="person3.jpg", img2_path="person4.jpg", model_name=models[1])
    print(json.dumps(result, indent=2))

    fig, axs = plt.subplots(1, 2, figsize=(15, 5))
    axs[0].imshow(plt.imread('person3.jpg'))
    axs[1].imshow(plt.imread('person4.jpg'))
    fig.suptitle(f"Verified {result['verified']} - Distance {result['distance']:0.4}: Model {model}")

    axs[0].axis("off")
    axs[1].axis("off")
    plt.show()


# find

# result = DeepFace.find(img_path="person3.jpg", db_path="Samples/")
# print(json.dumps(result, indent=2))
# print(result)


# Facial Attribute Analysis

# result = DeepFace.analyze(img_path="person3.jpg")
# print(result)
#
# pd.DataFrame(result["emotion"], index=[0]).T.plot(kind='bar')

from glob import glob
imgs = glob("Samples/*")

# def plot_img_emotion(img, emo_df, figsize=(15, 5)):
#     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#     fig, axs = plt.subplots(1, 2, figsize=figsize)
#     axs[0].axis("off")
#     axs[0].imshow(img)
#     emo_df.sort_values("prediction").plot(kind="barh", figsize=figsize, ax=axs[1])
#     plt.tight_layout()
#     plt.show()
#

# for img in imgs:
#     demography = DeepFace.analyze(img_path=img, detector_backend=backends[4])
#     print(demography[0])
#     emo_df = pd.DataFrame(demography[0]["emotion"], index=[0]))
#     plot_img_emotion(img, emo_df)
#     plt.show()

# Streaming api

# DeepFace.stream(db_path="/Samples", source=0)