import os
import shutil
import json
from PIL import Image
from deepface import DeepFace
import face_detector

w = 255
output_filename = "Samples"

data_dir = "Celeb Dataset"
for directory in os.listdir(data_dir):
        joined = os.path.join(data_dir, directory)
        if os.path.isdir(joined):
                # 사진에 사람 얼굴이 1명만 존재하는지 확인 후, 그렇지 않으면 이미지 삭제
                face_detector.remove_non_single_faces(joined)
                face_detector.crop_face(joined)

                first_file = os.listdir(joined)[0]
                first_file_path = joined+"/"+first_file
                if os.path.isfile(first_file_path):
                        img = Image.open(first_file_path)
                        img = img.resize((w, int(w * (img.height / img.width))))
                        # os.makedirs(os.path.dirname(output_filename), exist_ok=True)
                        img.save(os.path.join(output_filename, f"{directory}.jpg"))
                        # shutil.copyfile(os.path.join(data_dir, directory, first_file), os.path.join("Samples", f"{directory}.jpg"))


smallest_distance = None

for file in os.listdir("Samples"):
        if file.endswith(".jpg"):
                result = DeepFace.verify("person8.jpg", f"Samples/{file}")
                print(json.dumps(result, indent=2))
                if result['verified']:
                        print("This person looks exactly like", file.split(".")[0])
                        break
                if smallest_distance is None:
                        smallest_distance = (file.split(".")[0], result['distance'])
                else:
                        smallest_distance = (file.split(".")[0], result['distance']) if result['distance'] < smallest_distance[1] else smallest_distance
else:
        print(f"No exact match found! Closest match is {smallest_distance[0]}")

# result = DeepFace.verify("person1.jpg", f"Samples/Angelina Jolie.jpg ")
# distance: 두 이미지가 얼마나 동떨어져있는지 확인 (distance가 낮으면 두 이미지가 유사하다는 의미)

