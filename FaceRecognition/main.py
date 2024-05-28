import os
import shutil
import json
from PIL import Image
from deepface import DeepFace
import numpy as np
import face_detector
import coremltools as ct
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Flatten, MaxPooling2D, Conv2D
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse

# FASTAPI 초기화
app = FastAPI()

saved_model = load_model("FaceFinder_model.h5")

# 라벨 매핑 (예시)
label_map = {'개그맨': 0, '가수': 1, '배우': 2}

# 이미지 사이즈
w = 255
h = 0

# 샘플 이미지 디렉토리
sample_dir = "Samples"

# 샘플 이미지를 사용하여 모델 학습
samples = []
labels = []
results = []
label_map = {'개그맨': 0, '가수': 1, '배우': 2}  # 라벨 매핑 예시

data_dir = "Celeb Dataset"
sample_file_path = ""


# PIL 이미지를 NumPy 배열로 변환하는 함수
def pil_to_np(image):
    return np.array(image)

# 이미지 전처리 함수
def preprocess_image(image_path, image_width, image_height):
    img = Image.open(image_path)
    img = img.resize((w, int(w * (img.height / img.width))))
    h = img.height
    # img = np.expand_dims(img, axis=0)
    img = np.array(img)
    if img.shape[-1] == 4:  # PNG 이미지에서 alpha 채널 제거
        img = img[:, :, :3]
    return img



# 이미지 크기 조정 및 샘플 이미지 생성 함수
for directory in os.listdir(data_dir):
        joined = os.path.join(data_dir, directory)
        # image_data_resize(joined)
        if os.path.isdir(joined):
                # 사진에 사람 얼굴이 1명만 존재하는지 확인 후, 그렇지 않으면 이미지 삭제
                # face_detector.remove_non_single_faces(joined)
                # face_detector.crop_face(joined)

                first_file = os.listdir(joined)[0]
                sample_file_path = joined + "/" + first_file
                # Celeb Dataset/Lee Eun ji/개그맨 이은지_67.jpg

                # print("samplefilepath:", sample_file_path)
                if os.path.isfile(sample_file_path):
                        # 이미지 리사이징 및 전처리
                        img = Image.open(sample_file_path)
                        img = img.resize((w, int(w * (img.height / img.width))))
                        h = int(img.height)
                        samples.append(img)
                        labels.append(directory)
                        img.save(os.path.join(sample_dir, f"{directory}.jpg"))


                        # TODO: 개그맨, 가수, 배우별 디렉토리 라벨링
                        # labels.append(label_map[directory])

                        # os.makedirs(os.path.dirname(output_filename), exist_ok=True)
                        # shutil.copyfile(os.path.join(data_dir, directory, first_file), os.path.join("Samples", f"{directory}.jpg"))


# 데이터 분할 (학습, 검증 데이터셋)
from sklearn.model_selection import train_test_split

train_images, val_images, train_labels, val_labels = train_test_split(samples, labels, test_size=0.2, random_state=42)

# train_images와 val_images를 PIL 이미지에서
# NumPy 배열로 변환
train_images = [pil_to_np(image) for image in train_images]
val_images = [pil_to_np(image) for image in val_images]



samples = np.array(samples)
labels = np.array(labels)
results = np.array(results)

# Keras 모델 정의
# 이 모델은 3개의 합성곱 레이어, 최대 풀링 레이어, 완전 연결 레이어로 구성. 입력 이미지의 크기는 (image_height, image_width, 3)로 가정
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(w, h, 3)),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(64, activation='relu'),
    Dense(len(label_map), activation='softmax')
])
# Core ML 모델로 변환할 때 입력 이미지의 차원과 형식을 설정
input_dim = (w, w, 3)
image_input = ct.ImageType(shape=input_dim, bias=[0, 0, 0], scale=1/255)


# 모델 컴파일
# model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# 모델 학습

# model.fit(samples, results, epochs=10, batch_size=32)
# model.fit(train_images, train_labels, epochs=10, batch_size=32, validation_data=(val_images, val_labels))

# 학습된 모델 저장

model.save("FaceFinder_model.h5")
# savedmodel = load_model("FaceFinder_model.h5")

# 모델 요약 출력 (입력 텐서 이름 확인)
model.summary()

# CoreML 변환

# Keras 모델의 입력 텐서 이름 확인
# ct.ImageType의 name 매개변수를 conv2d_16_input으로 설정하여 입력 텐서의 이름을 모델의 실제 입력 레이어와 일치
input_name = model.input_names[0]  # 첫 번째 입력 텐서의 이름 가져오기 (conv2d_16_input)



# Core ML 모델로 변환
# inputs 매개변수를 사용하여 입력 형식을 이미지로 설정
# image_input = ct.converters.mil.input_types.ImageType(name=input_name, shape=(1, w, w, 3))
# coreml_model = ct.convert(
#     model,
#     inputs = [image_input],
#     convert_to="mlprogram"
# )
#
#
# coreml_model.save("FaceFinder")



@app.post("/compare", status_code=200)
def compare(file: UploadFile = File(...)):
    img = Image.open(file.file)
    # numpy 처리 과정에서 사이즈 바뀜(?)
    # processed_img = preprocess_image(image_path=sample_file_path, image_width=img.width, image_height=img.height)

    # 모델을 사용하여 특징 벡터 추출
    # features = model.predict(processed_img)

    smallest_distance = None
    closest_match = None

    # 유사도 계산 함수
    # DeepFace를 사용하여 유사도 계산
    for file in os.listdir(sample_dir):
        if file.endswith(".jpg") or file.endswith(".png"):
            print("file check:", file)
            result = DeepFace.verify(np.array(img), f"Samples/{file}")
            print(json.dumps(result, indent=2))
            # results.append(result)
            if result['verified']:
                print("This person looks exactly like", file.split(".")[0])
                closest_match = file.split(".")[0]
                print("Closest match is", file.split(".")[0])
                smallest_distance = (file.split(".")[0], result['distance'])
                break
            if smallest_distance is None:
                smallest_distance = (file.split(".")[0], result['distance'])
                closest_match = (file.split(".")[0], result['distance'])
            else:
                smallest_distance = (file.split(".")[0], result['distance']) if result['distance'] < smallest_distance[
                    1] else smallest_distance
    else:
        print(f"No exact match found! Closest match is {smallest_distance[0]}")
        closest_match = smallest_distance[0]

    # distance: 두 이미지가 얼마나 동떨어져있는지 확인 (distance가 낮으면 두 이미지가 유사하다는 의미)

    return JSONResponse(content={"closest_match":closest_match, "distance":smallest_distance})




# 실행
# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="127.0.0.1", port=8000)