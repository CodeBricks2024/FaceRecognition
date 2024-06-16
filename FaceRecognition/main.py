import os
import json
from io import BytesIO
from PIL import Image
from deepface import DeepFace
import numpy as np
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Flatten, MaxPooling2D, Conv2D
from fastapi import FastAPI, File, UploadFile, HTTPException, Form, Depends
from fastapi.responses import JSONResponse
from fastapi.encoders import jsonable_encoder
from pydantic import BaseModel
from typing import Optional
from fastapi.middleware.cors import CORSMiddleware
import asyncio

import face_analyzer
import face_detector
from ImageEncoding import encode_image_to_base64
from MatchModel import Match

# FASTAPI 초기화
app = FastAPI()

# cors 미들웨어 설정
origins = [
    "http://localhost",
    # "http://127.0.0.1",
    "http://192.168.0.13",  # 추가: macOS의 로컬 IP 주소
    "http://114.70.121.21",  # 추가: macOS의 공인 IP 주소
    "http://192.168.0.25",  # 추가: iOS 디바이스 IP 주소
    "https://facefinder-dad2a7cf64e7.herokuapp.com"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 저장된 모델 로드
# saved_model = load_model("FaceFinder_model.h5")

# 라벨 매핑 (예시)
label_map = {'개그맨': 0, '가수': 1, '배우': 2}

# 이미지 사이즈
w = 255
h = 0

# (distance 기준 오름차순) 가장 유사한 인물 순으로 정보 담는 배열
distanceArr: list[Match] = []

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
# def pil_to_np(image):
#     return np.array(image)

# 이미지 전처리 함수
# def preprocess_image(image_path, image_width, image_height):
#     img = Image.open(image_path)
#     img = img.resize((w, int(w * (img.height / img.width))))
#     h = img.height
#     # img = np.expand_dims(img, axis=0)
#     img = np.array(img)
#     if img.shape[-1] == 4:  # PNG 이미지에서 alpha 채널 제거
#         img = img[:, :, :3]
#     return img

# # 이미지를 NumPy 배열로 변환하는 함수
# def preprocess_image(img, target_size=(255, 255)):
#     img = img.resize(target_size)
#     img = np.expand_dims(img, axis=0)
#     img = np.array(img) / 255.0
#     return img


def image_arrange():
    # 이미지 크기 조정 및 샘플 이미지 생성 함수
    for directory in os.listdir(data_dir):
        joined = os.path.join(data_dir, directory)
        if os.path.isdir(joined):
            # 사진에 사람 얼굴이 1명만 존재하는지 확인 후, 그렇지 않으면 이미지 삭제
            # face_detector.remove_non_single_faces(joined)
            face_detector.crop_face(joined)

            first_file = os.listdir(joined)[0]
            sample_file_path = joined + "/" + first_file
            # Celeb Dataset/Lee Eun ji/개그맨 이은지_67.jpg

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


def create_model():
    # # 데이터 분할 (학습, 검증 데이터셋)
    # from sklearn.model_selection import train_test_split
    #
    # train_images, val_images, train_labels, val_labels = train_test_split(samples, labels, test_size=0.2, random_state=42)
    #
    # # train_images와 val_images를 PIL 이미지에서
    # # NumPy 배열로 변환
    # train_images = [pil_to_np(image) for image in train_images]
    # val_images = [pil_to_np(image) for image in val_images]
    #
    # samples = np.array(samples)
    # labels = np.array(labels)
    # results = np.array(results)

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
    # input_dim = (w, w, 3)
    # image_input = ct.ImageType(shape=input_dim, bias=[0, 0, 0], scale=1/255)


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
# input_name = model.input_names[0]  # 첫 번째 입력 텐서의 이름 가져오기 (conv2d_16_input)


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


# 테스트 리퀘스트 모델
class TestRequest(BaseModel):
    # id: int
    id: Optional[int] = None


# 테스트 엔드포인트 정의
@app.post("/test", status_code=200)
def test(request: TestRequest):
    print("test request: ", request)
    idx = request.id
    return JSONResponse(content={"idx": idx})


# 리퀘스트 데이터 모델 정의
class CompareRequest(BaseModel):
    image_file: UploadFile = Form(...)


# 비교 엔드포인트 정의
@app.post("/compare", status_code=200)
async def compare(request: CompareRequest = Depends()):
    await asyncio.sleep(20)  # 비동기로 처리되는 작업 예시
    try:
        # 이미지 파일 읽어오기
        content = await request.image_file.read()
        img = Image.open(BytesIO(content))

        print("img check: ", img)

        # numpy 처리 과정에서 사이즈 바뀜(?)
        # processed_img = preprocess_image(image_path=sample_file_path, image_width=img.width, image_height=img.height)

        # 모델을 사용하여 특징 벡터 추출
        # features = model.predict(processed_img)

        smallest_distance = None
        closest_match = None

        # DeepFace를 사용하여 유사도 계산
        for file in os.listdir(sample_dir):
            if file.endswith(".jpg") or file.endswith(".png"):
                print("file check:", file)
                print("file path check: ", f"Samples/{file}")
                result = DeepFace.verify(np.array(img), f"Samples/{file}")
                print(json.dumps(result, indent=2))
                # results.append(result)
                if result['verified']:
                    print("This person looks exactly like", file.split(".")[0])
                    closest_match = file.split(".")[0]
                    print("Closest match is", file.split(".")[0])
                    smallest_distance = (file.split(".")[0], result['distance'])
                    print("smallest distance: ", smallest_distance)
                    createMatchCollection(closest_match, smallest_distance)
                    break
                if smallest_distance is None:
                    smallest_distance = (file.split(".")[0], result['distance'])
                    closest_match = (file.split(".")[0], result['distance'])

                    print("smallest distance2: ", smallest_distance)
                    createMatchCollection(closest_match[0], smallest_distance)
                else:
                    smallest_distance = (file.split(".")[0], result['distance']) if result['distance'] < \
                                                                                    smallest_distance[
                                                                                        1] else smallest_distance
                    print("smallest distance3: ", smallest_distance)
                    # createMatchCollection(closest_match[0], smallest_distance)
        else:
            print(f"No exact match found! Closest match is {smallest_distance[0]}")
            closest_match = smallest_distance[0]
            createMatchCollection(closest_match, smallest_distance)

        closest_match_img_path = os.path.join(sample_dir, f"{closest_match}.jpg")
        closest_match_img = encode_image_to_base64(closest_match_img_path)

        response = face_analyzer.face_analyze(np.array(img))
        response.closest_match_img = closest_match_img
        response.distance = smallest_distance[1]
        response.closest_match = closest_match
        # distance 값을 기준으로 오름차순 정렬
        response.distances = sorted(distanceArr, key=lambda match: match.distance)

        # distance 배열 초기화
        distanceArr.clear()


        return JSONResponse(content=jsonable_encoder(response))

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


def createMatchCollection(closest_match, smallest_distance):
    if len(distanceArr) >= 5:
        return
    else:
        # name을 기준으로 중복 제거
        seen_names = set()

        for match in distanceArr:
            if match.name in seen_names:
                return

        closest_match_img_path = os.path.join(sample_dir, f"{closest_match}.jpg")
        print("closest image append: ", closest_match_img_path)
        closest_match_img = encode_image_to_base64(closest_match_img_path)
        distanceArr.append(Match(image=closest_match_img, name=smallest_distance[0], distance=smallest_distance[1]))
        seen_names.add(smallest_distance[0])


# 실행
if __name__ == "__main__":
    import uvicorn


    image_arrange()
    # uvicorn.run(app, host="127.0.0.1", port=8000)
    # uvicorn.run(app, host="0.0.0.0", port=8000)
    uvicorn.run(app, host="192.168.0.13", port=8000)
    uvicorn.run(app, host="0.0.0.0", port=8000)
    # uvicorn.run(app, host="192.168.0.13", port=8000)
