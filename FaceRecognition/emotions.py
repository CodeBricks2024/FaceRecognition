import cv2
from deepface import DeepFace

# 이미지 파일 경로 설정
image_path = r"C:\image crawl\Emma Stone\Image_70.jpg"
db_path = r"C:\image crawl"  # 데이터베이스 이미지들이 저장된 디렉토리 경로

# 이미지 파일 읽기
image = cv2.imread(image_path)

# 얼굴 감정 분석, 나이 추측, 성별 추정 수행
results = DeepFace.analyze(image, actions=["emotion", "age", "gender"])

# 닮은꼴 찾기 수행
similar_faces = DeepFace.find(img_path=image_path, db_path=db_path)

# 분석 결과 출력 및 이미지에 레이블 표시
for face in results:
    emotion = face["dominant_emotion"]
    confidence = face["emotion"][emotion]
    age = face["age"]
    gender = face["gender"]

    print("Emotion:", emotion)
    print("Confidence:", confidence)
    print("Age:", age)
    print("Gender:", gender)

    # 얼굴 영역에 감정, 나이, 성별 레이블과 확률 표시
    x, y, w, h = (
        face["region"]["x"],
        face["region"]["y"],
        face["region"]["w"],
        face["region"]["h"],
    )

    # 텍스트를 세로 형식으로 한 줄씩 표시, 얼굴 박스 바깥에 배치
    label_emotion = f"Emotion: {emotion} ({confidence*100:.2f}%)"
    label_age = f"Age: {age}"
    label_gender_woman = f"Gender (Woman): {gender['Woman']*100:.2f}%"
    label_gender_man = f"Gender (Man): {gender['Man']*100:.2f}%"

    # 텍스트가 얼굴 박스 바깥에 위치하도록 충분히 y 좌표를 떨어뜨림
    y_offset = y - 80 if y - 80 > 20 else y + h + 10

    cv2.putText(
        image,
        label_emotion,
        (x, y_offset),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (36, 255, 12),
        2,
    )
    cv2.putText(
        image,
        label_age,
        (x, y_offset + 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (36, 255, 12),
        2,
    )
    cv2.putText(
        image,
        label_gender_woman,
        (x, y_offset + 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (36, 255, 12),
        2,
    )
    cv2.putText(
        image,
        label_gender_man,
        (x, y_offset + 75),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (36, 255, 12),
        2,
    )
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

# 결과 이미지 저장 (필요한 경우)
output_path = r"C:\image crawl\output_image.png"
cv2.imwrite(output_path, image)

# 결과 이미지 표시
cv2.imshow("Emotion, Age, Gender Analysis", image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 닮은꼴 찾기 결과 출력
print("Most similar faces found in the database:")
print(similar_faces)
