import os
import cv2
import numpy as np

# OpenCV의 얼굴 감지기 & 전신 감지기 준비
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
full_body_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_fullbody.xml')

def crop_face(image_paths):
    imgs = os.listdir(image_paths)
    for img in imgs:
        new_img_path = image_paths + "/" + img
        if new_img_path.endswith(".jpg"):
            # 이미지 로드
            img = cv2.imread(new_img_path)
            # 그레이스케일 변환
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # 사람 감지
            full_bodies = full_body_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
            for (x, y, w, h) in full_bodies:
                # 사람 영역에 대해 얼굴 감지
                roi_gray = gray[y:y+h, x:x+w]
                faces = face_cascade.detectMultiScale(roi_gray)
                # print("check faces:", len(faces), faces)
                if len(faces) > 0:
                    # 첫 번째 얼굴 영역 자르기
                    (fx, fy, fw, fh) = faces[0]
                    face_crop = img[y + fy:y + fy + fh, x + fx:x + fx + fw]

                    # 얼굴 영역을 255x255 크기로 리사이즈
                    face_crop_resized = cv2.resize(face_crop, (255, 255))
                    # 리사이즈한 얼굴 영역 저장
                    cv2.imwrite(f"{img}.jpg", face_crop_resized)

                    # print("outputfile:", os.path.join(data_dir, directory, first_file))
                    # # os.makedirs(os.path.dirname(output_filename), exist_ok=True)
                    # img.save(os.path.join(output_filename, f"{directory}.jpg"))
                    # # shutil.copyfile(os.path.join(data_dir, directory, first_file), os.path.join("Samples", f"{directory}.jpg"))


                    # break  # 첫 번째 얼굴만 처리하고 종료


def remove_non_single_faces(image_paths):
    imgs = os.listdir(image_paths)
    for img in imgs:
        new_img_path = image_paths + "/" + img
        if new_img_path.endswith(".jpg"):
            img = cv2.imread(new_img_path)

            print("img check:", img)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        # 사진에 얼굴이 한 명만 존재하지 않으면 삭제
        if (len(faces) != 1):
            os.remove(new_img_path)