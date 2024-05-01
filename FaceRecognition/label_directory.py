import os
import shutil

# 이미지 파일이 들어있는 원본 디렉토리 경로
source_directory = 'Celeb Dataset'

# 각 클래스(레이블)에 해당하는 디렉토리 생성
# 클래스(레이블) 리스트
classes = ['comedian', 'singer', 'actor']

def dir_labeling():
    for class_name in classes:
        os.makedirs(os.path.join(source_directory, class_name), exist_ok=True)

    # 이미지 파일을 해당하는 클래스(레이블)의 디렉토리로 이동
    for filename in os.listdir(source_directory):
        if filename.endswith('.jpg') or filename.endswith('.png'):
            image_path = os.path.join(source_directory, filename)
            # 이미지 파일의 이름에서 클래스(레이블)을 추출하거나 데이터 소스에서 클래스 정보를 얻어와야 합니다.
            # 이 예시에서는 파일 이름을 기준으로 클래스를 추출하는 것으로 가정합니다.
            if 'comedian' in filename:
                target_directory = os.path.join(source_directory, 'comedian')
            elif 'singer' in filename:
                target_directory = os.path.join(source_directory, 'singer')
            elif 'actor' in filename:
                target_directory = os.path.join(source_directory, 'actor')
            else:
                continue  # 다른 클래스(레이블)를 가진 파일은 무시합니다.
            # 이미지 파일을 해당하는 클래스(레이블)의 디렉토리로 이동시킵니다.
            shutil.move(image_path, os.path.join(target_directory, filename))

