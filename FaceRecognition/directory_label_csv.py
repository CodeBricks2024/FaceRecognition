import os
import pandas as pd
from shutil import move

# 이미지 파일이 들어있는 원본 디렉토리 경로
source_directory = 'Celeb Dataset'

# 레이블을 포함한 이미지 파일 리스트 생성
image_files = []
for root, dirs, files in os.walk(source_directory):
    for file in files:
        if file.endswith('.jpg') or file.endswith('.png'):
            image_files.append({'file_path': os.path.join(root, file), 'label': None})

# 이미지 파일 리스트를 데이터프레임으로 변환
df = pd.DataFrame(image_files)

# 레이블을 입력 받아 데이터프레임에 추가하는 함수 정의
def label_images(label):
    for index, row in df.iterrows():
        print(f"Processing image {index+1}/{len(df)}")
        move(row['file_path'], os.path.join(source_directory, label))  # 이미지 파일을 레이블에 해당하는 디렉토리로 이동
        df.at[index, 'label'] = label  # 데이터프레임에 레이블 정보 추가

# 레이블을 입력받아 이미지 파일을 해당하는 레이블 디렉토리로 이동 및 레이블링
labels = ['comedian', 'singer', 'actor']  # 레이블 리스트
for label in labels:
    label_images(label)

# 결과를 CSV 파일로 저장
df.to_csv('/path/to/labelled_images.csv', index=False)