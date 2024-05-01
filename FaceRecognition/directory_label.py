import os
from shutil import move

# 이미지 디렉토리 경로
img_directory = 'Celeb Dataset'

# 라벨과 해당 라벨에 대응하는 디렉토리 매핑
label_mapping = {
    'comedian': '/path/to/comedian_images',
    'singer': '/path/to/singer_images',
    'actor': '/path/to/actor_images'
}

# 이미지 파일을 라벨에 따라 정리하는 함수
def label_images(label, image_files):
    destination_directory = label_mapping[label]
    for image_file in image_files:
        source_path = os.path.join(img_directory, image_file)
        destination_path = os.path.join(destination_directory, image_file)
        move(source_path, destination_path)

# 각 라벨에 대응하는 디렉토리가 없다면 생성
for label, directory in label_mapping.items():
    if not os.path.exists(directory):
        os.makedirs(directory)

# 이미지 파일을 읽어와 라벨에 따라 정리
for root, dirs, files in os.walk(img_directory):
    for file in files:
        if file.endswith('.jpg') or file.endswith('.png'):
            # 이미지 파일 이름에서 라벨 추출
            label = file.split('_')[0]  # 이미지 파일 이름이 '라벨_번호' 형식이라면
            label_images(label, [file])