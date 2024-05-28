import os
from PIL import Image

# 이미지 사이즈
# 너비, 높이는 너비의 비율에 따라 정해짐
w = 255

# 이미지 데이터 디렉토리
data_dir = "Celeb Dataset"

# Celeb Dataset 이미지 데이터 크기 리사이징하는 함수
def image_data_resize(images, root_dir):
    for image in images:
        image_path = root_dir + "/" + image
        if image.endswith(".jpg") or image.endswith(".png"):
            img = Image.open(image_path)
            img = img.resize((w, int(w * (img.height / img.width))))
            print("img size check: ", img.width, img.height)
            print("resized image:", os.path.join(root_dir, image), img.size)
            # 원래 이미지 리사이징하여 저장
            img.save(os.path.join(root_dir, image))


for directory in os.listdir(data_dir):
    joined = os.path.join(data_dir, directory)
    if os.path.isdir(joined):
        images = os.listdir(joined)
        image_data_resize(images, joined)
