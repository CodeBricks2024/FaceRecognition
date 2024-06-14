import base64
from io import BytesIO
from PIL import Image

def encode_image_to_base64(image_path: str) -> str:
    """
    이미지를 base64 형식으로 인코딩
    Args:
        image_path (str): 이미지 파일 경로
    Returns:
        str: base64 인코딩된 이미지 데이터
    """
    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode("utf-8")
    return encoded_string
