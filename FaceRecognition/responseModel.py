from pydantic import BaseModel

# API 리스폰스 데이터 모델 정의
class CompareResponse(BaseModel):
    closest_match: str = ""
    # distance: 두 이미지가 얼마나 동떨어져있는지 확인 (distance가 낮으면 두 이미지가 유사하다는 의미)
    distance: float = 0.0
    emotion: str = ""
    confidence: str = ""
    predicted_age: str = ""
    race: str = ""
    gender: str = ""
