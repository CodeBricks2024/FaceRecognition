from pydantic import BaseModel
from MatchModel import Match
from fastapi.responses import FileResponse

# API 리스폰스 데이터 모델 정의
class CompareResponse(BaseModel):
    closest_match: str = ""
    closest_match_img: str = ""
    # distance: 두 이미지가 얼마나 동떨어져있는지 확인 (distance가 낮으면 두 이미지가 유사하다는 의미)
    distance: float = 0.0
    emotion: str = ""
    confidence: float = 0.0
    predicted_age: str = ""
    race: str = ""
    distances: list[Match] = []