from pydantic import BaseModel
class Match(BaseModel):
    image: str
    name: str
    distance: float