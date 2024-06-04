from deepface import DeepFace
from responseModel import CompareResponse

def face_analyze(img):
    results = DeepFace.analyze(img)

    for face in results:
        print("face check: ", face)
        emotion = face["dominant_emotion"]
        confidence = face["emotion"][emotion]
        age = face["age"]
        gender = face["gender"]
        race = face["dominant_race"]

        response = CompareResponse()
        response.emotion = emotion
        response.confidence = confidence
        response.predicted_age = age
        response.gender = gender
        response.race = race

        return response