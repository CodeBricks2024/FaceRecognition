from deepface import DeepFace
from responseModel import CompareResponse

def face_analyze(img):
    results = DeepFace.analyze(img, actions=['emotion'])

    for face in results:
        print("face check: ", face)
        emotion = face["dominant_emotion"]
        # confidence = face["emotion"][emotion]
        # age = face["age"]
        # race = face["dominant_race"]

        response = CompareResponse()
        response.emotion = emotion
        # response.confidence = confidence
        # response.predicted_age = age
        # response.race = race

        return response