
from ultralytics import YOLO
from PIL import Image


class SkinDiseasePredictor:

    def __init__(self, model_path):
        self.model = YOLO(model_path)

        if self.model.task != "classify":
            raise ValueError(
                "The supplied YOLO model is not a classification model."
            )

    def predict(self, image: Image.Image):

        image = image.convert("RGB")

        results = self.model.predict(
            source=image,
            verbose=False
        )

        result = results[0]

        if result.probs is None:
            raise ValueError("No classification probabilities returned.")

        probabilities = result.probs
        class_indices = probabilities.top5
        confidences = probabilities.top5conf

        predictions = []

        for index, confidence in zip(class_indices, confidences):

            class_name = result.names[int(index)]

            predictions.append({
                "disease": class_name,
                "confidence": float(confidence) * 100
            })

        return {
            "predicted_disease": predictions[0]["disease"],
            "confidence": predictions[0]["confidence"],
            "all_predictions": predictions
        }

    def get_classes(self):
        return self.model.names
