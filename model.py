import io
import onnxruntime as ort
import numpy as np
from PIL import Image
from torchvision import transforms


def preprocess_numpy(img):
    resize = transforms.Resize((224, 224))
    crop = transforms.CenterCrop((224, 224))
    to_tensor = transforms.ToTensor()
    normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    if img.mode != "RGB":
        img = img.convert("RGB")
    img = resize(img)
    img = crop(img)
    img = to_tensor(img)
    img = normalize(img)
    return img


class OnnxModel:
    def __init__(self, model_path="model.onnx"):
        self.session = ort.InferenceSession(
            model_path, providers=["CPUExecutionProvider"]
        )

    def predict(self, image_bytes: bytes) -> int:
        img = Image.open(io.BytesIO(image_bytes))
        inp = preprocess_numpy(img).unsqueeze(0).numpy().astype(np.float32)
        outputs = self.session.run(None, {"input": inp})
        pred = np.argmax(outputs[0])
        return int(pred)


if __name__ == "__main__":
    model = OnnxModel("model.onnx")

    with open("n01440764_tench.jpeg", "rb") as f:
        image_bytes = f.read()

    class_id = model.predict(image_bytes)
    print(f"Predicted class ID: {class_id}")
