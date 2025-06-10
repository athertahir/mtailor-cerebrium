import pytest
from model import OnnxModel, preprocess_numpy
from PIL import Image


MODEL_PATH = "model.onnx"
IMAGE_TENCH = "n01440764_tench.jpeg"  # Expected class: 0
IMAGE_MUD_TURTLE = "n01667114_mud_turtle.JPEG"  # Expected class: 35


@pytest.fixture(scope="session")
def model():
    return OnnxModel(MODEL_PATH)


def test_model_loads(model):
    """Test that the ONNX model loads correctly."""
    assert model.session is not None


def test_preprocessing_output_shape():
    """Test that image preprocessing outputs the expected shape."""
    img = Image.open(IMAGE_TENCH)
    arr = preprocess_numpy(img)
    assert arr.shape == (3, 224, 224)


def test_inference_tench(model):
    """Model should correctly predict class 0 for tench image."""
    with open(IMAGE_TENCH, "rb") as f:
        class_id = model.predict(f.read())
    assert class_id == 0


def test_inference_mud_turtle(model):
    """Model should correctly predict class 35 for mud turtle image."""
    with open(IMAGE_MUD_TURTLE, "rb") as f:
        class_id = model.predict(f.read())
    assert class_id == 35


def test_invalid_image_fails(model):
    """Model should raise an error on invalid image input."""
    with pytest.raises(Exception):
        model.predict(b"not an image")
