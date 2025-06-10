import torch
from pytorch_model import Classifier, BasicBlock


def main():
    mtailor = Classifier(BasicBlock, [2, 2, 2, 2])
    # Load provided weights
    state_dict = torch.load("resnet18-f37072fd.pth", map_location="cpu")
    mtailor.load_state_dict(state_dict)
    mtailor.eval()

    example_input = torch.randn(1, 3, 224, 224)

    # Export to ONNX
    torch.onnx.export(
        mtailor,
        example_input,
        "model.onnx",
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
        opset_version=11,
    )
    print("ONNX model exported as model.onnx")


if __name__ == "__main__":
    main()
