import os
import torch
import torch.nn as nn
from torchvision.models import efficientnet_b2


def export_efficientnet(weights_path, onnx_path="efficientnet/effnet_b2.onnx", num_classes=3, opset=12):
    # 1. Load base EfficientNet-B2 (without classifier head)
    model = efficientnet_b2(pretrained=False)

    # 2. Replace classifier to match your dataset
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)

    # 3. Load your trained weights
    state_dict = torch.load(weights_path, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()

    # 4. Dummy input (batch=1, 3x224x224)
    dummy_input = torch.randn(1, 3, 224, 224)

    # 5. Ensure folder exists
    os.makedirs(os.path.dirname(onnx_path), exist_ok=True)

    # 6. Export ONNX
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
        opset_version=opset
    )

    print(f"✅ Exported trained EfficientNet-B2 to {onnx_path}")
    print(f"📌 Classes: {num_classes}")


if __name__ == "__main__":
    # Use your trained model path here
    weights = "efficientnet/effnet_b2.pth"  # or "efficientnet/effnet_checkpoint.pth"
    export_efficientnet(weights, "efficientnet/effnet_b2.onnx", num_classes=3)
