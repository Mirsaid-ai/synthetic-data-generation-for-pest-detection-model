"""Model factory functions for DETR object detection."""

from transformers import (
    DetrForObjectDetection,
    DetrImageProcessor,
)

from training.config import (
    DETR_MODEL_NAME, DETR_NUM_CLASSES, DETR_ID_TO_LABEL, DETR_LABEL_TO_ID,
)


# ---------------------------------------------------------------------------
# DETR — object detection with bounding boxes (3 pest classes)
# ---------------------------------------------------------------------------

def create_detr_model(model_name: str = DETR_MODEL_NAME):
    """Create a DETR model and its processor for fine-tuning or inference.

    Returns:
        (model, processor) — both ready to use.
    """
    processor = DetrImageProcessor.from_pretrained(model_name)
    model = DetrForObjectDetection.from_pretrained(
        model_name,
        num_labels=DETR_NUM_CLASSES,
        id2label=DETR_ID_TO_LABEL,
        label2id=DETR_LABEL_TO_ID,
        ignore_mismatched_sizes=True,
    )
    return model, processor


def apply_freeze_strategy(model, freeze_backbone: bool, partial_freeze: int):
    """Control which parts of the DETR backbone are trained.

    Args:
        freeze_backbone: Freeze entire backbone (head-only training).
        partial_freeze: Number of backbone stages to unfreeze from the end.
                        Ignored when freeze_backbone is True.
    """
    if freeze_backbone:
        print("Strategy: HEAD-ONLY (backbone frozen)")
        for param in model.model.backbone.parameters():
            param.requires_grad = False
    elif partial_freeze > 0:
        print(f"Strategy: PARTIAL FINE-TUNE (unfreezing last {partial_freeze} backbone stages)")
        for param in model.model.backbone.parameters():
            param.requires_grad = False
        stages = ["layer1", "layer2", "layer3", "layer4"]
        for stage_name in stages[-partial_freeze:]:
            stage = getattr(model.model.backbone.conv_encoder.model, stage_name, None)
            if stage is not None:
                for param in stage.parameters():
                    param.requires_grad = True
    else:
        print("Strategy: FULL FINE-TUNE (all parameters trainable)")

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"  Trainable: {trainable:,} / {total:,} ({100 * trainable / total:.1f}%)")
