"""COCO annotation serialization — pure Python, no bpy."""

import json
import os


def save_coco_dataset(images, annotations, categories, output_path):
    """Write a COCO-format annotations.json.

    Args:
        images:       List of COCO image dicts.
        annotations:  List of COCO annotation dicts.
        categories:   List of COCO category dicts.
        output_path:  Full path to write annotations.json.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    coco = {
        "images": images,
        "annotations": annotations,
        "categories": categories,
    }
    with open(output_path, "w") as f:
        json.dump(coco, f, indent=2)
