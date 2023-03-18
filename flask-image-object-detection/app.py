import io
import os
from PIL import Image

from PIL import ImageDraw
from flask import send_file

import torch
import torchvision
import torchvision.transforms as transforms
from torchvision import models
from flask import Flask, request, jsonify, render_template
from pycocotools.coco import COCO

app = Flask(__name__)
model = models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
model.eval()


def detect_objects(image_bytes):
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
        ]
    )

    image = Image.open(io.BytesIO(image_bytes))
    image_tensor = transform(image).unsqueeze(0)
    output = model(image_tensor)

    labels = output[0]["labels"].tolist()
    boxes = output[0]["boxes"].tolist()
    scores = output[0]["scores"].tolist()

    # COCO instance category names
    coco_instance_category_names = [
        '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
        'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
        'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
        'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
        'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
        'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
        'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
        'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
        'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
        'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
        'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
        'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
    ]

    # Draw bounding boxes and labels on the image
    draw = ImageDraw.Draw(image)
    for label, box, score in zip(labels, boxes, scores):
        if score > 0.5:  # You can adjust the threshold
            name = coco_instance_category_names[label]
            draw.rectangle(box, outline="red", width=2)
            draw.text((box[0], box[1]), name, fill="red")

    return image



@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/detect", methods=["POST"])
def upload_image():
    if "file" not in request.files:
        return jsonify(error="No file uploaded"), 400

    file = request.files["file"]
    img_bytes = file.read()
    image_with_boxes = detect_objects(img_bytes)

    # Save the image with bounding boxes to a buffer
    buffer = io.BytesIO()
    image_with_boxes.save(buffer, format="JPEG")
    buffer.seek(0)

    return send_file(buffer, mimetype="image/jpeg")


if __name__ == "__main__":
    app.run(debug=True)
