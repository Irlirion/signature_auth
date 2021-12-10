from pathlib import Path

import cv2
import numpy as np
from pytorch_metric_learning.distances import cosine_similarity
import torch
from PIL import Image
from signature_id.utils import Drower
from signature_id.auth.inference import get_inference_model, transform


@torch.inference_mode()
def check_sign(img, model, images):
    image = transform(Image.fromarray(img)).unsqueeze(0)
    distances = model.get_nearest_neighbors(image, 7)[0].flatten()
    scores = sorted(
        model.match_finder.distance(
            model.get_embeddings(images), model.get_embeddings(image)
        )
        .flatten()
        .tolist(),
        reverse=True,
    )
    print("distance =", ", ".join(f"{q:.2f}" for q in distances))
    print("cosine_similarity =", ", ".join(f"{q:.2f}" for q in scores))
    return model.get_matches(images, ref=image, threshold=0.8).any()


if __name__ == "__main__":
    model = get_inference_model()
    root = Path("signature_id", "images")
    images = list(
        transform(Image.open(f)) for f in root.iterdir() if f.suffix == ".jpg"
    )
    if len(images) == 0:
        print("First you have to register by run registration.py")
        exit(0)
    model.train_knn(images)

    img = np.ones((512, 512, 3), np.uint8) * 255
    drower = Drower(img)
    cv2.namedWindow("SignAuth")
    cv2.setMouseCallback("SignAuth", drower.line_drawing)

    while 1:
        cv2.imshow("SignAuth", drower.img)

        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            break
        elif k == ord("p"):
            is_valid = check_sign(drower.img, model, images)
            if is_valid:
                print("Pass")
                break
            else:
                print("Invalid sign")
                drower.img = np.ones((512, 512, 3), np.uint8) * 255

    cv2.destroyAllWindows()
