import torchvision
from torchvision import transforms as T
import torch
import cv2
from pathlib import Path

from signature_id.auth.inference import crop_space


if __name__ == '__main__':
    resize = T.Resize((128, 128))
    root = Path("signature_id/images")
    images = torch.stack(
        tuple(
            resize(crop_space(torchvision.io.read_image(str(img))))
            for img in root.iterdir()
            if img.suffix == ".jpg"
        )
    )
    grid_img = torchvision.utils.make_grid(images, nrow=5)

    cv2.imshow("GridImages", grid_img.permute(1, 2, 0).numpy())

    while cv2.waitKey(1) != 27:
        pass

    cv2.destroyAllWindows()