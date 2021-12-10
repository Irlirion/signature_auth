import cv2
import numpy as np
from pathlib import Path

from signature_id.utils import Drower


if __name__ == '__main__':
    img = np.ones((512, 512, 3), np.uint8) * 255
    drower = Drower(img)
    cv2.namedWindow("Registration")
    cv2.setMouseCallback("Registration", drower.line_drawing)

    while 1:
        cv2.imshow("Registration", drower.img)

        k = cv2.waitKey(1)
        if k & 0xFF == 27:
            break
        elif k & 0xFF == ord('s'):
            root = Path("signature_id", "images")
            image_ids = sorted(map(int, (f.stem for f in root.iterdir())))
            image_id = 0 if len(image_ids) == 0 else image_ids[-1] + 1
            path = f"signature_id/images/{image_id}.jpg"
            print(f"Saving image to {path}")
            cv2.imwrite(path, drower.img)   
            drower.img = np.ones((512, 512, 3), np.uint8) * 255

    cv2.destroyAllWindows()