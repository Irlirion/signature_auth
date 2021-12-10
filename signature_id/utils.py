import cv2


class Drower:
    def __init__(self, img) -> None:
        self.pt1_x = None
        self.pt1_y = None
        self.drawing = False
        self.img = img

    def line_drawing(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.pt1_x, self.pt1_y = x, y

        elif event == cv2.EVENT_MOUSEMOVE:
            if self.drawing == True:
                cv2.line(
                    self.img,
                    (self.pt1_x, self.pt1_y),
                    (x, y),
                    color=(0, 0, 0),
                    thickness=3,
                )
                self.pt1_x, self.pt1_y = x, y
        elif event == cv2.EVENT_LBUTTONUP:
            self.drawing = False
            cv2.line(
                self.img, (self.pt1_x, self.pt1_y), (x, y), color=(0, 0, 0), thickness=3
            )

