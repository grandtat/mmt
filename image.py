import numpy as np
import cv2

dst = cv2.imread('macintosh.png', cv2.IMREAD_COLOR_RGB)

gray = cv2.cvtColor(dst, cv2.COLOR_RGB2GRAY)
gray = np.clip((gray+64.0), 0, 255)
gray = gray.astype(np.uint8)

cv2.imshow('image', gray)

cv2.waitKey(0)