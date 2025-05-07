import numpy as np
import cv2 

dst = cv2.imread('testimg.png', cv2.IMREAD_COLOR_RGB)

#for i in range(np.shape(dst)[0]):
#    for j in range(np.shape(dst)[1]):
#        rgb=dst[i][j]
#        dst[i][j][2] = 0#Pixel i,j Wert 0 entspricht Rot
b,g,r = cv2.split(dst)
r = np.clip((r*0.8),0,255)
r = r.astype(np.uint8)
img = cv2.merge((b,g,r))


#gray = cv2.cvtColor(dst, cv2.COLOR_RGB2GRAY)
#gray = np.clip(((gray-64.0)*5.0)+64.0,0,255)
#gray = np.clip(gray+64,0,255)
#gray = gray.astype(np.uint8)
cv2.imshow("Window",img)
cv2.waitKey(0) #input() funktioniert nicht