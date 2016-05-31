
from Tkinter import *
from matplotlib import pyplot as plt
import time
import math
import Predector
sys.path.append('../libfreenect/wrappers/python')
import freenect
import cv2
import frame_convert2
 

cv2.namedWindow('Video')
print('Press ESC in window to stop')

def predict(img):
    print('save')
    plt.imshow(img, cmap = 'gray', interpolation = 'bicubic')
    plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
    plt.show()
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    pass




def get_video():
    return frame_convert2.video_cv(freenect.sync_get_video()[0])


while 1:
    cv2.imshow('Video', get_video())

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    if cv2.waitKey(30) & 0xFF == ord('s'):
        img = get_video()
        time_sec = int(time.time())
        image_path = "kinect/" + str(time_sec) + ".png"
        cv2.imwrite(image_path,img)
        cv2.destroyAllWindows()
        #predict(img)

        Predector.classifier(image_path);
        break
