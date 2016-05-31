from Tkinter import *
import cv2
from matplotlib import pyplot as plt
import time
import math
import Predector


def predict(img):
    print('save')
    plt.imshow(img, cmap = 'gray', interpolation = 'bicubic')
    plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
    plt.show()
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    pass


cap = cv2.VideoCapture(0)

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Our operations on the frame come here
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2BGRA)
	
    cv2.putText(img,'Kinect Live',(10,50), cv2.FONT_HERSHEY_PLAIN , 4,(255,255,255),2,cv2.CV_AA)

	# Display the resulting frame
    cv2.imshow('frame',img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    if cv2.waitKey(30) & 0xFF == ord('s'):
        time_sec = int(time.time())
        image_path = "kinect/" + str(time_sec) + ".png"
        cv2.imwrite(image_path,img)
        cap.release()
        cv2.destroyAllWindows()
        #predict(img)

        Predector.classifier(image_path);
        break