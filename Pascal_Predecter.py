#import Predector
# import cv2
# import time

# camera   = cv2.VideoCapture(0)
# time_sec = int(time.time())
# image_path = "kinect/" + str(time_sec) + ".png"
# return_value, image = camera.read()
# cv2.imwrite(image_path, image)
# del(camera)

# cv2.imshow('ImageWindow', image)
# cv2.waitKey()

#Predector.classifier(image_path);



from Tkinter import *  
from matplotlib import pyplot as plt

top = Tk()



Lb1 = Listbox(top)
Lb1.insert(1, "Python")
Lb1.insert(2, "Perl")
Lb1.insert(3, "C")
Lb1.insert(4, "PHP")
Lb1.insert(5, "JSP")
Lb1.insert(6, "Ruby")


Lb1.pack()


top.mainloop()