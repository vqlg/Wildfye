import cv2
import numpy as np
import pandas as pd
color_ranges = {
    'red': ([0, 100, 100], [10, 255, 255]),
    'green': ([40, 40, 40], [70, 255, 255]),
    'blue': ([110, 50, 50], [130, 255, 255])
}
image = cv2.imread('blue.jpg')
hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV) 
for color_name, (lower, upper) in color_ranges.items():
    mask = cv2.inRange(hsv_image, np.array(lower), np.array(upper))
    result = cv2.bitwise_and(image, image, mask=mask)

    if cv2.countNonZero(mask) > 0:
        print(f"Color detected: {color_name}")
        cv2.imshow(f"{color_name}", result)
        cv2.waitKey(0)