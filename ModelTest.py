import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the original thermal image (in color)
thermal_image = cv2.imread('thermal_image.jpg')

# Convert the image from BGR to HSV color space
hsv_image = cv2.cvtColor(thermal_image, cv2.COLOR_BGR2HSV)

# Define the HSV range for the color red
# Red often wraps around 0 and 180 in the HSV hue scale
lower_red1 = np.array([0, 50, 50])     # Lower range for red (hue ~0)
upper_red1 = np.array([20, 255, 255])
lower_red2 = np.array([170, 50, 50])   # Upper range for red (hue ~180)
upper_red2 = np.array([180, 255, 255])

# Create masks for red regions
mask1 = cv2.inRange(hsv_image, lower_red1, upper_red1)
mask2 = cv2.inRange(hsv_image, lower_red2, upper_red2)
red_mask = cv2.bitwise_or(mask1, mask2)

# Perform morphological operations to clean up the mask
kernel = np.ones((5, 5), np.uint8)
red_mask_cleaned = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel)

# Find contours of red regions
contours, _ = cv2.findContours(red_mask_cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Copy the original image to draw contours on it
output_image = thermal_image.copy()

# Draw contours around detected red regions
for contour in contours:
    if cv2.contourArea(contour) > 50:  # Filter small regions
        cv2.drawContours(output_image, [contour], -1, (0, 0, 255), 2)  # Green contour
for contour in contours:
    if cv2.contourArea(contour) > 50:  # Filter small regions
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(output_image, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Green bounding box

# Display the results
plt.imshow(cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB))
plt.title("Hotspot Detection")
plt.axis('off')
plt.show()

# Save the output image
cv2.imwrite('hotspot_detected_contours.jpg', output_image)

