import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import shutil
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate
from tensorflow.keras.models import Model
from sklearn.model_selection import train_test_split
import os

# Define the U-Net model
def unet(input_size=(128, 128, 3)):
    inputs = Input(input_size)
    
    # Encoder
    c1 = Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
    p1 = MaxPooling2D((2, 2))(c1)
    c2 = Conv2D(128, (3, 3), activation='relu', padding='same')(p1)
    p2 = MaxPooling2D((2, 2))(c2)
    
    # Bottleneck
    c3 = Conv2D(256, (3, 3), activation='relu', padding='same')(p2)
    
    # Decoder
    u1 = UpSampling2D((2, 2))(c3)
    c4 = Conv2D(128, (3, 3), activation='relu', padding='same')(u1)
    u2 = UpSampling2D((2, 2))(c4)
    c5 = Conv2D(64, (3, 3), activation='relu', padding='same')(u2)
    
    # Output layer
    outputs = Conv2D(1, (1, 1), activation='sigmoid')(c5)
    
    return Model(inputs, outputs)

# Compile the model


#___________________________________________________________________________________________________
def getMask(png):
        # Load the original thermal image (in color)
# Convert the image from BGR to HSV color space
    hsv_image = cv2.cvtColor(png, cv2.COLOR_BGR2HSV)

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
    output_image = png.copy()

# Draw contours around detected red regions
    for contour in contours:
        if cv2.contourArea(contour) > 50:  # Filter small regions
            cv2.drawContours(output_image, [contour], -1, (0, 0, 255), 2)  # Green contour
    for contour in contours:
        if cv2.contourArea(contour) > 50:  # Filter small regions
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(output_image, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Green bounding box

    return output_image

#_______________________________________________________________________________________________________
def maskDirCreation(image_dir, mask_dir):
    x = 1
    for filename in os.listdir(image_dir):
        img_path = os.path.join(image_dir, filename)
        png = cv2.imread(img_path)
        mask = getMask(png)
        cv2.imwrite(filename, mask)
        shutil.copy(filename, mask_dir)
        os.remove(filename)
        x = x+1
#_______________________________________________________________________________________________________
def load_data(image_dir, mask_dir, target_size=(128, 128)):
    images = []
    masks = []
    for filename in os.listdir(image_dir):
        image = cv2.imread(os.path.join(image_dir, filename))
        image = cv2.resize(image, target_size)
        images.append(image)
        
        mask = cv2.imread(os.path.join(mask_dir, filename), cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask, target_size)
        masks.append(mask)
    
    return np.array(images), np.array(masks)

# Load data
def doTheRest():
    model = unet()
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    image_dir = "images"
    mask_dir = "masks"
    maskDirCreation("images", "masks")
    images, masks = load_data(image_dir, mask_dir)

    images = images / 255.0
    masks = masks / 255.0
    masks = masks[..., np.newaxis]  # Add channel dimension
    X_train, X_val, y_train, y_val = train_test_split(images, masks, test_size=0.2, random_state=42)

# Train the model
    model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=10, batch_size=16)

# Save the model
    model.save('hotspot_detector.h5')
# Load the trained model
    model = tf.keras.models.load_model('hotspot_detector.h5')

# Predict on a new image
    test_image = cv2.imread('another_thermal.jpg')
    test_image_resized = cv2.resize(test_image, (128, 128)) / 255.0
    test_image_resized = np.expand_dims(test_image_resized, axis=0)

# Generate prediction
    prediction = model.predict(test_image_resized)

# Threshold the prediction to create a binary mask
    hotspot_mask = (prediction[0, :, :, 0] > 0.5).astype(np.uint8)

# Ensure the mask is resized to match the test image dimensions
    hotspot_mask_resized = cv2.resize(hotspot_mask, (test_image.shape[1], test_image.shape[0]))

# Convert the resized mask to 3 channels (BGR)
    hotspot_mask_colored = cv2.cvtColor(hotspot_mask_resized * 255, cv2.COLOR_GRAY2BGR)

# Overlay the mask onto the original image
    overlay = cv2.addWeighted(test_image, 0.7, hotspot_mask_colored, 0.3, 0)

# Display the result
    plt.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
    plt.title("Hotspot Detection (ML Model)")
    plt.axis('off')
    plt.show()

    return overlay

def doTheRest1(image):
    model = unet()
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    image_dir = "images"
    mask_dir = "masks"
    maskDirCreation("images", "masks")
    images, masks = load_data(image_dir, mask_dir)

    images = images / 255.0
    masks = masks / 255.0
    masks = masks[..., np.newaxis]  # Add channel dimension
    X_train, X_val, y_train, y_val = train_test_split(images, masks, test_size=0.2, random_state=42)

# Train the model
    model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=10, batch_size=16)

# Save the model
    model.save('hotspot_detector.h5')
# Load the trained model
    model = tf.keras.models.load_model('hotspot_detector.h5')

# Predict on a new image
    test_image = image
    test_image_resized = cv2.resize(test_image, (128, 128)) / 255.0
    test_image_resized = np.expand_dims(test_image_resized, axis=0)

# Generate prediction
    prediction = model.predict(test_image_resized)

# Threshold the prediction to create a binary mask
    hotspot_mask = (prediction[0, :, :, 0] > 0.5).astype(np.uint8)

# Ensure the mask is resized to match the test image dimensions
    hotspot_mask_resized = cv2.resize(hotspot_mask, (test_image.shape[1], test_image.shape[0]))

# Convert the resized mask to 3 channels (BGR)
    hotspot_mask_colored = cv2.cvtColor(hotspot_mask_resized * 255, cv2.COLOR_GRAY2BGR)

# Overlay the mask onto the original image
    overlay = cv2.addWeighted(test_image, 0.7, hotspot_mask_colored, 0.3, 0)

# Display the result
    plt.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
    plt.title("Hotspot Detection (ML Model)")
    plt.axis('off')
    plt.show()

    return overlay