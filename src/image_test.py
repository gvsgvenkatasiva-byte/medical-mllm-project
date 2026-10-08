import cv2

# Read the image
image = cv2.imread("data/test.jpg")

# Check if image was loaded
if image is None:
    print("ERROR: Image could not be loaded.")
else:
    print("SUCCESS: Image loaded!")
    print("Image shape:", image.shape)

    # Display the image
    cv2.imshow("Test Image", image)

    # Wait for a keyboard key
    cv2.waitKey(0)

    # Close the image window
    cv2.destroyAllWindows()