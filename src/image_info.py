import cv2

# Read image
image = cv2.imread("data/test.jpg")

if image is None:
    print("Image not found!")
else:
    print("Image loaded successfully!")

    # Get image dimensions
    height, width, channels = image.shape

    print("Height:", height)
    print("Width:", width)
    print("Channels:", channels)

    # Get total number of pixels
    total_pixels = height * width

    print("Total pixels:", total_pixels)