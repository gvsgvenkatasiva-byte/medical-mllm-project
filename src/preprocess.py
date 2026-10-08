import cv2

# Read the original image
image = cv2.imread("data/test.jpg")

if image is None:
    print("Image not found!")
else:
    print("Original size:", image.shape)

    # 1. Resize image
    resized = cv2.resize(image, (224, 224))
    print("Resized size:", resized.shape)

    # 2. Convert to grayscale
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    print("Grayscale size:", gray.shape)

    # 3. Normalize pixel values
    normalized = gray / 255.0

    print("Minimum pixel value:", normalized.min())
    print("Maximum pixel value:", normalized.max())

    # 4. Save processed image
    cv2.imwrite("data/processed.jpg", gray)

    print("Processed image saved!")