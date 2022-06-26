import cv2
import matplotlib.pyplot as plt

rgb_image = cv2.imread('test_unbalanced/anomaly/0.jpg', cv2.COLOR_BGR2RGB)
greyscale_image = cv2.imread('test_unbalanced/anomaly/0.jpg', cv2.IMREAD_GRAYSCALE)
resized_image = cv2.resize(greyscale_image, (32, 32))
blur_image = cv2.GaussianBlur(resized_image, (9, 9), 0)
threshold_image = cv2.adaptiveThreshold(
    blur_image,
    maxValue=255.0,
    adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    thresholdType=cv2.THRESH_BINARY_INV,
    blockSize=9,
    C=9
)

plt.imshow(rgb_image)
plt.show()

for image in [greyscale_image, resized_image, blur_image, threshold_image]:
    plt.imshow(image, cmap='gray')
    plt.show()