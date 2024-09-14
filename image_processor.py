import cv2
import numpy as np
from PIL import Image

class ImageProcessor:
    def __init__(self):
        pass

    def apply_filter(self, image, filter_name):
        # Ensure image is in BGR format for OpenCV operations
        if isinstance(image, np.ndarray):
            image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        else:
            image_bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        image_gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

        if filter_name == "LPF":
            image = cv2.blur(image_bgr, (5, 5))
        elif filter_name == "HPF":
            kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
            image = cv2.filter2D(image_bgr, -1, kernel)
        elif filter_name == "Mean":
            image = cv2.blur(image_bgr, (5, 5))
        elif filter_name == "Median":
            image = cv2.medianBlur(image_bgr, 5)
        elif filter_name == "Roberts":
            kernel_x = np.array([[1, 0], [0, -1]])
            kernel_y = np.array([[0, 1], [-1, 0]])
            gx = cv2.filter2D(image_gray, cv2.CV_64F, kernel_x)
            gy = cv2.filter2D(image_gray, cv2.CV_64F, kernel_y)
            image = np.sqrt(gx**2 + gy**2).astype(np.uint8)
        elif filter_name == "Prewitt":
            kernel_x = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
            kernel_y = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]])
            gx = cv2.filter2D(image_gray, cv2.CV_64F, kernel_x)
            gy = cv2.filter2D(image_gray, cv2.CV_64F, kernel_y)
            image = np.sqrt(gx**2 + gy**2).astype(np.uint8)
        elif filter_name == "Sobel_x":
            image = cv2.Sobel(image_gray, cv2.CV_64F, 1, 0, ksize=5)
            image = np.uint8(np.absolute(image))
        elif filter_name == "Sobel_y":
            image = cv2.Sobel(image_gray, cv2.CV_64F, 0, 1, ksize=5)
            image = np.uint8(np.absolute(image))
        elif filter_name == "Erosion":
            kernel = np.ones((5,5), np.uint8)
            image = cv2.erode(image_bgr, kernel, iterations=1)
        elif filter_name == "Dilation":
            kernel = np.ones((5,5), np.uint8)
            image = cv2.dilate(image_bgr, kernel, iterations=1)
        elif filter_name == "Hough":
            edges = cv2.Canny(image_gray, 50, 150, apertureSize=3)
            lines = cv2.HoughLines(edges, 1, np.pi/180, 200)
            image = np.copy(image_bgr)
            if lines is not None:
                for rho, theta in lines[:, 0]:
                    a = np.cos(theta)
                    b = np.sin(theta)
                    x0 = a * rho
                    y0 = b * rho
                    x1 = int(x0 + 1000 * (-b))
                    y1 = int(y0 + 1000 * (a))
                    x2 = int(x0 - 1000 * (-b))
                    y2 = int(y0 - 1000 * (a))
                    cv2.line(image, (x1, y1), (x2, y2), (0, 0, 255), 2)
        elif filter_name == "Region Split":
            # Implement a simple region split using thresholding
            _, image = cv2.threshold(image_gray, 128, 255, cv2.THRESH_BINARY)
        elif filter_name == "Thresholding":
            _, image = cv2.threshold(image_gray, 127, 255, cv2.THRESH_BINARY)
        
        # Convert back to RGB for PIL
        if len(image.shape) == 2:  # If grayscale
            image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        else:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        return Image.fromarray(image_rgb)