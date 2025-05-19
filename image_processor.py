import cv2
import numpy as np
from PIL import Image


class ImageProcessor:
    def __init__(self):
        pass

    def apply_filter(self, image, filter_name):
        """
        Apply various image filters to the input image.

        Args:
            image: PIL Image or numpy array
            filter_name: String specifying the filter to apply

        Returns:
            PIL Image with the filter applied
        """
        # Convert PIL Image to numpy array if needed
        if isinstance(image, Image.Image):
            image_np = np.array(image)
        else:
            image_np = image.copy()

        # Convert to grayscale if not already
        if len(image_np.shape) == 3 and image_np.shape[2] == 3:
            # Convert RGB to grayscale
            image_gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
        else:
            # Already grayscale
            image_gray = image_np.copy()

        result = None
        is_color_output = False  # Default to grayscale output

        # Apply selected filter
        if filter_name == "LPF":  # Low Pass Filter
            # Gaussian blur
            result = cv2.GaussianBlur(image_gray, (5, 5), 0)

        elif filter_name == "HPF":  # High Pass Filter
            kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
            result = cv2.filter2D(image_gray, cv2.CV_32F, kernel)
            # Normalize to 0-255 range
            result = cv2.normalize(result, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

        elif filter_name == "Mean":  # Mean Filter
            # Simple averaging blur with 5x5 kernel
            result = cv2.blur(image_gray, (5, 5))

        elif filter_name == "Median":  # Median Filter
            # Median blur is good for salt-and-pepper noise removal
            result = cv2.medianBlur(image_gray, 5)

        elif filter_name == "Roberts":  # Roberts Cross
            # Classic Roberts Cross edge detector
            kernel_x = np.array([[1, 0], [0, -1]], dtype=np.float32)
            kernel_y = np.array([[0, 1], [-1, 0]], dtype=np.float32)

            # Apply to grayscale image
            gx = cv2.filter2D(image_gray, cv2.CV_32F, kernel_x)
            gy = cv2.filter2D(image_gray, cv2.CV_32F, kernel_y)

            # Compute magnitude
            magnitude = np.sqrt(np.square(gx) + np.square(gy))

            # Normalize to 0-255 range
            result = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

        elif filter_name == "Prewitt":  # Prewitt Operator
            # Classic Prewitt edge detector
            kernel_x = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], dtype=np.float32)
            kernel_y = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]], dtype=np.float32)

            # Apply to grayscale image
            gx = cv2.filter2D(image_gray, cv2.CV_32F, kernel_x)
            gy = cv2.filter2D(image_gray, cv2.CV_32F, kernel_y)

            # Compute magnitude
            magnitude = np.sqrt(np.square(gx) + np.square(gy))

            # Normalize to 0-255 range
            result = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

        elif filter_name == "Sobel_x":  # Sobel X
            # Sobel operator for horizontal edges
            sobel_x = cv2.Sobel(image_gray, cv2.CV_32F, 1, 0, ksize=3)

            # Take absolute value and convert to 8-bit
            abs_sobel_x = cv2.convertScaleAbs(sobel_x)
            result = abs_sobel_x

        elif filter_name == "Sobel_y":  # Sobel Y
            # Sobel operator for vertical edges
            sobel_y = cv2.Sobel(image_gray, cv2.CV_32F, 0, 1, ksize=3)

            # Take absolute value and convert to 8-bit
            abs_sobel_y = cv2.convertScaleAbs(sobel_y)
            result = abs_sobel_y

        elif filter_name == "Erosion":  # Erosion
            # Morphological erosion
            kernel = np.ones((5, 5), np.uint8)
            result = cv2.erode(image_gray, kernel, iterations=1)

        elif filter_name == "Dilation":  # Dilation
            # Morphological dilation
            kernel = np.ones((5, 5), np.uint8)
            result = cv2.dilate(image_gray, kernel, iterations=1)

        elif (
            filter_name == "Hough"
        ):  # Hough Transform - Special case with colored lines
            # Create a 3-channel image (pseudo-color) to draw colored lines on
            result_bgr = cv2.cvtColor(image_gray, cv2.COLOR_GRAY2BGR)

            # First apply Gaussian blur to reduce noise
            blurred = cv2.GaussianBlur(image_gray, (5, 5), 0)

            # Detect edges with Canny - more aggressive parameters
            edges = cv2.Canny(blurred, 50, 150, apertureSize=3)

            # Use probabilistic Hough Transform for better line detection
            lines = cv2.HoughLinesP(
                edges,
                rho=1,
                theta=np.pi / 180,
                threshold=50,  # Lower threshold to detect more lines
                minLineLength=50,  # Minimum line length
                maxLineGap=10,  # Maximum allowed gap between line segments
            )

            if lines is not None:
                for line in lines:
                    x1, y1, x2, y2 = line[0]
                    # Draw red lines on the image
                    cv2.line(result_bgr, (x1, y1), (x2, y2), (0, 0, 255), 2)

            # Convert BGR to RGB for PIL
            result_rgb = cv2.cvtColor(result_bgr, cv2.COLOR_BGR2RGB)
            return Image.fromarray(result_rgb)

        elif (
            filter_name == "Region Split"
        ):  # Region Split - Improved watershed algorithm
            # Create a more robust region split using watershed algorithm

            # Apply bilateral filter to reduce noise while preserving edges
            bilateral = cv2.bilateralFilter(image_gray, 9, 75, 75)

            # Apply Otsu's thresholding
            _, thresh = cv2.threshold(
                bilateral, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
            )

            # Noise removal with morphological operations
            kernel = np.ones((3, 3), np.uint8)
            opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)

            # Sure background area
            sure_bg = cv2.dilate(opening, kernel, iterations=3)

            # Finding sure foreground area with distance transform
            dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
            _, sure_fg = cv2.threshold(
                dist_transform, 0.3 * dist_transform.max(), 255, 0
            )
            sure_fg = sure_fg.astype(np.uint8)

            # Finding unknown region
            unknown = cv2.subtract(sure_bg, sure_fg)

            # Create visual representation of the regions
            # Convert to 3-channel for visualization
            result_visual = cv2.cvtColor(image_gray, cv2.COLOR_GRAY2BGR)

            # Mark regions with colors
            result_visual[sure_fg > 0] = [0, 255, 0]  # Green for foreground
            result_visual[unknown > 0] = [0, 0, 255]  # Red for boundaries

            # Convert BGR to RGB for PIL
            result_rgb = cv2.cvtColor(result_visual, cv2.COLOR_BGR2RGB)
            return Image.fromarray(result_rgb)

        elif filter_name == "Thresholding":  # Thresholding
            # Otsu's thresholding for better results
            _, thresh = cv2.threshold(
                image_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
            )
            result = thresh

        else:
            # If no filter is specified, return the original grayscale image
            result = image_gray.copy()

        # Ensure result is a valid 8-bit image
        if result.dtype != np.uint8:
            result = cv2.normalize(result, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

        # Convert the result to PIL Image (keeping as grayscale)
        # For PIL, we need to use mode 'L' for grayscale images
        return Image.fromarray(result, mode="L")
