# Advanced-Image-Filtering-Application
![image](https://github.com/user-attachments/assets/80353a5d-d8c3-43cc-aa9b-128223953a8f)



## Overview
This Image Processing Application is a powerful and user-friendly tool designed for applying various image filters and transformations. Built with Python, it leverages popular libraries such as OpenCV, NumPy, and Pillow to provide a robust set of image processing capabilities.

## Features
- **User-friendly GUI**: Built with Tkinter for easy navigation and operation.
- **Multiple Filter Options**: Includes a wide range of filters such as:
  - Low Pass Filter (LPF)
  - High Pass Filter (HPF)
  - Mean Filter
  - Median Filter
  - Roberts Edge Detection
  - Prewitt Edge Detection
  - Sobel Edge Detection (X and Y directions)
  - Erosion
  - Dilation
  - Hough Transform
  - Region Split
  - Thresholding
- **Image Loading**: Ability to open and process various image formats.
- **Real-time Processing**: Instantly view the results of applied filters.
- **Side-by-side Comparison**: Original and processed images displayed simultaneously.

## Technical Details
- **Language**: Python
- **Main Libraries**:
  - OpenCV (cv2): For core image processing operations
  - NumPy: For numerical operations on image arrays
  - Pillow (PIL): For image handling and additional processing
  - Tkinter: For the graphical user interface

## How to Use
1. Run the application.
2. Click "Open Image" to select an image file.
3. Choose a filter from the available options on the left panel.
4. The original image will be displayed in the center, and the filtered image on the right.

## Installation
1. Ensure Python is installed on your system.
2. Install required libraries:
   ```
   pip install opencv-python numpy pillow
   ```
3. Run the main script:
   ```
   python main.py
   ```

## Future Enhancements
- Add more advanced filters and transformations
- Implement batch processing capabilities
- Add option to save processed images
- Introduce custom filter creation functionality

## Contributing
Contributions to improve the application are welcome. Please fork the repository and submit a pull request with your changes.
