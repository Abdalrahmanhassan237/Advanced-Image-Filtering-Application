import tkinter as tk
from tkinter import filedialog
from PIL import ImageTk, Image
from image_processor import ImageProcessor
import cv2 as cv
import numpy as np

class Application(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Image Filter Application")
        self.geometry("1920x1200")

        self.image_processor = ImageProcessor()

        # Left Frame
        self.left_frame = tk.Frame(self, width=400, height=400, borderwidth=2, relief=tk.GROOVE)
        self.left_frame.pack(padx=50, pady=100, side=tk.LEFT, fill=tk.Y)

        # Center Frame
        self.center_frame = tk.Frame(self, width=800, height=400)
        self.center_frame.pack(side=tk.LEFT, fill=tk.BOTH)

        # Right Frame
        self.right_frame = tk.Frame(self, width=800, height=600)
        self.right_frame.pack(side=tk.LEFT, fill=tk.BOTH)
        
        # Buttons
        self.create_buttons()

    def create_buttons(self):
        self.open_button = tk.Button(self.left_frame, text="Open Image", command=self.open_image)
        self.open_button.pack(pady=5)

        # Filter Buttons
        filters = ["LPF", "HPF", "Mean", "Median", "Roberts", "Prewitt", "Sobel_x", "Sobel_y", "Erosion", "Dilation", "Hough", "Region Split", "Thresholding"]
        for filter_name in filters:
            button = tk.Button(self.left_frame, text=filter_name, command=lambda filter_name=filter_name: self.apply_filter(filter_name))
            button.pack(side=tk.TOP, pady=5)

    def open_image(self):
        file_path = filedialog.askopenfilename()
        self.original_image = cv.imread(file_path, cv.IMREAD_COLOR)
        self.display_image(self.original_image, self.center_frame)

    def apply_filter(self, filter_name):
        filtered_image = self.image_processor.apply_filter(self.original_image, filter_name)
        self.display_image(filtered_image, self.right_frame)

    def display_image(self, image, frame):
        if isinstance(image, np.ndarray):
            # Convert OpenCV image to PIL Image
            image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
            image = Image.fromarray(image)
        
        image = image.resize((600, 600), Image.LANCZOS)
        photo = ImageTk.PhotoImage(image)
        
        # Clear previous image
        for widget in frame.winfo_children():
            widget.destroy()
        
        # Create and display the label with the image
        label = tk.Label(frame, image=photo)
        label.image = photo  # Keep a reference to prevent garbage collection
        label.pack()

if __name__ == "__main__":
    app = Application()
    app.mainloop()




