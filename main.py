import tkinter as tk
from tkinter import filedialog
from PIL import ImageTk, Image
from image_processor import ImageProcessor
import cv2 as cv
import numpy as np
import os


class Application(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Image Filter Application")
        self.geometry("1200x700")  # Adjusted for better fit

        self.image_processor = ImageProcessor()
        self.original_image = None  # Will store PIL Image object
        self.original_cv_image = None  # Will store OpenCV image format
        self.grayscale_image = None  # Will store grayscale PIL Image
        self.current_filtered_image = None  # Will store currently filtered image
        self.current_filter = None  # Track current filter
        self.filter_count = 0  # Track how many times current filter has been applied

        # Left Frame for buttons
        self.left_frame = tk.Frame(
            self, width=200, height=600, borderwidth=2, relief=tk.GROOVE
        )
        self.left_frame.pack(padx=10, pady=10, side=tk.LEFT, fill=tk.Y)

        # Center Frame for original image
        self.center_frame = tk.Frame(
            self, width=450, height=600, borderwidth=2, relief=tk.GROOVE
        )
        self.center_frame.pack(
            padx=10, pady=10, side=tk.LEFT, fill=tk.BOTH, expand=True
        )
        self.center_label = tk.Label(self.center_frame, text="Grayscale Image")
        self.center_label.pack(pady=5)
        self.image_label_original = tk.Label(self.center_frame)
        self.image_label_original.pack(pady=5, expand=True)

        # Right Frame for filtered image
        self.right_frame = tk.Frame(
            self, width=450, height=600, borderwidth=2, relief=tk.GROOVE
        )
        self.right_frame.pack(padx=10, pady=10, side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.right_label = tk.Label(self.right_frame, text="Filtered Image")
        self.right_label.pack(pady=5)
        self.image_label_filtered = tk.Label(self.right_frame)
        self.image_label_filtered.pack(pady=5, expand=True)

        # Create buttons
        self.create_buttons()

        # Status bar
        self.status_bar = tk.Label(
            self, text="Ready", bd=1, relief=tk.SUNKEN, anchor=tk.W
        )
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)

    def create_buttons(self):
        # Open image button
        open_frame = tk.Frame(self.left_frame)
        open_frame.pack(pady=10, fill=tk.X)
        self.open_button = tk.Button(
            open_frame, text="Open Image", command=self.open_image
        )
        self.open_button.pack(fill=tk.X, padx=10)

        # Save result button
        save_frame = tk.Frame(self.left_frame)
        save_frame.pack(pady=5, fill=tk.X)
        self.save_button = tk.Button(
            save_frame, text="Save Result", command=self.save_image
        )
        self.save_button.pack(fill=tk.X, padx=10)

        # Separator
        separator = tk.Frame(self.left_frame, height=2, bg="gray")
        separator.pack(fill=tk.X, padx=10, pady=5)

        # Filter buttons
        filter_label = tk.Label(self.left_frame, text="Select Filter:")
        filter_label.pack(pady=5)

        # Filter Buttons
        filters = [
            "LPF",
            "HPF",
            "Mean",
            "Median",
            "Roberts",
            "Prewitt",
            "Sobel_x",
            "Sobel_y",
            "Erosion",
            "Dilation",
            "Hough",
            "Region Split",
            "Thresholding",
        ]

        for filter_name in filters:
            button = tk.Button(
                self.left_frame,
                text=filter_name,
                command=lambda name=filter_name: self.apply_filter(name),
                width=15,
            )
            button.pack(pady=3, padx=10)

    def open_image(self):
        """Open an image file, convert to grayscale, and display it in the center frame"""
        try:
            file_path = filedialog.askopenfilename(
                filetypes=[
                    ("Image files", "*.jpg *.jpeg *.png *.bmp *.tif *.tiff"),
                    ("All files", "*.*"),
                ]
            )

            if not file_path:  # User cancelled
                return

            # Store the image in both OpenCV and PIL formats
            self.original_cv_image = cv.imread(file_path)
            if self.original_cv_image is None:
                raise ValueError("Could not open image file")

            # Convert to RGB for PIL
            rgb_image = cv.cvtColor(self.original_cv_image, cv.COLOR_BGR2RGB)
            self.original_image = Image.fromarray(rgb_image)

            # Convert to grayscale
            gray_cv_image = cv.cvtColor(self.original_cv_image, cv.COLOR_BGR2GRAY)
            self.grayscale_image = Image.fromarray(gray_cv_image)
            self.current_filtered_image = (
                self.grayscale_image
            )  # Initialize the current filtered image

            # Display the grayscale image
            self.display_image(self.grayscale_image, self.image_label_original)

            # Clear the filtered image when a new image is loaded
            self.clear_filtered_image()

            # Update status
            filename = os.path.basename(file_path)
            self.status_bar.config(
                text=f"Loaded and converted to grayscale: {filename}"
            )

        except Exception as e:
            self.status_bar.config(text=f"Error: {str(e)}")
            print(f"Error opening image: {e}")

    def apply_filter(self, filter_name):
        """Apply the selected filter to the current image and display the result"""
        if self.grayscale_image is None:
            self.status_bar.config(text="Please open an image first")
            return

        try:
            # Check if this is the same filter as before
            if filter_name == self.current_filter:
                # Incrementing filter count for the same filter
                self.filter_count += 1
                input_image = (
                    self.current_filtered_image
                )  # Use the current filtered image as input
                status_text = f"Reapplying {filter_name} filter (x{self.filter_count})"
            else:
                # New filter selected, reset count
                self.filter_count = 1
                input_image = (
                    self.grayscale_image
                )  # Start from the original grayscale image
                status_text = f"Applying {filter_name} filter"

            self.status_bar.config(text=status_text)
            self.update_idletasks()  # Update the UI

            # Apply the filter using our processor
            filtered_image = self.image_processor.apply_filter(input_image, filter_name)

            # Store the result as the current filtered image for potential reapplication
            self.current_filtered_image = filtered_image

            # Display the filtered image
            self.display_image(filtered_image, self.image_label_filtered)

            # Store the current filter name
            self.current_filter = filter_name

            # Update status
            self.status_bar.config(text=f"{status_text} - Complete")

        except Exception as e:
            self.status_bar.config(text=f"Error applying filter: {str(e)}")
            print(f"Error applying filter: {e}")

    def display_image(self, image, label_widget):
        """Display a PIL Image in the specified label widget"""
        if image is None:
            return

        # Calculate appropriate size for display
        max_width = 400
        max_height = 400

        # Get the original image dimensions
        width, height = image.size

        # Calculate aspect ratio
        aspect_ratio = width / height

        # Determine new dimensions while preserving aspect ratio
        if width > height:
            new_width = min(width, max_width)
            new_height = int(new_width / aspect_ratio)
        else:
            new_height = min(height, max_height)
            new_width = int(new_height * aspect_ratio)

        # Resize image for display
        display_image = image.resize((new_width, new_height), Image.LANCZOS)

        # Convert to PhotoImage for Tkinter
        photo = ImageTk.PhotoImage(display_image)

        # Update the label with the new image
        label_widget.config(image=photo)
        label_widget.image = photo  # Keep a reference to prevent garbage collection

    def clear_filtered_image(self):
        """Clear the filtered image display"""
        self.image_label_filtered.config(image="")
        self.current_filter = None
        self.filter_count = 0
        self.current_filtered_image = (
            self.grayscale_image if self.grayscale_image else None
        )

    def save_image(self):
        """Save the filtered image if available"""
        if self.image_label_filtered.image is None:
            self.status_bar.config(text="No filtered image to save")
            return

        try:
            # Get the current filtered image (already processed)
            filtered_image = self.current_filtered_image

            # Ask for save location
            file_path = filedialog.asksaveasfilename(
                defaultextension=".png",
                filetypes=[
                    ("PNG files", "*.png"),
                    ("JPEG files", "*.jpg"),
                    ("All files", "*.*"),
                ],
            )

            if not file_path:  # User cancelled
                return

            # Save the image
            filtered_image.save(file_path)

            # Update status
            filename = os.path.basename(file_path)
            self.status_bar.config(text=f"Saved: {filename}")

        except Exception as e:
            self.status_bar.config(text=f"Error saving image: {str(e)}")
            print(f"Error saving image: {e}")


if __name__ == "__main__":
    app = Application()
    app.mainloop()
