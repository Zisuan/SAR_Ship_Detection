import json
import os
from PIL import Image, ImageDraw, ImageTk
import tkinter as tk
from tkinter import ttk
from itertools import islice


def visualize_json_on_image_batch(json_paths, image_folder):
    root = tk.Tk()
    root.title("Image Annotation Visualization")
    main_frame = ttk.Frame(root)
    main_frame.grid(column=0, row=0, padx=10, pady=10)

    num_cols = 5  # Number of images per row
    for i, json_path in enumerate(json_paths):
        with open(json_path, 'r') as f:
            data = json.load(f)

        image_path = os.path.join(image_folder, data['imagePath'])
        image = Image.open(image_path)
        draw = ImageDraw.Draw(image)

        for shape in data['shapes']:
            label = shape['label']
            points = shape['points']
            shape_type = shape['shape_type']

            if shape_type == 'polygon':
                converted_points = [tuple(point) for point in points]  # Convert the list of lists to a list of tuples
                draw.polygon(converted_points, outline=(0, 255, 0, 128))

        row = i // num_cols
        col = i % num_cols
        img_tk = ImageTk.PhotoImage(image)

        panel = ttk.Label(main_frame, image=img_tk)
        panel.grid(row=row, column=col, padx=5, pady=5)

        panel.image = img_tk

    root.mainloop()


if __name__ == '__main__':
    folder_path = "SSDD_coco" # Replace with your dataset path
    json_files = [file for file in os.listdir(folder_path) if file.endswith('.json')]

    for i in range(0, len(json_files), 10):
        batch_files = json_files[i:i + 10]
        json_paths = [os.path.join(folder_path, file) for file in batch_files]
        visualize_json_on_image_batch(json_paths, folder_path)
