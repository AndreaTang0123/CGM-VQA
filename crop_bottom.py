from PIL import Image
import os


input_folder = "graphs"


output_folder = "graphs_cropped"
os.makedirs(output_folder, exist_ok=True)


bottom_crop_ratio = 0.15

for filename in os.listdir(input_folder):

    if filename.endswith(".png") or filename.endswith(".jpg"):

        img_path = os.path.join(input_folder, filename)
        img = Image.open(img_path)

        width, height = img.size

        crop_height = int(height * bottom_crop_ratio)

        cropped = img.crop((0, 0, width, height - crop_height))

        save_path = os.path.join(output_folder, filename)
        cropped.save(save_path)

        print("Processed:", filename)

print("All images finished.")