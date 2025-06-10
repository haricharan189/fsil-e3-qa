from PIL import Image
import os
from PIL import Image, ImageChops

input_dir = "/Users/vidhyakshayakannan/Desktop/figures"
output_dir = "pdf_output"
os.makedirs(output_dir, exist_ok=True)

def trim(im):
    bg = Image.new(im.mode, im.size, im.getpixel((0,0)))
    diff = ImageChops.difference(im, bg)
    bbox = diff.getbbox()
    return im.crop(bbox) if bbox else im

for filename in os.listdir(input_dir):
    if filename.endswith(".png"):
        path = os.path.join(input_dir, filename)
        img = Image.open(path).convert("RGBA")
        white_bg = Image.new("RGBA", img.size, (255, 255, 255, 255))
        white_bg.paste(img, (0, 0), img)
        cropped = trim(white_bg.convert("RGB"))
        output_path = os.path.join(output_dir, f"{os.path.splitext(filename)[0]}.pdf")
        cropped.save(output_path, "PDF")
