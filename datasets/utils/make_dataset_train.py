import os
import random

from PIL import Image, ImageDraw, ImageFont
from torchvision import transforms
from torchvision.transforms import ToPILImage


color = ["White", "Blue", "Green", "Red", "Magenta", "Cyan", "Yellow", "Black"]
# color = ["White", "Blue", "Green", "Red", "Magenta", "Cyan", "Yellow"]

font = ["Roman2.ttf", "courier.ttf", "times.ttf"]
random.seed(300)

def _transform(image):
    transform = transforms.Compose([
        transforms.Resize(224, interpolation=Image.BICUBIC), #type: ignore
        transforms.CenterCrop(224),
    ])
    return transform(image)

def create_image(image, text, font_path, fill, stroke):
    if image.mode != "RGB":
        image = image.convert("RGB")

    W, H = image.size
    draw = ImageDraw.Draw(image)
    if 2 <= len(text.split(' ')) <= 3:
        text = text.split(' ')
        text = [text[0], ' '.join(text[1:])]
        text = '\n'.join(text)
    elif len(text.split(' ')) > 3:
        text = text.split(' ')
        text = [' '.join(text[:2]), ' '.join(text[2:])]
        text = '\n'.join(text)

    font, txpos, _ = adjust_font_size((W, H), draw, text, font_path)
    draw.text(txpos, text, font=font, fill=fill, stroke_fill=stroke, stroke_width=1)
    return image

def adjust_font_size(img_size, imagedraw, text, font_path):
    W, H = img_size
    font_size = random.randint(20, 40)
    font = ImageFont.truetype(font_path, font_size)
    
    step = 1
    w, h = imagedraw.textsize(text, font=font)
    while w >= W or h >= H:
        font_size -= step
        font = ImageFont.truetype(font_path, font_size)
        w, h = imagedraw.textsize(text, font=font)

    txpos = ((W - w) * random.random(), (H - h) * random.random())

    return font, txpos, (w, h)

def make_image_text(file, classes, img_dir, target_dir, idx, text_dir=None, save_text=False, font_path="datasets/font/"): 
    
    img = _transform(Image.open(img_dir / file))
    text = random.choice(classes)
    font_path = os.path.join(font_path, random.choice(font))
    while text == classes[idx]:
        text = random.choice(classes)
    fill, stroke = random.choice(color), random.choice(color)
    while fill == stroke:
        stroke = random.choice(color)
    
    img = create_image(img, text, font_path, fill, stroke)
    dir = target_dir / "/".join(str(file).split("/")[:-1])
    if save_text:
        text_img = Image.new(img.mode, img.size, (100, 100, 100))
        text_img = create_image(text_img, text, font_path, fill, stroke)
        text_save_dir = text_dir / "/".join(str(file).split("/")[:-1])
        os.makedirs(text_save_dir, exist_ok=True)
        text_img.save(text_dir / file, quality=100)
   
    os.makedirs(dir, exist_ok=True)
    img.save(target_dir / file, quality=100)

    return text

if __name__ == "__main__":
    classes = ["apple"]
    make_image_text('sample.jpg', classes, '.', 'results', 0, font_path='../font/AdobeVFPrototype.ttf')