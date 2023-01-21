import numpy as np
from PIL import Image 
import argparse
import random

def generate_randsquares(height, width, bg_color, pattern_color, sz):
    bg_color=np.array(bg_color)
    pattern_color=np.array(pattern_color)

    # generate blank image
    img = np.zeros((height, width,3), np.uint8)
    img[::] = bg_color

    num_pixels = height*width/sz
    for i in range(int(num_pixels)):
        y_coord=random.randint(0, height - 1 - sz)
        x_coord=random.randint(0, width - 1 - sz)
        if (img[y_coord:y_coord+sz, x_coord:x_coord+sz] == bg_color).all():
            img[y_coord:y_coord+sz, x_coord:x_coord+sz] = pattern_color

    return img

if __name__ == "__main__":
    # parse the command line
    parser = argparse.ArgumentParser()
    parser.add_argument("--width", type=int, default=1800, help="width of the projected checkerboard (default is 1800 pixels)")
    parser.add_argument("--height", type=int, default=1200, help="height of the projected checkerboard (default is 1200 pixels)")
    parser.add_argument("--bg_color", type=int, nargs='+', default=[255,255,255], help="background color (default is white)")
    parser.add_argument("--pattern_color", type=int, nargs='+', default=[255,0,0], help="pattern color (default is red)")
    parser.add_argument("--sz", type=int, default=50, help="square size (default is 50 pixels)")
    opt = parser.parse_args()

    array = generate_randsquares(opt.height, opt.width, opt.bg_color, opt.pattern_color, opt.sz)
    im = Image.fromarray(array)
    im.save(f'randsquares_{opt.width}x{opt.height}_{opt.sz}.png')