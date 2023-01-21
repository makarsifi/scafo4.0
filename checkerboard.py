import numpy as np
from PIL import Image 
import argparse
import random

def generate_checkerboard(height, width, bg_color, pattern_color, size, seed):
#(height, width, bg_color, pattern_color, size, seed):
    bg_color=np.array(bg_color)
    pattern_color=np.array(pattern_color)
    random.seed(seed)

    # generate blank image
    img = np.zeros((height, width,3), np.uint8)
    img[::] = bg_color

    j=0
    while j < height:
        i=0
        while i < width:
            rand=random.random()
            if rand > 0.5: img[j:j+size, i:i+size] = pattern_color
            i+=size
        j+=size

    return img

if __name__ == "__main__":
    # parse the command line
    parser = argparse.ArgumentParser()
    parser.add_argument("--width", type=int, default=1800, help="width of the projected checkerboard (default is 1800 pixels)")
    parser.add_argument("--height", type=int, default=1200, help="height of the projected checkerboard (default is 1200 pixels)")
    parser.add_argument("--bg_color", type=int, nargs='+', default=[255,255,255], help="background color (default is white)")
    parser.add_argument("--pattern_color", type=int, nargs='+', default=[255,0,0], help="pattern color (default is red)")
    parser.add_argument("--size", type=int, default=50, help="square size (default is 50 pixels)")
    parser.add_argument("--seed", type=int, default=10, help="random seed (default is 10)")
    opt = parser.parse_args()

    array = generate_checkerboard(opt.height, opt.width, opt.bg_color, opt.pattern_color, opt.size, opt.seed)
    im = Image.fromarray(array)
    im.save(f'checkerboard_{opt.width}x{opt.height}_{opt.size}_{opt.seed}.png')