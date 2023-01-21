import numpy as np
from PIL import Image 
import argparse
import random


def generate_grid(height, width, bg_color, pattern_color, vs, hs, n):
    bg_color=np.array(bg_color)
    pattern_color=np.array(pattern_color)

    # generate blank image
    img = np.zeros((height, width,3), np.uint8)
    img[::] = bg_color

    # add vertical bars
    if vs > 0:
        i=vs
        while i < width:
            img[:,i:i+vs] = pattern_color
            i=i+2*vs

    # add horizontal bars
    if hs > 0:
        j=hs
        while j < height:
            img[j:j+hs,:] = pattern_color
            j=j+2*hs

    # add noise
    if n > 0:
        num_pixels = height*width*(n/100)
        for i in range(int(num_pixels)):
            y_coord=random.randint(0, height - 1)
            x_coord=random.randint(0, width - 1)
            img[y_coord, x_coord] = bg_color if (img[y_coord, x_coord] == pattern_color).all() else pattern_color

    return img

if __name__ == "__main__":
    # parse the command line
    parser = argparse.ArgumentParser()
    parser.add_argument("--width", type=int, default=1800, help="width of the projected grid (default is 1800 pixels)")
    parser.add_argument("--height", type=int, default=1200, help="height of the projected grid (default is 1200 pixels)")
    parser.add_argument("--bg_color", type=int, nargs='+', default=[255,255,255], help="background color (default is white)")
    parser.add_argument("--pattern_color", type=int, nargs='+', default=[255,0,0], help="pattern color (default is red)") # e.g --pattern_color 0 255 0
    parser.add_argument("--vs", type=int, default=50, help="vertical bars thickness (default is 50 pixels)")
    parser.add_argument("--hs", type=int, default=50, help="horizontal bars thickness (default is 50 pixels)")
    parser.add_argument("--n", type=int, default=0, help="percentage of salt and pepper noise")
    opt = parser.parse_args()

    array = generate_grid(opt.height, opt.width, opt.bg_color, opt.pattern_color, opt.vs, opt.hs, opt.n)
    im = Image.fromarray(array)
    im.save(f'grid_{opt.width}x{opt.height}_{opt.vs}_{opt.hs}_{opt.n}.png')