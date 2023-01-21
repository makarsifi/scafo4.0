
import grid
import checkerboard
import randsquares
import cv2
import os

class PatternGenerator:
    def __init__(self):
        # define dimension of the main display
        self.width_main  = 1920
        self.height_main = 1080

        # define dimension of the second display
        self.width_projector  = 1024
        self.height_projector = 768

        self.bg_color = [255,255,255]
        self.pattern_color = [255,0,0]
        
        os.environ['DISPLAY'] = ":0"

    def display_image(self, image):
        window_name = 'projector'
        cv2.namedWindow(window_name, cv2.WND_PROP_FULLSCREEN) 
        cv2.moveWindow(window_name, 0, -self.height_main)
        cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        cv2.startWindowThread()
        cv2.imshow(window_name, image_bgr)
        #cv2.waitKey()
        #cv2.destroyAllWindows()
    
    def close_display(self):
        cv2.destroyAllWindows()

    def generate_grid(self, vs, hs, n):
        image = grid.generate_grid(self.height_projector, self.width_projector, self.bg_color, self.pattern_color, vs, hs, n)
        self.display_image(image)

    def generate_checkerboard(self, size, seed):
        image = checkerboard.generate_checkerboard(self.height_projector, self.width_projector, self.bg_color, self.pattern_color, size, seed)
        self.display_image(image)

    def generate_randsquares(self, sz):
        image = randsquares.generate_randsquares(self.height_projector, self.width_projector, self.bg_color, self.pattern_color, sz)
        self.display_image(image)

def input_digit_or_default(default):
    str = input()
    if not str or not str.isdigit():
        return default
    else:
        return int(str)

if __name__ == "__main__":
    generator = PatternGenerator()

    while True:
        print('Which pattern to generate:')
        print('1 Grid')
        print('2 Checkerboard')
        print('3 Random squares')

        pattern = input()
        if pattern == '1':
            # print('Width (default is 1800 pixels): ')
            # width = input_digit_or_default(1800)
            # print('Height (default is 1200 pixels): ')
            # height = input_digit_or_default(1200)
            print('Vertical bars thickness vs (default is 50 pixels): ')
            vs = input_digit_or_default(50)
            print('Horizontal bars thickness hs (default is 50 pixels): ')
            hs = input_digit_or_default(50)
            print('Percentage of salt and pepper noise (default is 0): ')
            n = input_digit_or_default(0)
            generator.generate_grid(vs=vs, hs=hs, n=n)
            
        elif pattern == '2':
            print('Square size (default is 50 pixels): ')
            size = input_digit_or_default(50)
            print('Random seed (default is 10): ')
            seed = input_digit_or_default(10)
            generator.generate_checkerboard(size=size, seed=seed)

        elif pattern == '3':
            print('Square size (default is 50 pixels): ')
            sz = input_digit_or_default(50)
            generator.generate_randsquares(sz=sz)