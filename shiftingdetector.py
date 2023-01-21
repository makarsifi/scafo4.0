from lib2to3.pygram import python_grammar_no_print_statement
import cv2
import numpy as np

class ShiftingDetector:
    def __init__(self):
        self.initial_shift_pos_px = None

        self.cc_median_w = None
        self.cc_median_w = None
        self.cc_count = None
        self.cc_labels = None
        self.cc_stats = None
        self.cc_centroids = None

    def hsv_filtering(self, img, min_s, min_v):
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV) # convert to hsv

        # lower mask for red
        # (fix Hue and manipulate s and v)
        lower_red = np.array([0,min_s,min_v])
        upper_red = np.array([15,255,255])
        mask0 = cv2.inRange(hsv, lower_red, upper_red)

        # upper mask for violet-red
        lower_red = np.array([140,min_s,min_v])
        upper_red = np.array([180,255,255])
        mask1 = cv2.inRange(hsv, lower_red, upper_red)
        
        # join masks
        mask = mask0+mask1

        # apply hsv filtering to the image
        masked_output = cv2.bitwise_and(img, img, mask=mask) 
        return masked_output

    def select_lines(self, img, open_kernel):
        morph_open_kernel = open_kernel # start from 10 and increase until we dont see any horizontal noise in v_lines.png, and any vertical noise in h_lines.png

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)

        # select only horizontal lines
        kernel_h = np.ones((1, morph_open_kernel), np.uint8)
        h_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_h)
        
         # select only vertical lines
        kernel_v = np.ones((morph_open_kernel, 1), np.uint8)
        v_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_v)

        return h_lines, v_lines
        
    def hough_lines(self, img, hough_threshold, direction_horizontal=True):
        height, width = img.shape
        
        rho = 1 # in pixels
        theta = np.pi/180 # in Radians = Degrees × π/180
        
        # The minimum number of intersections to "*detect*" a line
        # starting from 1 will connect all points together, increase with 1 step until seeing unique lines only
        threshold = hough_threshold   
        

        if direction_horizontal:
            minLineLength = width/2 # The minimum number of points that can form a line. Lines with less than this number of points are disregarded
            # we consider that lines that are longer than half of the image size are real grid lines, others are noise
            maxLineGap = width/2 # The maximum gap between two points to be considered in the same line.
            # we consider 2 lines as one if the gap between them does not exceed [half] the image size
        else:
            minLineLength = height/2
            maxLineGap = height/2 

        linesP = cv2.HoughLinesP(img, rho, theta, threshold, None, minLineLength, maxLineGap)
        
        # Draw the lines
        connected = np.zeros_like(img)
        if linesP is not None:
            for line in linesP:
                x1, y1, x2, y2 = line[0]
                if direction_horizontal:
                    cv2.line(connected, (0, y1), (width, y2), 255, 3, cv2.LINE_AA)
                else:
                    cv2.line(connected, (x1, 0), (x2, height), 255, 3, cv2.LINE_AA)

        return connected

    def connect_lines(self, h_lines, v_lines, hough_threshold):
        # connect horizontal lines
        h_lines = self.hough_lines(h_lines, hough_threshold)

        # connect vertical lines
        v_lines = self.hough_lines(v_lines, hough_threshold, direction_horizontal=False)

        return h_lines, v_lines

    def dilate_lines(self, img, dilation_kernel):
        dilation_iterations = 2 # how much times we want to dilate 
                            # (this could be fixed to 2 since increasing dilation_kernel achieves same 
                            # result which is increasing the thickness of the squares)  

        kernel = np.ones((dilation_kernel, dilation_kernel), np.uint8)
        img = cv2.dilate(img, kernel, iterations=dilation_iterations)
        return img

    def skeletonize(self, img):
        """ OpenCV function to return a skeletonized version of img"""
        # hat tip to http://felix.abecassis.me/2011/09/opencv-morphological-skeleton/
        # https://homepages.inf.ed.ac.uk/rbf/HIPR2/thin.htm
        skel = img.copy()

        skel[:,:] = 0
        kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3,3))

        while True:
            eroded = cv2.morphologyEx(img, cv2.MORPH_ERODE, kernel)
            temp = cv2.morphologyEx(eroded, cv2.MORPH_DILATE, kernel)
            temp  = cv2.subtract(img, temp)
            skel = cv2.bitwise_or(skel, temp)
            img[:,:] = eroded[:,:]
            if cv2.countNonZero(img) == 0:
                break

        return skel

    def merge_lines(self, h_lines, v_lines, dilation_kernel):
        # dilate to remove gaps from close lines
        h_lines = self.dilate_lines(h_lines, dilation_kernel)
        v_lines = self.dilate_lines(v_lines, dilation_kernel)

        # skeletonize to reduce lines to 1 pixel thickness
        h_lines = self.skeletonize(h_lines)
        v_lines = self.skeletonize(v_lines)

        # merge h and v lines
        merged = h_lines|v_lines
        
        # dilate the whole grid
        merged = self.dilate_lines(merged, dilation_kernel)
        
        return merged

    def grid_reconstruction(self, img, cell_size_px):
        # binarize again after transformations
        _, merged = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

        # inverse
        inverse=~merged

        # detect connected components
        rec_img = np.zeros_like(merged)
        ret, labels, stats, centroids = cv2.connectedComponentsWithStats(inverse, connectivity=8, ltype=cv2.CV_32S)
        
        # dimensions histograms
        array_w = stats[:,2] # third column is width
        array_h = stats[:,3] # fourth column is height
        # plt.hist(array_w, 30, rwidth=0.5, range=[0, 100], align='mid')
        # plt.ylabel('frequency')
        # plt.xlabel('width')
        # plt.show()
        median_w = np.median(array_w)
        print(f'median_w {median_w}')

        # plt.hist(array_h, 30,  rwidth=0.5, range=[0, 100], align='mid')  
        # plt.ylabel('frequency')
        # plt.xlabel('height')
        # plt.show()
        median_h = np.median(array_h)
        print(f'median_h {median_h}')

        min_det_size_px = cell_size_px + 50 # only draw rectangles that are smalller than this dimension
                                            # add a margin of error of 50px
        for x,y,w,h,area in stats:
            if w < min_det_size_px and h < min_det_size_px:
                #cv2.rectangle(image, (x,y), (x+w,y+h), (0, 0, 255), 2) # just to show the detection result in red
                cv2.rectangle(rec_img, (x,y), (x+w,y+h), 255, -1)
                
        print('number of squares ', ret)
        
        # connected components details
        self.cc_median_w = median_w
        self.cc_median_w = median_w
        self.cc_count = ret
        self.cc_labels = labels
        self.cc_stats = stats
        self.cc_centroids = centroids
        
        return rec_img

    def draw_dots(self, point_b, reconstructed):
        """Draw green dots on the reconstructed grid image"""
        dots_img = cv2.merge([reconstructed, reconstructed, reconstructed])
        point_a = self.initial_shift_pos_px
        cv2.circle(dots_img, point_a, radius=7, color=(0, 255, 0), thickness=-1)
        cv2.circle(dots_img, point_b, radius=7, color=(0, 255, 0), thickness=-1)
        return dots_img

    def point_in_stat(self, p):
        """Check in which stat/gridcell is a point"""
        found_stat = None
        for s in self.cc_stats: # for each stat check if the green point is within its surface
            x, y, w, h, a = s
            rect_x_min = x
            rect_x_max = x+w
            rect_y_min = y
            rect_y_max = y+h
            if (rect_x_min <= p[0] <= rect_x_max) and (rect_y_min <= p[1] <= rect_y_max):
                found_stat = s
                
        #print(f'found square {found_stat}')
        return found_stat

    def squares_horizontal_coverage(self, h_stats, first_x, last_x):
        # add the percentage column
        hundred_col = np.ones((len(h_stats), 1))
        h_stats = np.append(h_stats, hundred_col, axis=1) # contains now x, y, w, h, area, coverage
        
        # shifting horizontal direction
        direction = 'ltr' if first_x < last_x else 'rtl'

        # Left square side   = Xpoint-Xsquare / Width
        # Right square side  = Xsquare+Width-Xpoint / Width
        # Top square side    = Ypoint-Ysquare / Height
        # Bottom square side = Ysquare+Height-Ypoint / Height

        # First square horizontal coverage
        # if direction is ltr then coverage=right square side
        # rf direction is rtl then coverage=left square side

        x, y, w, h, a, c  = h_stats[0]
        left_side = (first_x-x) / w
        right_side = (x+w-first_x) / w 
        h_stats[0, 5] = right_side if direction == 'ltr' else left_side

        # Last square coverage is obtained by swapping the previous conditions
        
        x, y, w, h, a, c  = h_stats[len(h_stats)-1]
        left_side = (last_x-x) / w
        right_side = (x+w-last_x) / w 
        h_stats[len(h_stats)-1, 5] = left_side if direction == 'ltr' else right_side
    
        return h_stats

    def squares_vertical_coverage(self, v_stats, first_y, last_y):
        hundred_col = np.ones((len(v_stats), 1))
        v_stats = np.append(v_stats, hundred_col, axis=1) # contains now x, y, w, h, area, coverage
        
        direction = 'ttb' if first_y < last_y else 'btt'
        
        x, y, w, h, a, c  = v_stats[0]
        top_side = (first_y-y) / h
        bottom_side = (y+h-first_y) / h
        v_stats[0, 5] = bottom_side if direction == 'ttb' else top_side
        
        x, y, w, h, a, c  = v_stats[len(v_stats)-1]
        top_side = (last_y-y) / h
        bottom_side = (y+h-last_y) / h 
        v_stats[len(v_stats)-1, 5] = top_side if direction == 'ttb' else bottom_side
    
        return v_stats

    def crawl_lines(self, x2, y2, img):
        """Crawl line for cells discovery, if the LED shifted its position a visualization is returned"""
        crawl_img = cv2.merge([img, img, img])

        x1 = self.initial_shift_pos_px[0]
        y1 = self.initial_shift_pos_px[1]

        previous = 0
        h_stats = []
        for x in range(x1, x2+1): # using +1 here meaning that we can only go forward (bug)
            if img[y1, x] == 255 and previous == 0: # if current is white and previous was black it means we entered a new square
                pis = self.point_in_stat((x, y1))
                cv2.rectangle(crawl_img, (pis[0],pis[1]), (pis[0]+pis[2],pis[1]+pis[3]), (255, 255, 0), -1)
                h_stats.append(pis)
            previous = img[y1, x]

        previous = 0
        v_stats = []
        first_detected = True
        for y in range(y1, y2+1): 
            if img[y, x1] == 255 and previous == 0:
                pis = self.point_in_stat((x1, y))
                color =  (0, 255, 255)
                if first_detected:
                    color = (0, 255, 0)
                    first_detected = False
                cv2.rectangle(crawl_img, (pis[0],pis[1]), (pis[0]+pis[2],pis[1]+pis[3]), color, -1)
                v_stats.append(pis)
            previous = img[y, x1]

        if len(h_stats) > 0: # if the 2 points are in the same column
            h_stats = self.squares_horizontal_coverage(h_stats, x1, x2)
            print(f'\nh_stats:\n {h_stats}')

        if len(v_stats) > 0: # if the 2 points are in the same row
            v_stats = self.squares_vertical_coverage(v_stats, y1, y2)
            print(f'\nv_stats:\n {v_stats}')

        return h_stats, v_stats, crawl_img
