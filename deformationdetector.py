import cv2
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import imutils
import math 

class DeformationDetector:
    def __init__(self):
        # roi
        self.frame_pos_px = None
        self.frame_dist_px = None
        self.frame_dist_cm = None

        # shifting tags
        self.shifting_tags_pos_px = None

        # 1 cm to px ratios
        self.horizontal_ratio = 0 
        self.vertical_ratio = 0 

        # calibration bumps dimensions
        self.cal_bumps_height = None
        self.cal_bumps_diameter = None
        self.cal_bumps_volume = None
        self.cal_bumps_similarity = None

        # minimum bump area to be detected
        self.min_det_area_cm = 2 
        # minimum shifting to be detected
        self.min_shift_cm = 1
        # mimimum green ratio to consider crop as a shifting tag
        self.max_green_ratio = 0.1

        self.noise_T = 0
        self.blur_T = 0

        # bumps calibration function (regression)
        self.reg_m = 0
        self.reg_b = 0

    def boxes_overlap(self, box1, box2): #https://www.geeksforgeeks.org/find-two-rectangles-overlap/
        l1 = (box1[0], box1[1]) #x, y
        r1 = (box1[0]+box1[2], box1[1]+box1[3])

        l2 = (box2[0], box2[1])
        r2 = (box2[0]+box2[2], box1[1]+box2[3])

        # To check if either rectangle is actually a line
        # For example  :  l1 ={-1,0}  r1={1,1}  l2={0,-1}  r2={0,1}
        
        if (l1[0] == r1[0] or l1[1] == r1[1] or l2[0] == r2[0] or l2[1] == r2[1]):
            # the line cannot have positive overlap
            return False
        
        # If one rectangle is on left side of other
        if(l1[0] >= r2[0] or l2[0] >= r1[0]):
            return False
    
        # If one rectangle is above other
        if(r1[1] >= l2[1] or r2[1] >= l1[1]):
            return False
    
        return True

    def color_ranges(self, color):
        if color == 'blue':
            lower = np.array([100, 200, 0])
            upper = np.array([125, 255, 255])
        elif color == 'green':
            lower = np.array([60, 150, 0])
            upper = np.array([89, 255, 255])
        elif color == 'red':
            lower = np.array([160, 0, 0])
            upper = np.array([179, 255, 255])
        return lower, upper

    def detect_led_tags(self, image, led_color, led_count):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) # convert to grayscale 
        gray = cv2.merge([gray, gray, gray]) # obtain 3 channel grayscale image
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV) # convert to hsv

        lower, upper = self.color_ranges(led_color)

        mask = cv2.inRange(hsv, lower, upper)
        masked_output = cv2.bitwise_and(image, image, mask=mask) # apply hsv filtering to the image
        masked_output_gray = cv2.cvtColor(masked_output, cv2.COLOR_BGR2GRAY)

        thresh, binary = cv2.threshold(masked_output_gray, 0, 255, cv2.THRESH_BINARY)

        # find contours and draw a bounding box around all of them on the binary image
        contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        with_contours = binary.copy()

        detected_boxes=[]
        # draw only the first x biggest contours
        led_positions = []
        for c in contours:
            box = cv2.boundingRect(c)
            x, y, w, h = box

            # Make sure contour area is large enough to draw a rect around it
            if (cv2.contourArea(c)) == 0:
                continue

            # Make sure that the next box does not overlap with any added box
            for db in detected_boxes:
                overlaps = self.boxes_overlap(box, db)
                if overlaps:
                    continue

            cv2.rectangle(with_contours,(x,y), (x+w,y+h), (255,255,255), 5) #the color of the box will be white always since the image is grayscale
            detected_boxes.append(box)
            led_positions.append([x+w//2, y+h//2])

            if len(led_positions) == led_count:
                break
        
        return led_positions, masked_output, with_contours

    def clockwise_frame_positions(self, corners):
        # order led positions ABCD clockwise
        corners = np.array(corners)
        # the top-left point will have the smallest sum, whereas the bottom-right point will have the largest sum
        led_sum = corners.sum(axis = 1)
        a = corners[np.argmin(led_sum)]
        c = corners[np.argmax(led_sum)]
        # the top-right point will have the smallest difference, whereas the bottom-left will have the largest difference
        led_diff = np.diff(corners, axis = 1)
        b = corners[np.argmin(led_diff)]
        d = corners[np.argmax(led_diff)]
        return (a, b, c, d)

    def get_frame_tags(self, image):
        led_positions, masked_output, with_contours = self.detect_led_tags(image, 'blue', 4)
        led_positions = self.clockwise_frame_positions(led_positions)

        a = led_positions[0]
        b = led_positions[1]
        c = led_positions[2]
        d = led_positions[3]

        print('\nFrame Position:')
        print(f'A: {a}, B: {b}, C: {c}, D: {d}')

        # compute distances
        ab = round(np.linalg.norm(a-b))
        bc = round(np.linalg.norm(b-c))
        cd = round(np.linalg.norm(c-d))
        da = round(np.linalg.norm(d-a))
        print('\nFrame Distance:')
        print(f'AB: {ab}, BC: {bc}, CD: {cd}, DA: {da}')

        self.frame_pos_px = (a, b, c, d)
        self.frame_dist_px = (ab, bc, cd, da)
        self.update_ratios()

        return masked_output, with_contours

    def update_ratios(self):
        # self.px_cm_ratio = np.average(np.array(self.frame_dist_px)/np.array(self.frame_dist_cm))
        # print(f'\npx_cm_ratio is {self.px_cm_ratio}')
        # self.min_det_area_px = self.min_det_area_cm * self.px_cm_ratio
        #print(f'\nmin_det_area_px is  {self.min_det_area_px}')

        self.horizontal_ratio = (self.frame_dist_px[0] + self.frame_dist_px[2]) / (self.frame_dist_cm[0] + self.frame_dist_cm[2])
        self.vertical_ratio = (self.frame_dist_px[1] + self.frame_dist_px[3]) / (self.frame_dist_cm[1] + self.frame_dist_cm[3])
        print(f'\n horizontal_ratio is {self.horizontal_ratio}')
        print(f'\n vertical_ratio is {self.vertical_ratio}')

    def get_shifting_tags(self, image, store_positions=False):
        """ Use raw images not the ones resulting from perspective transform """
        led_positions, masked_output, with_contours = self.detect_led_tags(image, 'green', 4)
        led_positions = self.clockwise_frame_positions(led_positions)

        if store_positions:
            self.shifting_tags_pos_px = led_positions
            print('\nInitial shifting tags positions:')
            print(f'A: {led_positions[0]}, B: {led_positions[1]}, C: {led_positions[2]}, D: {led_positions[3]}')

        return led_positions, masked_output, with_contours

    def perspective_transform(self, image):
        a, b, c, d = self.frame_pos_px
        # compute the width of the new image, which will be the maximum distance between 
        widthA = np.sqrt(((c[0] - d[0]) ** 2) + ((c[1] - d[1]) ** 2)) # bottom-right and bottom-left x-coordiates 
        widthB = np.sqrt(((b[0] - a[0]) ** 2) + ((b[1] - a[1]) ** 2)) # or top-right and top-left x-coordinates
        maxWidth = max(int(widthA), int(widthB))
        # compute the height of the new image, which will be the maximum distance between
        heightA = np.sqrt(((b[0] - c[0]) ** 2) + ((b[1] - c[1]) ** 2)) # top-right and bottom-right y-coordinates
        heightB = np.sqrt(((a[0] - d[0]) ** 2) + ((a[1] - d[1]) ** 2)) # top-left and bottom-left y-coordinates
        maxHeight = max(int(heightA), int(heightB))

        dst_coordinates = np.array([[0, 0], [maxWidth - 1, 0], [maxWidth - 1, maxHeight - 1], [0, maxHeight - 1]], dtype = "float32")
        src_coordinates = np.float32([a, b, c, d]) 
        # compute the perspective transform matrix and then apply it
        perspective_matrix = cv2.getPerspectiveTransform(src_coordinates, dst_coordinates)
        warped = cv2.warpPerspective(image, perspective_matrix, (maxWidth, maxHeight))
        
        return warped

    def combine_images(self, img1, img2):
        combined_image = img1.copy()
        i = 1
        alpha = 1.0/(i + 1)
        beta = 1.0 - alpha
        combined_image = cv2.addWeighted(img2, alpha, combined_image, beta, 0.0)
        return combined_image

    def detect_shifting(self, led_positions, base_img, shifted_img):
        n = len(led_positions)
        shifting_array = np.zeros(n)
        combined_img = self.combine_images(base_img, shifted_img)
        for i in range(n):
            before = self.shifting_tags_pos_px[i]
            after = led_positions[i]
            
            # convert to cm
            before_cm = np.array(before).copy()
            before_cm[0] = before[0] / self.horizontal_ratio
            before_cm[1] = before[1] / self.vertical_ratio
            after_cm = np.array(after).copy()
            after_cm[0] = after[0] / self.horizontal_ratio
            after_cm[1] = after[1] / self.vertical_ratio
            
            # L2 norm
            shifting_array[i] = round(np.linalg.norm(before_cm-after_cm), 2)
            if shifting_array[i] > self.min_shift_cm: # draw arrow if larger than 1 cm
                print(f'Detected shifting of led {str(i+1)} by {shifting_array[i]} cm')
                combined_img = cv2.arrowedLine(combined_img, before, after, (0, 0, 255), 9)

        return combined_img, shifting_array

    def threshold_evaluation(self, iref_img, i_img):
        # convert iref to grayscale
        iref_gray_img = cv2.cvtColor(iref_img, cv2.COLOR_BGR2GRAY) 

        # convert I to grayscale
        i_gray_img = cv2.cvtColor(i_img, cv2.COLOR_BGR2GRAY)

        # interpolate Iref and I to [-127,127]
        iref_gray_img_interp = np.interp(iref_gray_img, [0,255], [-127,127])
        i_gray_img_interp = np.interp(i_gray_img, [0,255], [-127,127])

        # compute the difference between I and Iref
        diff_img = np.absolute(i_gray_img_interp - iref_gray_img_interp) 

        # compute the histogram of Idiff
        vals = diff_img.flatten()
        fig = plt.figure()
        b, bins, patches = plt.hist(vals, 255)

        # get the mean and max values and plot them. What is T in that case??
        mean = vals.mean()
        max = vals.max()
        plt.axvline(mean, color='k', linestyle='dashed', linewidth=1)
        plt.axvline(max, color='r', linestyle='dashed', linewidth=1)

        # return the plot
        fig.canvas.draw()
        image_from_plot = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        image_from_plot = image_from_plot.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        return image_from_plot

    def get_Idiff_stand(self, iref_img, bump_img):
        """ Another method is to use opencv substraction method """

        # convert to grayscale and interpolate to 
        iref_gray_img = cv2.cvtColor(iref_img, cv2.COLOR_BGR2GRAY)  
        bump_gray_img = cv2.cvtColor(bump_img, cv2.COLOR_BGR2GRAY)

        # interpolate to [-127,127]
        iref_gray_img_interp = np.interp(iref_gray_img, [0,255], [-127,127])
        bump_gray_img_interp = np.interp(bump_gray_img, [0,255], [-127,127])

        # difference between Iref and image with bump
        diff_img = np.absolute(bump_gray_img_interp - iref_gray_img_interp)

        # substract the noise threshold T detected earlier
        diff_img_denoise = diff_img - self.noise_T
        diff_img_denoise[diff_img_denoise < 0]  = 0
        diff_img_denoise = diff_img_denoise.astype(np.uint8)

        return diff_img_denoise

    def construct_morph_shapes(self, bump_diff_img):
        img = bump_diff_img.copy()

        # here we can try to find an automatic method to detect the blur threshold        
        # we can compute by counting the number of bars / width of the cropped image
        # this could be a machine learning problem
        bar_width = 3 

        # blur the image 
        blur_ksize = (self.blur_T, 5) 
        img = cv2.blur(img, blur_ksize)

        # perform a series of erosions and dilations
        img = cv2.dilate(img, None, iterations = 4)
        img = cv2.erode(img, None, iterations = 4)

        # binarize the image
        img = cv2.threshold(img, self.noise_T, 255, cv2.THRESH_BINARY)[1]

        # construct a closing kernel and apply it to the thresholded image
        closing_ksize = (50, 50) # sliding winow of pixels to be connected together
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, closing_ksize)
        img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)

        return img

    def retrieve_color_ratio(self, img, color):
        # apply hsv mask
        lower, upper = self.color_ranges(color)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV) 
        mask = cv2.inRange(hsv, lower, upper)
        masked_output = cv2.bitwise_and(img, img, mask=mask)

        # convert to grayscale
        masked_output_gray = cv2.cvtColor(masked_output, cv2.COLOR_BGR2GRAY)

        # get all non black Pixels from grayscale
        non_black = cv2.countNonZero(masked_output_gray)

        # get ratio of non black pixels
        total_pixels = masked_output_gray.shape[0] * masked_output_gray.shape[1]
        ratio = non_black / total_pixels
        return ratio
        
    def detect_bump_location(self, morph_img, iref_img, bump_img, ltr=True):
        # find the contours in the binary image, then sort the contours
        img = morph_img.copy().astype(np.uint8)
        contours = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = imutils.grab_contours(contours)
        contours = sorted(contours, key = cv2.contourArea, reverse = True)

        large_boxes = []
        combined_img = self.combine_images(iref_img, bump_img) 
        for c in contours:
            # use straight rect (cv.boundingRect) instead of cv.minAreaRect which is rotated 
            x, y, w, h = cv2.boundingRect(c)
            
            # filter out small boxes (lower than min_det_area_cm)
            if w < (self.min_det_area_cm * self.horizontal_ratio) or h < (self.min_det_area_cm * self.vertical_ratio):
                continue

            # filter out greenish boxes because those will be related to shifting leds
            crop = combined_img[y:y+h, x:x+w]
            ratio = self.retrieve_color_ratio(crop, 'green')
            if ratio > self.max_green_ratio:
                print('greenish bump: to be skipped')
                continue

            large_boxes.append((x, y, w, h))

        # order rtl or ltr (only compare on the x axis, all the bumps should be aligned)
        large_boxes = sorted(large_boxes, key=lambda t: (t[0]+t[2])/2, reverse=not ltr)

        return large_boxes
        
    def evaluate_similarity(self, box, bump_img, iref_img, binary_evaluation=False):
        # crop the bump
        bump_crop = bump_img[box[1]:box[1]+box[3], box[0]:box[0]+box[2]]
        # crop the same location from Iref
        original_crop = iref_img[box[1]:box[1]+box[3], box[0]:box[0]+box[2]]

        bump_crop = cv2.cvtColor(bump_crop, cv2.COLOR_BGR2GRAY)
        original_crop = cv2.cvtColor(original_crop, cv2.COLOR_BGR2GRAY)

        if binary_evaluation:
            bump_crop = cv2.threshold(bump_crop, 50, 255, cv2.THRESH_BINARY_INV)[1]
            original_crop = cv2.threshold(original_crop, 50, 255, cv2.THRESH_BINARY_INV)[1]

        # if show_crops:
        #     display_images_compare(bump_crop, original_crop, f'bump_{box[2]}x{box[3]}')
        #     display_images_compare(bump_crop, original_crop)

        # 3.
        #res = cv2.norm(bump_crop, original_crop, cv2.NORM_L2)

        # 2.
        # cv2.TM_CCOEFF cv2.TM_CCOEFF_NORMED cv2.TM_CCORR cv2.TM_CCORR_NORMED cv2.TM_SQDIFF cv2.TM_SQDIFF_NORMED'
        # res = cv2.matchTemplate(bump_crop, original_crop, cv2.TM_CCORR_NORMED) 
        # print(res)
        # res = res[0]

        # 1. 
        # normalize both crops between 0 and 1
        bump_crop = bump_crop / np.linalg.norm(bump_crop) 
        original_crop = original_crop / np.linalg.norm(original_crop) 
        
        intersection = np.minimum(bump_crop, original_crop)
        #intersection = np.multiply(bump_crop, original_crop)
        union = np.maximum(bump_crop, original_crop)
        res = np.sum(intersection) / np.sum(union)

        res = res / (box[2]*box[3])

        return res

    def evaluate_bumps(self, large_boxes, iref_img, bump_img, reverse_evaluation=False, binary_evaluation=False):
        correlations = []
        with_contours = bump_img.copy()
        i = 1
        for box in large_boxes:
            x, y, w, h = box
            corr = self.evaluate_similarity(box, bump_img, iref_img, binary_evaluation)
            correlations.append(corr)
            # draw boxes on image
            cv2.rectangle(with_contours, (x,y), (x+w,y+h), (0,255,0), 5) 
            cv2.putText(with_contours, str(i), (x+20, y+(h//2)), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3, cv2.LINE_AA)
            if reverse_evaluation:
                volume = (corr - self.reg_b) / self.reg_m
                volume = round(volume, 2)
                str_volume = f'{volume} cm^3'
                cv2.putText(with_contours, str_volume, (x, y+h+60), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 0), 3, cv2.LINE_AA)
            else:
                str_corr = format(corr,'.1E') #'{:.7f}'.format(corr)
                cv2.putText(with_contours, str_corr, (x, y+h+60), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 0), 3, cv2.LINE_AA)
            i+=1

        print('\ncorrelation values:')
        for i in range(len(large_boxes)):
            print(f'\t{large_boxes[i][2]}x{large_boxes[i][3]}px --> {correlations[i]}')

        return correlations, with_contours

    def set_avg_correlations(self, corr_ltr, corr_rtl):
        both_directions = np.array([corr_ltr, corr_rtl])
        mean_values = np.mean(both_directions, axis=0)
        self.cal_bumps_similarity = mean_values

    def set_calibration_bumps(self, dimensions):
        h_array = []
        d_array = []
        v_array = []
        for dim in dimensions:
            h_array.append(dim[0])
            d_array.append(dim[1])
            volume = math.pi * (dim[1]/2)**2 * dim[0]
            v_array.append(volume)
        v_array = np.sort(v_array)
        
        self.cal_bumps_height = np.array(h_array)
        self.cal_bumps_diameter = np.array(d_array)
        self.cal_bumps_volume = np.array(v_array)

    def plot_regression(self, logscale=False):
        fig = plt.figure()
        x = np.array(self.cal_bumps_volume) * 0.001 # from cubic mm to cubic cm
        y = np.log(self.cal_bumps_similarity) if logscale else self.cal_bumps_similarity
        plt.scatter(x, y)

        m, b = np.polyfit(x, y, 1)
        plt.plot(x, m*x + b)

        self.reg_m = m
        self.reg_b = b

        plt.grid()
        plt.xlabel("bump volume (cm^3)")
        # plt.xticks(x)
        plt.ylabel("similarity")

        # return the plot
        fig.canvas.draw()
        image_from_plot = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        image_from_plot = image_from_plot.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        return image_from_plot