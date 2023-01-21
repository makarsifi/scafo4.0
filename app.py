from logging.config import valid_ident
from flask import Flask, render_template, Response, request, session
from flask_socketio import SocketIO
from scafocapture import ScafoCapture, CapAvgMethod
from patternsgenerator import PatternGenerator
from deformationdetector import DeformationDetector
from shiftingdetector import ShiftingDetector
from imagemanager import ImageManager
import numpy as np
import time 
from datetime import datetime
from threading import Thread

app = Flask(__name__)
app.secret_key = b'_5#y2L"F4Q8z\n\xec]/' # session secret (should be changed)
socketio = SocketIO(app)

CAP = ScafoCapture()
PAT = PatternGenerator()
DEF = DeformationDetector()
SHI = ShiftingDetector()
MAN = ImageManager()
MAN.output = 'static/output/'

#########################################################
# region Common
def init_session():
    if 'sessionid' not in session:
        session['sessionid'] = str(time.time()).replace('.', '_')
    MAN.filename_prefix = session['sessionid']

@app.route('/')
def index():
    init_session()
    print(session)
    print('prefix: ', MAN.filename_prefix)
    
    return render_template('dashboard.html')

@app.route('/video_feed')
def video_feed():
    return Response(CAP.start_stream(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/store_distances', methods=['POST'])
def store_distances():
    ab_str = request.form['AB']
    bc_str = request.form['BC']
    cd_str = request.form['CD']
    da_str = request.form['DA']

    ab = float(ab_str) 
    bc = float(bc_str)
    cd = float(cd_str)
    da = float(da_str)
    DEF.frame_dist_cm = (ab, bc, cd, da)

    return 'Distances stored'

@app.route('/detect_roi', methods=['POST'])
def detect_roi():
    frame_img = CAP.capture_n()
    MAN.store_image(frame_img, 'frame_img')

    frame_mask_img, frame_detect_img = DEF.get_frame_tags(frame_img)
    MAN.store_image(frame_mask_img, 'frame_mask_img')
    frame_detect_img_path = MAN.store_image(frame_detect_img, 'frame_detect_img')
    
    return frame_detect_img_path
# endregion
#########################################################

#########################################################
# region Bumps Detection

## Uncomment for testing 
# DEF.frame_pos_px = ((601, 920), (2285, 886), (2261, 2186), (586, 2187))
# DEF.frame_dist_px = (1684, 1300, 1675, 1267)
# DEF.horizontal_ratio = 24.69
# DEF.vertical_ratio = 24.7
# DEF.shifting_tags_pos_px = ((905, 1087), (1999, 1125), (1954, 1929), (892, 1936))
# DEF.noise_T = 4
# DEF.blur_T = 60
##

@app.route('/setup')
def setup():
    init_session()
    return render_template('setup.html')

@app.route('/project_pattern', methods=['POST'])
def project_pattern():
    PAT.close_display()
    vs_str = request.form['vs']
    print('bars width: ', vs_str)
    vs = int(vs_str) if vs_str.isdigit() else 1 
    if vs > 0:
        PAT.generate_grid(vs, 0, 0)
        return 'On'
    else:
        return 'Off'

@app.route('/capture_baseline', methods=['POST'])
def capture_baseline():
    base_img_raw  = CAP.capture_n()
    MAN.store_image(base_img_raw, 'base_img_raw')
    
    _, shifting_tags_mask_img, shifting_tags_detect_img = DEF.get_shifting_tags(base_img_raw, store_positions=True)
    MAN.store_image(shifting_tags_mask_img, 'shifting_tags_mask_img')
    MAN.store_image(shifting_tags_detect_img, 'shifting_tags_detect_img')

    base_img = DEF.perspective_transform(base_img_raw)
    base_img_path = MAN.store_image(base_img, 'base_img')

    return base_img_path

@app.route('/noise_evaluation', methods=['POST'])
def noise_evaluation():
    succ_img_raw  = CAP.capture_n()
    MAN.store_image(succ_img_raw, 'succ_img_raw')

    succ_img = DEF.perspective_transform(succ_img_raw)
    MAN.store_image(succ_img, 'succ_img')

    base_img = MAN.get_image('base_img')

    t_plot_img = DEF.threshold_evaluation(base_img, succ_img)
    t_plot_img_path = MAN.store_image(t_plot_img, 't_plot_img')
    
    return t_plot_img_path

@app.route('/store_T', methods=['POST'])
def store_T():
    T_str = request.form['T']
    DEF.noise_T = int(T_str) if T_str.isdigit() else 0
    return 'Noise threshold stored'

@app.route('/store_bumps_dimensions', methods=['POST'])
def store_bumps_dimensions():
    dimensions_str = request.form['dimensions']
    dimensions_arr = dimensions_str.split(';')
    dimensions = []
    for h_d in dimensions_arr:
        dimensions.append(h_d.split(','))
    dimensions = np.array(dimensions).astype(float)

    DEF.set_calibration_bumps(dimensions)
    return 'Bumps depth stored'

@app.route('/capture_bumps', methods=['POST'])
def capture_bumps():
    direction = request.form['direction']
    
    bumps_img_raw  = CAP.capture_n()
    MAN.store_image(bumps_img_raw, f'bumps_{direction}_img_raw')
    bumps_img = DEF.perspective_transform(bumps_img_raw)
    bumps_img_path = MAN.store_image(bumps_img, f'bumps_{direction}_img')

    base_img = MAN.get_image('base_img')
    diff_img = DEF.get_Idiff_stand(base_img, bumps_img)
    MAN.store_image(diff_img, f'diff_{direction}_img')

    return bumps_img_path
    
@app.route('/detect_bumps', methods=['POST'])
def detect_bumps():
    diff_ltr_img = MAN.get_image('diff_ltr_img')
    diff_rtl_img = MAN.get_image('diff_rtl_img')

    threshold = int(request.form['blur_T'])
    DEF.blur_T = threshold if threshold > 0 else 1

    morph_ltr_img = DEF.construct_morph_shapes(diff_ltr_img)
    morph_ltr_img_path = MAN.store_image(morph_ltr_img, 'morph_ltr_img')
   
    morph_rtl_img = DEF.construct_morph_shapes(diff_rtl_img)
    morph_rtl_img_path = MAN.store_image(morph_rtl_img, 'morph_rtl_img')

    return f'{morph_ltr_img_path},{morph_rtl_img_path}'

@app.route('/evaluate_bumps', methods=['POST'])
def evaluate_bumps():
    base_img = MAN.get_image('base_img')
    
    morph_ltr_img = MAN.get_image('morph_ltr_img', grayscale=True)
    bumps_ltr_img = MAN.get_image('bumps_ltr_img')
    bumps_locations_ltr = DEF.detect_bump_location(morph_ltr_img, base_img, bumps_ltr_img, ltr=True)
    bumps_correlations_ltr, contour_ltr_img = DEF.evaluate_bumps(bumps_locations_ltr, base_img, bumps_ltr_img)
    MAN.store_image(contour_ltr_img, 'contour_ltr_img')

    morph_rtl_img = MAN.get_image('morph_rtl_img', grayscale=True)
    bumps_rtl_img = MAN.get_image('bumps_rtl_img')
    bumps_locations_rtl = DEF.detect_bump_location(morph_rtl_img, base_img, bumps_rtl_img, ltr=False)
    bumps_correlations_rtl, contour_rtl_img = DEF.evaluate_bumps(bumps_locations_rtl, base_img, bumps_rtl_img)
    MAN.store_image(contour_rtl_img, 'contour_rtl_img')

    DEF.set_avg_correlations(bumps_correlations_ltr, bumps_correlations_rtl)
    reg_plot_img = DEF.plot_regression()
    reg_plot_img_path = MAN.store_image(reg_plot_img, 'reg_plot_img')

    return reg_plot_img_path

@app.route('/monitoring')
def monitoring():
    return render_template('monitoring.html')

@app.route('/start_monitoring', methods=['POST'])
def start_monitoring():
    #thread = Thread(target=test_thread)
    thread = Thread(target=monitoring_thread, args=(request,))
    thread.start()
    return 'Monitoring...'

def monitoring_thread(request):
    CAP.monitoring = True
    print ("Starting monitoring thread")

    while CAP.monitoring:
        socketio.emit('monitor_delay_elapsed')

        capture_time = datetime.now().strftime("%H:%M:%S")
        print(f'Capturing @ {capture_time}')
        mon_img_raw = CAP.capture_n()
        mon_img = DEF.perspective_transform(mon_img_raw)
        MAN.store_image(mon_img, 'mon_img')

        base_img = MAN.get_image('base_img')
        diff_img = DEF.get_Idiff_stand(base_img, mon_img)
        MAN.store_image(diff_img, 'mon_diff_img')
        
        morph_img = DEF.construct_morph_shapes(diff_img)
        MAN.store_image(morph_img, 'mon_morph_img')

        bumps_locations = DEF.detect_bump_location(morph_img, base_img, mon_img, ltr=True)

        result = {
            "capture_time" : capture_time,
            "bumps_detected": 0,
            "shifting_detected" : False
        }

        # build bumps results
        if len(bumps_locations) > 0:
            bumps_correlations, contour_img = DEF.evaluate_bumps(bumps_locations, base_img, mon_img, reverse_evaluation=True)
            result["bumps_detected"] = len(bumps_locations)
            result["contour_img_path"] = MAN.store_image(contour_img, 'mon_detected_img')

        # build shifting results
        base_img_raw = MAN.get_image('base_img_raw')
        green_led_positions, _, _ = DEF.get_shifting_tags(mon_img_raw)
        combined_image, shifting_array = DEF.detect_shifting(green_led_positions, base_img_raw, mon_img_raw)
        if np.any(shifting_array > 1):
            result["shifting_detected"] = True
            shifting_results = []
            for i in range(len(shifting_array)):
                if shifting_array[i]>1:
                    shifting_results.append(f'Detected shifting of led {str(i+1)} by {shifting_array[i]} cm')
            result["shifting_results"] = shifting_results

            shifting_image_path = MAN.store_image(combined_image, 'mon_shifting_image')
            result["shifting_image_path"] = shifting_image_path

        socketio.emit('monitor_executed', result)
        time.sleep(CAP.monitoring_delay)

    print("Exiting monitoring thread")

@socketio.on('disconnect')
def socketio_disconnect():
    CAP.monitoring = False
    print('Socketio client disconnected')

# endregion
#########################################################

#########################################################
# region Shifting Detection

## Call this method on page load for testing
def load_from_session(sessionid):
    session['sessionid'] = sessionid
    DEF.frame_pos_px = ((1291, 1031), (2740, 929), (2806, 2037), (1349, 2086))
    DEF.frame_dist_px = (1453, 1110, 1458, 1057)
    DEF.horizontal_ratio = 26
    DEF.vertical_ratio = 29
    SHI.initial_shift_pos_px = (247, 605)

@app.route('/setup_shifting')
def setup_shifting():
    load_from_session('1659598981_7212632') # call to test
    init_session()
    return render_template('setup_shifting.html')

@app.route('/monitor_shifting')
def monitor_shifting():
    return render_template('monitor_shifting.html')

@app.route('/capture_grid', methods=['POST'])
def capture_grid():
    grid_img_raw  = CAP.capture_n(n=30, avg_method=CapAvgMethod.Max)
    MAN.store_image(grid_img_raw, 'grid_img_raw')
    
    grid_img = DEF.perspective_transform(grid_img_raw)
    grid_img_path = MAN.store_image(grid_img, 'grid_img')
    return grid_img_path

@app.route('/hsv_filtering', methods=['POST'])
def hsv_filtering():
    saturation = int(request.form['saturation'])
    value = int(request.form['value'])

    shift_base_img = MAN.get_image('grid_img')
    hsv_filtered_img = SHI.hsv_filtering(shift_base_img, saturation, value)
    hsv_filtered_img_path = MAN.store_image(hsv_filtered_img, 'hsv_filtered_img')
    return hsv_filtered_img_path

@app.route('/lines_selection', methods=['POST'])
def lines_selection():
    openkernel = int(request.form['openkernel'])

    hsv_filtered_img = MAN.get_image('hsv_filtered_img')
    h_lines, v_lines = SHI.select_lines(hsv_filtered_img, openkernel)
    h_lines_img_select_path = MAN.store_image(h_lines, 'h_lines_select_img')
    v_lines_img_select_path = MAN.store_image(v_lines, 'v_lines_select_img')
    return f'{h_lines_img_select_path},{v_lines_img_select_path}'

@app.route('/connect_lines', methods=['POST'])
def connect_lines():
    hough_threshold = int(request.form['houghthres'])

    h_lines_select_img = MAN.get_image('h_lines_select_img', grayscale=True)
    v_lines_select_img = MAN.get_image('v_lines_select_img', grayscale=True)
    h_lines, v_lines = SHI.connect_lines(h_lines_select_img, v_lines_select_img, hough_threshold)
    h_lines_connect_img_path = MAN.store_image(h_lines, 'h_lines_connect_img')
    v_lines_connect_img_path = MAN.store_image(v_lines, 'v_lines_connect_img')
    return f'{h_lines_connect_img_path},{v_lines_connect_img_path}'

@app.route('/merge_lines', methods=['POST'])
def merge_lines():
    dilation_kernel = int(request.form['dilationkernel'])

    h_lines_connect_img = MAN.get_image('h_lines_connect_img', grayscale=True)
    v_lines_connect_img = MAN.get_image('v_lines_connect_img', grayscale=True)

    merged_img = SHI.merge_lines(h_lines_connect_img, v_lines_connect_img, dilation_kernel)
    merged_img_path = MAN.store_image(merged_img, 'merged_img')
    return merged_img_path

@app.route('/grid_reconstruction', methods=['POST'])
def grid_reconstruction():
    cell_size = float(request.form['cell'])
    cell_size_px = DEF.horizontal_ratio * cell_size

    merged_img = MAN.get_image('merged_img', grayscale=True)
    reconstructed_img = SHI.grid_reconstruction(merged_img, cell_size_px)
    reconstructed_img_path = MAN.store_image(reconstructed_img, 'reconstructed_img')
    return reconstructed_img_path

@app.route('/capture_shifting_reference', methods=['POST'])
def capture_shifting_reference():
    shifting_base_img_raw  = CAP.capture_n()
    MAN.store_image(shifting_base_img_raw, 'shifting_base_img_raw')
    
    shifting_base_img = DEF.perspective_transform(shifting_base_img_raw)
    shifting_base_img_path = MAN.store_image(shifting_base_img, 'shifting_base_img')

    led_positions, _, shifting_base_detect_img = DEF.detect_led_tags(shifting_base_img, 'green', 1)
    shifting_base_detect_img_path = MAN.store_image(shifting_base_detect_img, 'shifting_base_detect_img')

    SHI.initial_shift_pos_px = led_positions[0]
    print(f'initial_shift_pos_px: {SHI.initial_shift_pos_px}' )

    return f'{shifting_base_img_path},{shifting_base_detect_img_path}'

@app.route('/detect_shifting', methods=['POST'])
def detect_shifting():
    mon_shift_img_raw = CAP.capture_n()   
    mon_shift_img = DEF.perspective_transform(mon_shift_img_raw)
    led_positions, _, _ = DEF.detect_led_tags(mon_shift_img, 'green', 1)
    point_b = led_positions[0]

    reconstructed_img = MAN.get_image('reconstructed_img', grayscale=True)
    dots_img = SHI.draw_dots(point_b, reconstructed_img)
    dots_img_path = MAN.store_image(dots_img, 'dots_img')
    h_stats, v_stats, crawl_img = SHI.crawl_lines(point_b[0], point_b[1], reconstructed_img)
    crawl_img_path = MAN.store_image(crawl_img, 'crawl_img')

    return f'{dots_img_path},{crawl_img_path}'
    

# endregion
#########################################################

if __name__ == "__main__":
    socketio.run(app, host='0.0.0.0')


# session data is now replaced by object reference DeformationDetector
# use the below code to serialize, store in session, deserialize data

# # serialize and store in session
# session['frame_pos_px'] = serialize_numpy(frame_pos_px)
# session['frame_dist_px'] = serialize_numpy(frame_dist_px)

# def serialize_numpy(obj):
#     # https://quick-adviser.com/are-numpy-arrays-json-serializable/
#     nplist = np.array(obj).tolist()
#     json_str = json.dumps(nplist)
#     return json_str


# def deserialze_numpy(str):
#     json_load = json.loads(str)
#     restored = np.asarray(json_load)
#     return restored