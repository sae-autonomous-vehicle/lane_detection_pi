#!/usr/bin/env python3
"""
lite_lane.py — lightweight lane detection pipeline optimized for Raspberry Pi.
Dependencies: Python3, numpy, OpenCV (cv2)
Usage:
  # Calibrate once (images in ./camera_cal/) -> saves camera_calib.p
  python lite_lane.py --calibrate

  # Run on a video (or use camera)
  python lite_lane.py --video input.mp4 --out out.mp4
"""

import os
import cv2
import glob
import pickle
import argparse
import numpy as np

# ---------- PARAMETERS (tune these) ----------
CALIB_DIR = "camera_cal"          # chessboard images directory (if calibrating)
CALIB_FILE = "camera_calib.p"     # saved calibration
RESIZE_WIDTH = 640                # choose 320/480/640 depending on your Pi speed
RESIZE_HEIGHT = None              # keep None to preserve aspect ratio
ROI_VERT_TOP_RATIO = 0.55         # region-of-interest start (relative to height)
NWINDOWS = 9
MARGIN = 100
MINPIX = 50
SOBEL_KSIZE = 3
# meters per pixel — typical Udacity conversions (can be tuned)
YM_PER_PIX = 30/720.0
XM_PER_PIX = 3.7/700.0
# ------------------------------------------------

def resize(img, width=RESIZE_WIDTH, height=RESIZE_HEIGHT):
    if height is None:
        h, w = img.shape[:2]
        scale = width / float(w)
        return cv2.resize(img, (width, int(h*scale)), interpolation=cv2.INTER_AREA)
    return cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)

def calibrate_camera_from_dir(chessboard_dir=CALIB_DIR, nx=9, ny=6):
    # Prepare object points
    objp = np.zeros((ny*nx, 3), np.float32)
    objp[:,:2] = np.mgrid[0:nx, 0:ny].T.reshape(-1, 2)
    objpoints = []
    imgpoints = []
    images = glob.glob(os.path.join(chessboard_dir, 'calibration*.jpg')) + glob.glob(os.path.join(chessboard_dir, '*.jpg'))
    if not images:
        raise FileNotFoundError("No calibration images found in %s" % chessboard_dir)
    for fname in images:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)
        if ret:
            imgpoints.append(corners)
            objpoints.append(objp)
    # calibrate
    img_shape = gray.shape[::-1]
    ret, mtx, dist, _, _ = cv2.calibrateCamera(objpoints, imgpoints, img_shape, None, None)
    pickle.dump({'mtx': mtx, 'dist': dist}, open(CALIB_FILE, 'wb'))
    print("Saved camera calibration to", CALIB_FILE)
    return mtx, dist

def load_calibration(calib_file=CALIB_FILE):
    if os.path.exists(calib_file):
        data = pickle.load(open(calib_file, 'rb'))
        return data['mtx'], data['dist']
    return None, None

def undistort(img, mtx, dist):
    return cv2.undistort(img, mtx, dist, None, mtx) if mtx is not None else img

def get_perspective_matrices(img):
    h, w = img.shape[:2]
    # Source and destination points (relative coords, tuned for typical forward-facing camera)
    src = np.float32([
        [w*0.45, h*0.63],
        [w*0.55, h*0.63],
        [w*0.9,  h*0.95],
        [w*0.1,  h*0.95],
    ])
    dst = np.float32([
        [w*0.25, 0.0],
        [w*0.75, 0.0],
        [w*0.75, h*1.0],
        [w*0.25, h*1.0],
    ])
    M = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst, src)
    return M, Minv

def threshold_pipeline(img):
    """Return binary (0/1) uint8 image after combined color+Sobel thresholding"""
    # Convert to HLS and extract S channel
    hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
    s_channel = hls[:,:,2]
    # Sobel x on L channel
    l_channel = hls[:,:,1]
    sobelx = np.absolute(cv2.Sobel(l_channel, cv2.CV_64F, 1, 0, ksize=SOBEL_KSIZE))
    sobelx = (255 * sobelx / np.max(sobelx)).astype(np.uint8)
    # Thresholds (fast defaults)
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= 150) & (s_channel <= 255)] = 1
    sx_binary = np.zeros_like(sobelx)
    sx_binary[(sobelx >= 20) & (sobelx <= 100)] = 1
    # Combine
    combined = np.zeros_like(sx_binary)
    combined[(s_binary == 1) | (sx_binary == 1)] = 1
    return (combined * 255).astype(np.uint8)

def warp(img, M):
    h, w = img.shape[:2]
    return cv2.warpPerspective(img, M, (w, h), flags=cv2.INTER_LINEAR)

def find_lane_pixels_sliding(binary_warped):
    # Assume binary_warped is single-channel (0/255)
    binary = (binary_warped // 255).astype(np.uint8)
    h, w = binary.shape
    histogram = np.sum(binary[h//2:,:], axis=0)
    midpoint = histogram.shape[0] // 2
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    # Sliding windows
    window_height = np.int(h // NWINDOWS)
    nonzero = binary.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    leftx_current = leftx_base
    rightx_current = rightx_base
    left_lane_inds = []
    right_lane_inds = []

    for window in range(NWINDOWS):
        win_y_low = h - (window + 1) * window_height
        win_y_high = h - window * window_height
        win_xleft_low = leftx_current - MARGIN
        win_xleft_high = leftx_current + MARGIN
        win_xright_low = rightx_current - MARGIN
        win_xright_high = rightx_current + MARGIN

        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                          (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                           (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        if len(good_left_inds) > MINPIX:
            leftx_current = int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > MINPIX:
            rightx_current = int(np.mean(nonzerox[good_right_inds]))

    left_lane_inds = np.concatenate(left_lane_inds) if len(left_lane_inds) else []
    right_lane_inds = np.concatenate(right_lane_inds) if len(right_lane_inds) else []

    leftx = nonzerox[left_lane_inds] if left_lane_inds.size else np.array([])
    lefty = nonzeroy[left_lane_inds] if left_lane_inds.size else np.array([])
    rightx = nonzerox[right_lane_inds] if right_lane_inds.size else np.array([])
    righty = nonzeroy[right_lane_inds] if right_lane_inds.size else np.array([])

    left_fit = np.polyfit(lefty, leftx, 2) if leftx.size and lefty.size else None
    right_fit = np.polyfit(righty, rightx, 2) if rightx.size and righty.size else None
    return left_fit, right_fit, leftx, lefty, rightx, righty

def search_around_poly(binary_warped, left_fit, right_fit):
    binary = (binary_warped // 255).astype(np.uint8)
    nonzero = binary.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    left_lane_inds = ((nonzerox > (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2] - MARGIN)) &
                      (nonzerox < (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2] + MARGIN)))
    right_lane_inds = ((nonzerox > (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2] - MARGIN)) &
                       (nonzerox < (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2] + MARGIN)))
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]
    left_fit_new = np.polyfit(lefty, leftx, 2) if leftx.size and lefty.size else left_fit
    right_fit_new = np.polyfit(righty, rightx, 2) if rightx.size and righty.size else right_fit
    return left_fit_new, right_fit_new, leftx, lefty, rightx, righty

def fit_poly_points(fit, ploty):
    return fit[0]*ploty**2 + fit[1]*ploty + fit[2]

def measure_curvature_real(left_fit, right_fit, ploty):
    # Evaluate curvature in meters (at bottom of image)
    y_eval = np.max(ploty)
    # convert coefficients to real world space
    left_fit_cr = np.array([
        left_fit[0]*(XM_PER_PIX/(YM_PER_PIX**2)),
        left_fit[1]*(XM_PER_PIX/YM_PER_PIX),
        left_fit[2]*XM_PER_PIX
    ])
    right_fit_cr = np.array([
        right_fit[0]*(XM_PER_PIX/(YM_PER_PIX**2)),
        right_fit[1]*(XM_PER_PIX/YM_PER_PIX),
        right_fit[2]*XM_PER_PIX
    ])
    left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*YM_PER_PIX + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
    right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*YM_PER_PIX + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])
    return (left_curverad + right_curverad) / 2.0

def draw_lane(original_img, binary_warped, left_fit, right_fit, Minv):
    h, w = binary_warped.shape
    ploty = np.linspace(0, h-1, h)
    left_fitx = fit_poly_points(left_fit, ploty)
    right_fitx = fit_poly_points(right_fit, ploty)

    # create an image to draw the lanes on
    color_warp = np.zeros((h, w, 3), dtype=np.uint8)
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])) )])
    pts = np.hstack((pts_left, pts_right))
    cv2.fillPoly(color_warp, np.int_([pts]), (0,255,0))
    # warp back to original
    newwarp = cv2.warpPerspective(color_warp, Minv, (original_img.shape[1], original_img.shape[0]))
    result = cv2.addWeighted(original_img, 1, newwarp, 0.3, 0)
    return result

def process_frame(frame, state):
    # state stores: mtx, dist, M, Minv, left_fit, right_fit, first_detected(Boolean)
    img = resize(frame)
    und = undistort(img, state['mtx'], state['dist'])
    if state['M'] is None:
        state['M'], state['Minv'] = get_perspective_matrices(und)

    # threshold and warp
    binary = threshold_pipeline(und)
    binary_warped = warp(binary, state['M'])

    # detect lanes
    if not state.get('first_detected', False):
        left_fit, right_fit, lx, ly, rx, ry = find_lane_pixels_sliding(binary_warped)
        state['first_detected'] = True if (left_fit is not None and right_fit is not None) else False
    else:
        left_fit, right_fit, lx, ly, rx, ry = search_around_poly(binary_warped, state['left_fit'], state['right_fit'])

    if left_fit is None or right_fit is None:
        # fallback: try sliding windows again
        left_fit, right_fit, lx, ly, rx, ry = find_lane_pixels_sliding(binary_warped)

    if left_fit is None or right_fit is None:
        return img  # unable to detect this frame, return original resized

    # update state fits
    state['left_fit'], state['right_fit'] = left_fit, right_fit

    # draw lane onto original-sized image (use Minv to warp back to original size)
    draw = draw_lane(img, binary_warped, left_fit, right_fit, state['Minv'])

    # curvature & offset
    h = binary_warped.shape[0]
    ploty = np.linspace(0, h-1, h)
    curvature = measure_curvature_real(left_fit, right_fit, ploty)
    # vehicle offset
    bottom_y = h - 1
    left_x = fit_poly_points(left_fit, bottom_y)
    right_x = fit_poly_points(right_fit, bottom_y)
    lane_center = (left_x + right_x) / 2.0
    image_center = binary_warped.shape[1] / 2.0
    center_offset_m = (image_center - lane_center) * XM_PER_PIX

    # annotate
    cv2.putText(draw, f"Radius: {curvature:.0f} m", (20,40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 2)
    cv2.putText(draw, f"Offset: {center_offset_m:.2f} m", (20,80), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 2)
    return draw

def main(args):
    parser = argparse.ArgumentParser()
    parser.add_argument('--calibrate', action='store_true', help='Run camera calibration and exit')
    parser.add_argument('--video', type=str, help='Input video file (or use --camera)')
    parser.add_argument('--out', type=str, default='out.mp4', help='Output video path')
    parser.add_argument('--camera', action='store_true', help='Use camera stream (device 0)')
    opts = parser.parse_args(args)

    if opts.calibrate:
        calibrate_camera_from_dir()
        return

    mtx, dist = load_calibration()
    state = {'mtx': mtx, 'dist': dist, 'M': None, 'Minv': None, 'left_fit': None, 'right_fit': None, 'first_detected': False}

    if opts.camera:
        cap = cv2.VideoCapture(0)
    else:
        if not opts.video:
            print("Provide --video or --camera")
            return
        cap = cv2.VideoCapture(opts.video)

    # Setup output writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        processed = process_frame(frame, state)
        if out is None:
            h, w = processed.shape[:2]
            out = cv2.VideoWriter(opts.out, fourcc, max(10.0, cap.get(cv2.CAP_PROP_FPS) or 20.0), (w, h))
        out.write(processed)
        # show (optional)
        cv2.imshow("lane", processed)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    if out: out.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    import sys
    main(sys.argv[1:])
