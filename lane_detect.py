import cv2
import numpy as np
from smbus2 import SMBus

# ---------------- I2C Setup ----------------
I2C_ADDR = 0x08     # Replace with your Arduino I2C address
bus = SMBus(1)  # 1 = /dev/i2c-1 on Raspberry Pi

def send_angle(angle):
    """Send steering angle as int via I2C"""
    angle_int = int(angle)
    # Limit to -30 to 30
    angle_int = max(-30, min(30, angle_int))
    try:
        bus.write_byte(I2C_ADDR, angle_int & 0xFF)
    except Exception as e:
        print("I2C Error:", e)

# ---------------- Lane Detection Functions ----------------
def preprocess(frame):
    """Convert frame to binary lane mask"""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # Simple edge + threshold
    edges = cv2.Canny(blur, 50, 150)

    # Mask lower half only
    h, w = edges.shape
    mask = np.zeros_like(edges)
    polygon = np.array([[
        (0, h),
        (w, h),
        (w, int(h*0.6)),
        (0, int(h*0.6))
    ]], np.int32)
    cv2.fillPoly(mask, polygon, 255)
    masked = cv2.bitwise_and(edges, mask)

    return masked

def fit_polynomial(binary_img):
    """Fit lane lines using sliding window"""
    histogram = np.sum(binary_img[binary_img.shape[0]//2:,:], axis=0)

    midpoint = int(histogram.shape[0]//2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    nwindows = 9
    window_height = int(binary_img.shape[0]//nwindows)
    nonzero = binary_img.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    leftx_current = leftx_base
    rightx_current = rightx_base
    margin = 50
    minpix = 50
    left_lane_inds = []
    right_lane_inds = []

    for window in range(nwindows):
        win_y_low = binary_img.shape[0] - (window+1)*window_height
        win_y_high = binary_img.shape[0] - window*window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin

        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                          (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                           (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)

        if len(good_left_inds) > minpix:
            leftx_current = int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:
            rightx_current = int(np.mean(nonzerox[good_right_inds]))

    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    if len(leftx) > 0 and len(lefty) > 0:
        left_fit = np.polyfit(lefty, leftx, 2)
    else:
        left_fit = None

    if len(rightx) > 0 and len(righty) > 0:
        right_fit = np.polyfit(righty, rightx, 2)
    else:
        right_fit = None

    return left_fit, right_fit

def measure_curvature_and_offset(left_fit, right_fit, ploty, img_shape):
    ym_per_pix = 30/720     # meters per pixel in y dimension
    xm_per_pix = 3.7/700    # meters per pixel in x dimension

    y_eval = np.max(ploty)

    if left_fit is not None and right_fit is not None:
        left_curverad = ((1 + (2*left_fit[0]*y_eval*ym_per_pix + left_fit[1])**2)**1.5) \
                        / np.absolute(2*left_fit[0])
        right_curverad = ((1 + (2*right_fit[0]*y_eval*ym_per_pix + right_fit[1])**2)**1.5) \
                         / np.absolute(2*right_fit[0])
        curvature = (left_curverad + right_curverad) / 2
    else:
        curvature = 10000  # straight approx

    # Vehicle center offset
    if left_fit is not None and right_fit is not None:
        h = img_shape[0]
        left_x = left_fit[0]*h**2 + left_fit[1]*h + left_fit[2]
        right_x = right_fit[0]*h**2 + right_fit[1]*h + right_fit[2]
        lane_center = (left_x + right_x) / 2
        vehicle_center = img_shape[1] / 2
        offset = (vehicle_center - lane_center) * xm_per_pix
    else:
        offset = 0

    return curvature, offset

def curvature_to_steering(curvature, offset):
    # Map curvature to steering angle
    if curvature == 0:
        curvature = 1e-6

    base_angle = np.arctan(1/curvature) * (180/np.pi) * 100  # scaled
    correction = offset * 15  # adjust lane centering sensitivity

    angle = -(base_angle + correction)
    return max(-30, min(30, angle))

# ---------------- Frame Processing ----------------
def process_frame(frame):
    binary = preprocess(frame)
    left_fit, right_fit = fit_polynomial(binary)
    ploty = np.linspace(0, binary.shape[0]-1, binary.shape[0])
    curvature, offset = measure_curvature_and_offset(left_fit, right_fit, ploty, binary.shape)
    steering_angle = curvature_to_steering(curvature, offset)
    return steering_angle

# ---------------- Main Loop ----------------
def main():
    cap = cv2.VideoCapture(0)  # Pi camera or USB cam

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        angle = process_frame(frame)
        print(f"Steering Angle: {angle:.2f}°")

        send_angle(angle)

    cap.release()

if __name__ == "__main__":
    main()
