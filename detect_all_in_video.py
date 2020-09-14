# coding:utf8
import cv2,sys,copy,json
import numpy as np
import lib_tennis

DEBUG_MSG=False


def detect_ball_in_one_frame_by_aspectratio(raw_frame):
    #--reference: https://github.com/aditirao7/tennis_ball_detection
    #--input: raw_frame: one capture frame from video
    #--output: list of (cX,cY,areac,aream,aspect_ratio,solidity)

    frame = raw_frame.copy()

    frame = cv2.GaussianBlur(frame, (3, 3), 0)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HLS_FULL)

    greenlower = np.array([42, 80, 61])
    greenupper = np.array([67, 226, 255])
    # greenlower = np.array([29, 37, 63])
    # greenupper = np.array([145, 255, 255])
    mask = cv2.inRange(hsv, greenlower, greenupper)

    kernel = np.ones((5, 5), np.uint8)
    mask=cv2.erode(mask,kernel,iterations = 1)
    mask = cv2.dilate(mask, kernel, iterations=10)
    mask = cv2.erode(mask, kernel, iterations=4)

    ret, thresh = cv2.threshold(mask, 200, 255, 0)
    _, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    res = cv2.bitwise_and(frame, frame, mask=mask)
    res_list=[]
    for c in contours:
        M = cv2.moments(c)
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
        else:
            cX, cY = 0, 0
        x, y, w, h = cv2.boundingRect(c)
        aspect_ratio = float(w) / h
        (x, y), radius = cv2.minEnclosingCircle(c)
        x = np.int(x)
        y = np.int(y)
        radius = np.int(radius)
        areac = M["m00"]
        aream = np.pi * radius ** 2
        hull = cv2.convexHull(c)
        hull_area = cv2.contourArea(hull)
        solidity = 0
        if hull_area != 0:
            solidity = float(areac) / hull_area
        if DEBUG_MSG:
            print((x, y), '\t', (cX, cY), '\t', areac, '\t', aream, '\t', aspect_ratio, '\t', solidity)


        res_list.append((int(cX),int(cY),areac,aream,aspect_ratio,solidity))
    return res_list


def detect_motion_in_one_frame_by_BackgroundSubtractor(bs,raw_frame):
    #--input: raw_frame: one capture frame from video
    #--output: list of

    frame = raw_frame.copy()
    fg_mask = bs.apply(frame)  # 获取 foreground mask

    # 对原始帧进行膨胀去噪
    th = cv2.threshold(fg_mask.copy(), 244, 255, cv2.THRESH_BINARY)[1]
    th = cv2.erode(th, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=2)
    dilated = cv2.dilate(th, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (8, 3)), iterations=2)
    # 获取所有检测框
    _, contours, hier = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    res_list = []
    for c in contours:
        # 获取矩形框边界坐标
        x, y, w, h = cv2.boundingRect(c)
        # 计算矩形框的面积
        area = cv2.contourArea(c)
        if area < 1200:
            #cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            res_list.append((int(x),int(y),int(x+w),int(y+h)))

    return res_list


def get_phase1_from_motion_and_ball(res_motion,res_ball):
    r_final = []
    for r_ball in res_ball:
        for r_motion in res_motion:
            if r_ball[0]>=r_motion[0] and r_ball[1]>=r_motion[1] and r_ball[0]<=r_motion[2] and r_ball[1]<=r_motion[3]:
                r_final.append(r_ball)
                break

    return r_final

def detect_ball_phase1(video_file,history=5):
    #--video_file: input video file
    #--output: res_list, frame: motion_detect: ball_detect: detected_ball:

    camera = cv2.VideoCapture(video_file)

    bs = cv2.createBackgroundSubtractorKNN(detectShadows=True)  # 背景减除器，设置阴影检测
    bs.setHistory(history)
    frames = 0

    res_list=[]

    while True:
        #--get one capture frame
        res, frame = camera.read()
        if not res:
            break
        if frames < history:
            frames += 1
            continue

        #--get motion and ball
        new_row={}
        new_row['frame'] = frame
        new_row['motion_detect'] = detect_motion_in_one_frame_by_BackgroundSubtractor(bs,frame)
        new_row['ball_detect'] = detect_ball_in_one_frame_by_aspectratio(frame)
        new_row['phase1'] = []

        #--get phase1 detected ball
        res = get_phase1_from_motion_and_ball(new_row['motion_detect'],new_row['ball_detect'])
        if len(res)==1:
            new_row['phase1'].append((res[0][0], res[0][1]))

        res_list.append(new_row)

    camera.release()

    return res_list



def detect_ball_phase2(res_phase1):
    points_list = lib_tennis.int_xy_data(res_phase1, 'phase1')

    #--把phase1的点分为3部分，下降，上升，不确定
    p_up,p_down,p_not = lib_tennis.split_to_3_parts(points_list)
    print('--->',len(points_list),len(p_down),len(p_not),len(p_up))

    #--为phase1的上升、下降曲线拟合
    curve_down = lib_tennis.fit_curve(p_down)
    curve_not = lib_tennis.fit_curve_not(p_not)
    curve_up = lib_tennis.fit_curve(p_up)

    curve_list = [curve_up,curve_not,curve_down]

    #lib_tennis.plot_curve(curve_list)

    #--判断motion_detect区域，不在phase1中，但是在拟合曲线附近，则加上


    return res_phase1,curve_list


def detect_ball_in_video(video_file):
    #--res_phase1, frame: motion_detect: ball_detect: phase1_ball:
    res_phase1 = detect_ball_phase1(video_file, history=5)
    lib_tennis.debug_res_phase(res_phase1,phase_name='phase1',out_img='phase1.jpg')

    res_phase2,curve_list = detect_ball_phase2(res_phase1)
    #debug_res_phase(res_phase2,phase_name='phase2',out_img='phase2.jpg')

    return res_phase2,curve_list

def find_bounce_point(res_phase2,curve_list):
    #--找到final_curve_down and final_curve_up
    final_curve_down,final_curve_up = lib_tennis.find_best_two_curve(curve_list)
    curve_list = [final_curve_up,final_curve_down]
    #lib_tennis.plot_curve(curve_list)

    #--根据两条曲线找到交点，即弹跳点
    h,w,_ = res_phase2[0]['frame'].shape
    bounce_point = lib_tennis.get_the_bounce_point(curve_list,h)

    lib_tennis.debug_res_final(res_phase2,curve_list,bounce_point,out_img='phase_final.jpg')

    return

def main(video_file):
    res_phase2,curve_list = detect_ball_in_video(video_file)
    find_bounce_point(res_phase2,curve_list)

if __name__ == '__main__':
    from fire import Fire
    Fire(main)
