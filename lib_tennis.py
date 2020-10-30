# -*- coding: utf-8 -*-
import cv2,os
import copy
import numpy as np
from lmfit.models import GaussianModel, LorentzianModel, LinearModel, PolynomialModel
from lmfit.printfuncs import getfloat_attr

def np_resize_max_size(img,max_size):
    height, width, channel = img.shape

    # magnify image size
    target_size = max(height, width, max_size)

    if target_size > max_size:
        target_size = max_size
    
    ratio = float(target_size) / float(max(height, width))

    target_h, target_w = int(height * ratio), int(width * ratio)
    proc = cv2.resize(img, (target_w, target_h), interpolation = cv2.INTER_LINEAR)

    return proc,ratio

def debug_res_phase(res_phase1,phase_name,out_img=None,max_width=640):
    for r in res_phase1:
        raw_frame = copy.deepcopy(res_phase1[0]['frame'])
        frame_resize,ratio = np_resize_max_size(r['frame'],max_width)
        for r_ball in r[phase_name]:
            cv2.circle(frame_resize, (int(r_ball[0][0]*ratio), int(r_ball[0][1]*ratio)), 4, (0, 0, 255), thickness=-1)
            if out_img is not None:
                cv2.circle(raw_frame, (int(r_ball[0][0]), int(r_ball[0][1])), 3, (0, 255, 255), thickness=-1)
        cv2.imshow("detection", frame_resize)
        if cv2.waitKey(110) & 0xff == 27:
            break
    if out_img is not None:
        cv2.imwrite(out_img,raw_frame)
    return

def debug_res_final(res_phase1,curve_list,bounce_point,out_img=None):
    raw_frame = copy.deepcopy(res_phase1[0]['frame'])
    h, w, _ = raw_frame.shape
    for c in curve_list:
        for idx in range(len(c[1])):
            if out_img is not None:
                x=int(c[1][idx])
                y=h-int(c[2][idx])
                cv2.circle(raw_frame, (x,y), 3, (0, 255, 255), thickness=-1)
                if idx<len(c[1])-1:
                    x = int(c[1][idx])
                    y = h-int(c[0][idx])
                    x2 = int(c[1][idx+1])
                    y2 = h-int(c[0][idx+1])
                    cv2.line(raw_frame, (x,y), (x2,y2), (0,0,255),2)


    if out_img is not None:
        cv2.circle(raw_frame, (bounce_point[0], bounce_point[1]), 9, (255, 0, 0), thickness=-1)
        cv2.imwrite(out_img,raw_frame)
    return


def get_the_bounce_point(curve_list,h):
    [curve_up, curve_down] = curve_list
    points_up = [[curve_up[1][i2],curve_up[2][i2]] for i2 in range(len(curve_up[1]))]
    points_down = [[curve_down[1][i2],curve_down[2][i2]] for i2 in range(len(curve_down[1]))]

    #print('down',len(points_down),points_down)
    #print('up',len(points_up),points_up)
    x = (int)((points_down[-1][0]+points_up[0][0])/2)
    y = h-(int)((points_down[-1][1]+points_up[0][1])/2)
    print('mid:',x,y)

    min_delta = 1000
    for xs in range(points_down[-1][0]-5,points_up[0][0]+5):
        y_up_eval = curve_up[4].eval(x=np.array([xs]))[0]
        y_down_eval = curve_down[4].eval(x=np.array([xs]))[0]
        if abs(y_down_eval-y_up_eval)<min_delta:
            min_delta = abs(y_down_eval-y_up_eval)
            x = xs
            y = h-int((y_down_eval+y_up_eval)/2)
    print('end:',x,y,min_delta)

    return (x,y)

def int_xy_data(res_phase,phase_name):
    #--output:ps_list, (x,y)
    ps_list = []
    for r in res_phase:
        h,w,_=r['frame'].shape
        for r_ball in r[phase_name]:
            x,y = int(r_ball[0][0]),int(h-r_ball[0][1])
            ps_list.append((x,y))

    return ps_list

def split_to_3_parts(ps_list,F_down_min_metric=3,F_down_offset=2,F_up_min_metric=3,F_up_offset=1):
    p_up, p_down, p_not = [],[],[]

    #--input:ps_list, (x,y)
    y = [r[1] for r in ps_list]

    #--按时序遍历，找到连续3个后续的点的y都是上升，
    for idx in range(len(y)-F_down_min_metric):
        found_bad=False
        for j in range(idx,idx+F_down_min_metric):
            if y[j]<(y[j+1]-F_down_offset):
                found_bad=True
                break
        if found_bad:
            break
        else:
            p_down.append(ps_list[idx])
            continue

    #--按时序遍历，找到连续3个后续的点的y都是上升，
    for idx in range(len(y)-1,F_up_min_metric,-1):
        #print('up',idx)
        found_bad=False
        for j in range(idx,idx-F_up_min_metric,-1):
            if y[j]<(y[j-1]-F_up_offset):
                found_bad=True
                break
        if found_bad:
            #print('--->bad',y[idx-F_up_min_metric:idx])
            break
        else:
            #print('--->OK',y[idx-F_up_min_metric:idx])
            p_up.append(ps_list[idx])
            continue
    p_up = p_up[::-1]

    #print(22222,len(ps_list),len(p_down),len(p_not),len(p_up))

    #'''
    delta_num = len(ps_list)- len(p_down) - len(p_up)
    if delta_num >0:
        p_not = ps_list[len(p_down):len(p_down)+delta_num]
    elif delta_num < 0:
        p_not = ps_list[len(p_down)-abs(delta_num):len(p_down)]
        p_up = p_up[:len(p_down)-abs(delta_num)]
        p_down = p_down[abs(delta_num):]
    #'''

    return p_up,p_down,p_not



def fit_curve(ps_list):
    x = np.array([r[0] for r in ps_list])
    y = np.array([r[1] for r in ps_list])

    mod = GaussianModel()

    pars = mod.guess(y, x=x)
    bestresult = mod.fit(y, pars, x=x)

    return (bestresult.best_fit,x,y,getfloat_attr(bestresult, 'chisqr'),bestresult)

def fit_curve_not(ps_list):
    #--input:ps_list, (x,y)
    x = np.array([r[0] for r in ps_list])
    y = np.array([r[1] for r in ps_list])

    return (y,x,y)


def plot_curve(curve_list,res_phase1=None,bounce_point=None):
    import matplotlib.pyplot as plt

    for r_curve in curve_list:
        #r_curve[0].plot()
        plt.plot(r_curve[1], r_curve[0], 'r-')
        plt.scatter(r_curve[1], r_curve[2])
    if bounce_point is not None and res_phase1 is not None:
        raw_frame = copy.deepcopy(res_phase1[0]['frame'])
        h, w, _ = raw_frame.shape
        plt.scatter(bounce_point[0],h-bounce_point[1])


    plt.show()

def find_best_two_curve(curve_list):
    [curve_up, curve_not, curve_down] = curve_list
    points_up = [[curve_up[1][i2],curve_up[2][i2]] for i2 in range(len(curve_up[1]))]
    points_down = [[curve_down[1][i2],curve_down[2][i2]] for i2 in range(len(curve_down[1]))]
    points_not = [[curve_not[1][i2],curve_not[2][i2]] for i2 in range(len(curve_not[1]))]

    if len(points_not)==0:
        return curve_down,curve_up

    curve_candi_list=[]
    for idx in range(-1,len(points_not)):
        todo_points_down = copy.deepcopy(points_down)
        todo_points_up = copy.deepcopy(points_up)

        if idx!=-1:
            todo_points_down = todo_points_down+points_not[:idx+1]
            todo_points_up = points_not[idx+1:]+todo_points_up
        else:
            todo_points_up = points_not+todo_points_up

        c_up = fit_curve(todo_points_up)
        c_down = fit_curve(todo_points_down)
        up_down_loss = float(c_up[3])+float(c_down[3])
        print(idx,len(todo_points_down),len(todo_points_up),up_down_loss)
        curve_candi_list.append((c_down,c_up,up_down_loss))

    curve_candi_list = sorted(curve_candi_list, key=lambda row: row[2])

    return curve_candi_list[0][0], curve_candi_list[0][1]



