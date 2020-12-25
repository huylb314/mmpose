import json
import os
import numpy as np
import mmcv
from matplotlib import pyplot as plt
import cv2
import math
from pylab import rcParams
import copy
import time
from sklearn import metrics
import glob

dict_angles = [
    ['right_hip', 'right_knee', 'right_ankle'], 
    ['left_hip', 'right_hip', 'right_knee'],
    ['right_shoulder', 'right_hip', 'left_hip'],
    ['left_shoulder', 'right_shoulder', 'right_hip'],
    ['left_shoulder', 'right_shoulder', 'right_elbow'],
    ['right_shoulder', 'right_elbow', 'right_wrist'],
    ['left_ankle', 'left_knee', 'left_hip'],
    ['left_knee', 'left_hip', 'right_hip'],
    ['right_hip', 'left_hip', 'left_shoulder'],
    ['left_hip', 'left_shoulder', 'right_shoulder'],
    ['left_elbow', 'left_shoulder', 'right_shoulder'],
    ['left_wrist', 'left_elbow', 'left_shoulder'],
]

dict_name_angles = [
    "RKnee_Angle", 
    "RHip_Angle",
    "RUperHip_Angle",
    "RShoulder_Angle",
    "RShoulderElbow_Angle",
    "RElbow_Angle",
    "LKnee_Angle",
    "LHip_Angle",
    "LUperHip_Angle",
    "LShoulder_Angle",
    "LShoulderElbow_Angle",
    "LElbow_Angle",
]

classnames = dict({0: 'nose',
        1: 'left_eye',
        2: 'right_eye',
        3: 'left_ear',
        4: 'right_ear',
        5: 'left_shoulder',
        6: 'right_shoulder',
        7: 'left_elbow',
        8: 'right_elbow',
        9: 'left_wrist',
        10: 'right_wrist',
        11: 'left_hip',
        12: 'right_hip',
        13: 'left_knee',
        14: 'right_knee',
        15: 'left_ankle',
        16: 'right_ankle'})

id_classnames = {v: k for k, v in classnames.items()}

palette = np.array([[255, 128, 0], [255, 153, 51], [255, 178, 102],
                        [230, 230, 0], [255, 153, 255], [153, 204, 255],
                        [255, 102, 255], [255, 51, 255], [102, 178, 255],
                        [51, 153, 255], [255, 153, 153], [255, 102, 102],
                        [255, 51, 51], [153, 255, 153], [102, 255, 102],
                        [51, 255, 51], [0, 255, 0], [0, 0, 255], [255, 0, 0],
                        [255, 255, 255]])

skeleton = [[16, 14], [14, 12], [17, 15], [15, 13], [12, 13], [6, 12],
                    [7, 13], [6, 7], [6, 8], [7, 9], [8, 10], [9, 11], [2, 3],
                    [1, 2], [1, 3], [2, 4], [3, 5], [4, 6], [5, 7]]
pose_limb_color = palette[[
            16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16
        ]]
pose_kpt_color = palette[[
    16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16
]]

rcParams['figure.figsize'] = 10, 10
folder_path = "./golf_pose"
# frontal view
folder_pos = os.path.join(folder_path, "positive", "3_1")
folder_neg = os.path.join(folder_path, "negative", "kakao_905")

fps = 30
size = (477*5, 900)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
videoWriter = cv2.VideoWriter("./out_{}.mp4".format(time.time()), fourcc, fps, size)

np_pos = []
raw_pos = []
image_file_pos = []
np_kp_pos_pairs = []
kp_name_pos_pairs = []

not_include = [1, 2, 3, 4, 5]

for file_path_jpg in sorted(glob.glob(folder_pos+"/*.jpg")):
    file_name_jpg = os.path.basename(file_path_jpg)
    file_name, jpg = file_name_jpg.split(".jpg")
    file_name_json = file_name + ".json"
    
    kp_pos = None
    with open(os.path.join(folder_pos, file_name_json)) as f:
        anno_pos = json.load(f)
        kp_pos = anno_pos['annotations'][0]['keypoints'][:17]
        kp_comp_pos = np.array([[x, y] for (x, y, c) in kp_pos])
        kp_name_pos = anno_pos['categories'][0]['keypoints'][:17]
        sk_pos = anno_pos['categories'][0]['skeleton']
        if len(kp_pos) > 0:
            np_pos.append(np.array(kp_pos))
            raw_pos.append(anno_pos)
            image_file_pos.append(file_path_jpg)
            kp_name_pos_pairs.append([(kp_name_pos[p1-1], kp_name_pos[p2-1]) for p1, p2 in sk_pos \
                                if p1 not in not_include and p2 not in not_include])

            np_kp_pos_pairs.append([np.array(kp_comp_pos[p1-1] - kp_comp_pos[p2-1]) for p1, p2 in sk_pos \
                                if p1 not in not_include and p2 not in not_include])
            
np_pos = np.array(np_pos)
np_kp_pos_pairs = np.array(np_kp_pos_pairs)

def vectorizer_kp(folder_path):
    np_kp_ = []
    raw_ = []
    image_file_ = []
    np_kp_pairs_ = []
    kp_name_pairs_ = []
    not_include = [1, 2, 3, 4, 5]

    for file_path_jpg in sorted(glob.glob(folder_path+"/*.jpg")):
        file_name_jpg = os.path.basename(file_path_jpg)
        file_name, jpg = file_name_jpg.split(".jpg")
        file_name_json = file_name + ".json"
        
        kp_pos = None
        with open(os.path.join(folder_path, file_name_json)) as f:
            anno = json.load(f)
            kp = anno['annotations'][0]['keypoints'][:17]
            kp_comp = np.array([[x, y] for (x, y, c) in kp])
            kp_name = anno['categories'][0]['keypoints'][:17]
            sk = anno['categories'][0]['skeleton']
            if len(kp) > 0:
                np_kp_.append(np.array(kp))
                raw_.append(anno)
                image_file_.append(file_path_jpg)
                kp_name_pairs_.append([(kp_name[p1-1], kp_name[p2-1]) for p1, p2 in sk \
                                    if p1 not in not_include and p2 not in not_include])

                np_kp_pairs_.append([np.array(kp_comp[p1-1] - kp_comp[p2-1]) for p1, p2 in sk \
                                    if p1 not in not_include and p2 not in not_include])

    np_kp_ = np.array(np_kp_)
    np_kp_pairs_ = np.array(np_kp_pairs_)

    return np_kp_, raw_, image_file_,  np_kp_pairs_, kp_name_pairs_

np_kp_pos, raw_pos, image_file_pos,  np_kp_pairs_pos, kp_name_pairs_pos = vectorizer_kp(folder_pos)
np_kp_neg, raw_neg, image_file_neg,  np_kp_pairs_neg, kp_name_pairs_neg = vectorizer_kp(folder_neg)

compare_kp_pairs_neg = np_kp_pairs_neg[:1, :, :]
compare_kp_pairs_pos = np_kp_pairs_pos[:10, :, :]

np_kp_pairs_neg[0], np_kp_pairs_pos[0]


import math
from numpy import linalg as LA

def pairwise_angles(v1, v2):
    acos_invalue = np.sum(v1*v2, axis=-1) / (LA.norm(v1, axis=-1) * LA.norm(v2, axis=-1))
    acos_value = np.arccos(np.round(acos_invalue, 2))
    degrees_value = np.degrees(acos_value)
    where_are_nans = np.isnan(degrees_value)
    degrees_value[where_are_nans] = 20000
    return degrees_value

threshold = 10
fps = 30
size = (477*5, 900)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
videoWriter = cv2.VideoWriter("./out_{}.mp4".format(time.time()), fourcc, fps, size)
for pairs_, image_file_, raw_ in zip(np_kp_pairs_neg, image_file_neg, raw_neg):    
    color_limb_transparency = np.array([False, False, False, False, False, False, False, False, False, \
        False, False, False, False, False, False, False, False, False, False])
    color_kpt_transparency = np.array([False, False, False, False, False, False, False, False, False, \
        False, False, False, False, False, False, False, False])

    time_start = time.time()
    pairs_temp = pairs_[None, :, :]
    p12 = pairwise_angles(pairs_temp, np_kp_pairs_pos)
    p12_sum = np.sum(p12, axis=-1)
    p12_argmin = np.argmin(p12_sum, axis=-1)

    image_neg_temp = mmcv.imread(image_file_)
    image_pos_temp = mmcv.imread(image_file_pos[p12_argmin])

    pose_limb_plot = np.concatenate((p12[p12_argmin] > threshold, [False]*7))
    color_kpt = np.array([
        16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16
    ])
    color_limb = np.array([
        16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16
    ])

    img_draw_temp = show_result(
                        image_file_,
                        raw_['annotations'],
                        skeleton,
                        classnames,
                        angles_list=[p12[p12_argmin]],
                        pose_kpt_color=palette[color_kpt],
                        pose_limb_plot=pose_limb_plot,
                        pose_limb_color=palette[color_limb])
    img_draw_neg = cv2.resize(img_draw_temp, (477, 900))
    img_draw_pos = cv2.resize(image_pos_temp, (477*4, 900))

    img_draw = np.hstack((img_draw_neg, img_draw_pos))
    print ("time_end: ", time.time() - time_start, p12_argmin)
    videoWriter.write(img_draw)

videoWriter.release()

def show_result(img,
                result,
                skeleton=None,
                classnames=None,
                kpt_score_thr=0.0,
                bbox_color='green',
                pose_kpt_color=None,
                pose_limb_plot=None,
                pose_limb_color=None,
                radius=4,
                text_color=(255, 0, 0),
                thickness=1,
                font_scale=0.5,
                win_name='',
                angles_list=None,
                show=False,
                wait_time=0,
                out_file=None):

        img = mmcv.imread(img)
        img = img.copy()
        img_h, img_w, _ = img.shape
        not_include = [0, 1, 2, 3, 4, 5]

        bbox_result = []
        pose_result = []
        for res in result:
            bbox_result.append(res['bbox'])
            pose_result.append(res['keypoints'])

        if len(bbox_result) > 0:
            bboxes = np.vstack(bbox_result)
            # draw bounding boxes
            # mmcv.imshow_bboxes(
            #     img,
            #     bboxes,
            #     colors=bbox_color,
            #     top_k=-1,
            #     thickness=thickness,
            #     show=False,
            #     win_name=win_name,
            #     wait_time=wait_time,
            #     out_file=None)

            for _, (kpts, angles) in enumerate(zip(pose_result, angles_list)):
                # draw each point on image
                if pose_kpt_color is not None:
                    kpts = kpts[:17]
                    assert len(pose_kpt_color) == len(kpts)
                    for kid, kpt in enumerate(kpts):
                        if kid in not_include:
                            continue
                        x_coord, y_coord, kpt_score = int(kpt[0]), int(
                            kpt[1]), kpt[2]
                        if kpt_score > kpt_score_thr:
                            img_copy = img.copy()
                            r, g, b = pose_kpt_color[kid]
                            cv2.circle(img_copy, (int(x_coord), int(y_coord)),
                                       radius, (int(r), int(g), int(b)), -1)
                            if classnames:
                                cv2.putText(img_copy, "{}".format(classnames[kid]),(int(x_coord), int(y_coord)),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (int(r), int(g), int(b)), 1)
                            transparency = max(0, min(1, kpt_score))
                            cv2.addWeighted(
                                img_copy,
                                transparency,
                                img,
                                1 - transparency,
                                0,
                                dst=img)

                # draw limbs
                if skeleton is not None and pose_limb_color is not None:
                    assert len(pose_limb_color) == len(skeleton)
                    for sk_id, sk in enumerate(skeleton):
                        if pose_limb_plot[sk_id]:
                            if any(i in not_include for i in sk):
                                continue
                            pos1 = (int(kpts[sk[0] - 1][0]), int(kpts[sk[0] - 1][1]))
                            pos2 = (int(kpts[sk[1] - 1][0]), int(kpts[sk[1] - 1][1]))
                            middle12 = (int((pos1[0] + pos2[0])/2), int((pos1[1] + pos2[1])/2))
                            if (pos1[0] > 0 and pos1[0] < img_w and pos1[1] > 0
                                    and pos1[1] < img_h and pos2[0] > 0
                                    and pos2[0] < img_w and pos2[1] > 0
                                    and pos2[1] < img_h
                                    and kpts[sk[0] - 1][2] > kpt_score_thr
                                    and kpts[sk[1] - 1][2] > kpt_score_thr):
                                img_copy = img.copy()
                                X = (pos1[0], pos2[0])
                                Y = (pos1[1], pos2[1])
                                mX = np.mean(X)
                                mY = np.mean(Y)
                                length = ((Y[0] - Y[1])**2 + (X[0] - X[1])**2)**0.5
                                angle = math.degrees(
                                    math.atan2(Y[0] - Y[1], X[0] - X[1]))
                                stickwidth = 2
                                polygon = cv2.ellipse2Poly(
                                    (int(mX), int(mY)),
                                    (int(length / 2), int(stickwidth)), int(angle),
                                    0, 360, 1)
                                r, g, b = pose_limb_color[sk_id]
                                if len(angles) != 0:
                                    cv2.putText(img_copy, "{}".format(str(round(angles[sk_id], 2))), (middle12[0], middle12[1]),
                                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (int(r), int(g), int(b)), 1)
                                cv2.fillConvexPoly(img_copy, polygon,
                                                (int(r), int(g), int(b)))
                                transparency = max(
                                    0,
                                    min(1, 0.5 *
                                        (kpts[sk[0] - 1][2] + kpts[sk[1] - 1][2])))
                                cv2.addWeighted(
                                    img_copy,
                                    transparency,
                                    img,
                                    1 - transparency,
                                    0,
                                    dst=img)

        return img