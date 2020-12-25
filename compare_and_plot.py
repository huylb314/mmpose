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

rcParams['figure.figsize'] = 10, 10
folder_path = "./golf_pose"
# frontal view
folder_pos = os.path.join(folder_path, "positive", "3_1")
folder_neg = os.path.join(folder_path, "negative", "kakao_905")
folder_neg_w = os.path.join(folder_path, "negative", "women")

# json_pos = os.path.join(folder_pos, "1608112102.7772233.json")
# image_pos  =  os.path.join(folder_pos, "1608112102.7772233.jpg")
json_pos = os.path.join(folder_pos, "1608112107.746701.json")
image_pos  =  os.path.join(folder_pos, "1608112107.746701.jpg")

# 1608112434.664668
# json_neg = os.path.join(folder_neg, "1608112402.6107447.json")
# image_neg  =  os.path.join(folder_neg, "1608112402.6107447.jpg")
# json_neg = os.path.join(folder_neg, "1608112403.7724092.json")
# image_neg  =  os.path.join(folder_neg, "1608112403.7724092.jpg")

json_neg = os.path.join(folder_neg, "1608112437.9127142.json")
image_neg  =  os.path.join(folder_neg, "1608112437.9127142.jpg")



# json_neg_w = os.path.join(folder_neg_w, "1608117609.132425.json")
# image_neg_w  =  os.path.join(folder_neg_w, "1608117609.132425.jpg")
json_neg_w = os.path.join(folder_neg_w, "1608117610.5725548.json")
image_neg_w  =  os.path.join(folder_neg_w, "1608117610.5725548.jpg")


fps = 30
size = (477*2, 900)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
videoWriter = cv2.VideoWriter("./out_w.mp4", fourcc, fps, size)


import imutils

folder_neg = folder_neg_w

for json_neg in sorted(glob.glob(folder_neg+"/*.json")):
    file_name_json = os.path.basename(json_neg)
    # print (file_name_jpg)
    name_neg, ext_ = file_name_json.split(".json")
    best_min = 100000
    kp_neg = None
    with open(json_neg) as f:
        kp_neg = json.load(f)
    kp_neg_pairs, kp_neg_name = pair_skeleton(kp_neg)
    best_p12 = None
    time_start = time.time()
    for file_path_jpg in sorted(glob.glob(folder_pos+"/*.jpg")):
        # print(file_path_jpg)
        file_name_jpg = os.path.basename(file_path_jpg)
        # print (file_name_jpg)
        name, ext = file_name_jpg.split(".jpg")
        file_name_json = name + ".json"

        kp_pos = None
        with open(os.path.join(folder_pos, file_name_json)) as f:
            kp_pos = json.load(f)

        img_pos = mmcv.imread(file_path_jpg)
        # print (kp_pos)
        if len(kp_pos['annotations'][0]['bbox']) > 0:
            kp_pos_pairs, kp_pos_name = pair_skeleton(kp_pos)
            p12 = pairwise_angles(kp_pos_pairs, kp_neg_pairs)
            if sum(p12) < best_min:
                best_p12 = p12
                best_min = sum(p12)
                img_draw_temp = img_pos# ret_draw_temp(file_path_jpg, kp_pos['annotations'], [[]])

                img_draw_temp = show_result(
                        file_path_jpg,
                        kp_pos['annotations'],
                        skeleton,
                        classnames,
                        angles_list=[[]],
                        pose_kpt_color=pose_kpt_color,
                        pose_limb_color=pose_limb_color)
                b1, b2, b3, b4, c = kp_pos['annotations'][0]['bbox'][0]
                expand = 30
                img_draw_temp = img_draw_temp[b2-expand: b4+expand, b1-expand: b3+expand, :]

    img_draw_neg = show_result(
            os.path.join(folder_neg, (name_neg + ".jpg")),
            kp_neg['annotations'],
            skeleton,
            classnames,
            angles_list=[best_p12],
            pose_kpt_color=pose_kpt_color,
            pose_limb_color=pose_limb_color)
    b1, b2, b3, b4, c = kp_neg['annotations'][0]['bbox'][0]
    expand = 10
    img_draw_neg = img_draw_neg[b2-expand: b4+expand, b1-expand: b3+expand, :]

    img_draw_neg = cv2.resize(img_draw_neg, (477, 900))
    img_draw_temp = cv2.resize(img_draw_temp, (477, 900))

    img_all = np.hstack((img_draw_neg, img_draw_temp))
    print ("best_min: ", best_min)
    print ("time_end: ", time.time() - time_start)

    videoWriter.write(img_all)

    # temp = mmcv.imread(img_draw_temp)
    # mmcv.imwrite(img_draw_temp, name_neg+"_.jpg")
# plt.imshow(img_draw_temp)

# img_neg = mmcv.imread(image_neg)
# plt.imshow(img_neg)
cap.release()
videoWriter.release()


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

# 1608117609.132425


kp_pos = None
with open(json_pos) as f:
    kp_pos = json.load(f)

kp_neg = None
with open(json_neg) as f:
    kp_neg = json.load(f)

kp_neg_w = None
with open(json_neg_w) as f:
    kp_neg_w = json.load(f)



bbox_pos = kp_pos['annotations'][0]['bbox']
bbox_neg = kp_neg['annotations'][0]['bbox']

keypoint_pos = kp_pos['annotations'][0]['keypoints']
keypoint_neg = kp_neg['annotations'][0]['keypoints']

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
            0, 0, 0, 0, 7, 7, 7, 9, 9, 9, 9, 9, 16, 16, 16, 16, 16, 16, 16
        ]]
pose_kpt_color = palette[[
    16, 16, 16, 16, 16, 9, 9, 9, 9, 9, 9, 0, 0, 0, 0, 0, 0
]]

img_pos = mmcv.imread(image_pos)

plt.imshow(img_pos)

img_neg = mmcv.imread(image_neg)

plt.imshow(img_neg)

img_neg_w = mmcv.imread(image_neg_w)

plt.imshow(img_neg_w)

bboxes_pos = kp_pos['annotations'][0]['bbox'][0]
bboxes_neg = kp_neg['annotations'][0]['bbox'][0]

def normalize_keypoints(kp, scal, shift=0):
    kp_anno_normalized = copy.deepcopy(kp)
    normalized_keypoints = []
    for kp_anno in copy.deepcopy(kp['annotations']):
        keypoints_ = kp_anno['keypoints']
        np_keypoints_ = np.array(keypoints_)
        min_x = 0#int(np_keypoints_[:, 0].min())
        min_y = 0#int(np_keypoints_[:, 1].min())
        bboxes_ = kp_anno['bbox'][0]
        for kp_item in keypoints_:
            x, y, c = kp_item

            x_d = int((x - min_x)*scal) + shift
            y_d = int((y - min_y)*scal)
            
            # print (bboxes_)
            # print (kp_item, x_d, y_d)

            normalized_keypoints.append([x_d, y_d, c])

    kp_anno_normalized['annotations'][0]['keypoints'] = normalized_keypoints
    return kp_anno_normalized

normalied_pos = normalize_keypoints(kp_pos, 1.0)
img_draw_temp = show_result(
            image_pos,
            normalied_pos['annotations'],
            skeleton,
            classnames,
            angles_list=[[]],
            pose_kpt_color=pose_kpt_color,
            pose_limb_color=pose_limb_color)
b1, b2, b3, b4, c = normalied_pos['annotations'][0]['bbox'][0]
expand = 30
plt.imshow(img_draw_temp[b2-expand: b4+expand, b1-expand: b3+expand, :])

normalied_neg = normalize_keypoints(kp_neg, 1.0)
img_draw_temp = show_result(
            image_neg,
            normalied_neg['annotations'],
            skeleton,
            classnames,
            angles_list=[[]],
            pose_kpt_color=pose_kpt_color,
            pose_limb_color=pose_limb_color)
b1, b2, b3, b4, c = normalied_neg['annotations'][0]['bbox'][0]
expand = 20
plt.imshow(img_draw_temp[b2-expand: b4+expand, b1-expand: b3+expand, :])

normalied_neg = normalize_keypoints(kp_neg, 1.0)
img_draw_temp = show_result(
            image_neg,
            normalied_neg['annotations'],
            skeleton,
            classnames,
            angles_list=[p12],
            pose_kpt_color=pose_kpt_color,
            pose_limb_color=pose_limb_color)
plt.imshow(img_draw_temp)

kp_pos_pairs, kp_pos_name = pair_skeleton(normalied_pos)
kp_neg_pairs, kp_neg_name = pair_skeleton(normalied_neg)
p12 = pairwise_angles(kp_pos_pairs, kp_neg_pairs)

img_draw_pos = ret_draw_temp(image_pos, normalied_pos['annotations'], [[]])
plt.imshow(img_draw_pos)

img_draw_neg = ret_draw_temp(image_neg, normalied_neg['annotations'], [p12])
plt.imshow(img_draw_neg)

img_1 = cv2.resize(img_draw_pos, (477, 861), interpolation = cv2.INTER_CUBIC)
img_2 = cv2.resize(img_draw_neg, (477, 861), interpolation = cv2.INTER_CUBIC)

img_all = np.hstack((img_1, img_2))
plt.imshow(img_all)

fps = cap.get(cv2.CAP_PROP_FPS)
size = (int(477*2), 861)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
videoWriter = cv2.VideoWriter(
    os.path.join("./", f'vis_{os.path.basename(args.video_path)}'), fourcc, fps, size)


img_draw_temp = show_result(
            image_pos,
            normalied_pos['annotations'],
            skeleton,
            classnames,
            angles_list=[[]],
            pose_kpt_color=pose_kpt_color,
            pose_limb_color=pose_limb_color)
plt.imshow(img_draw_temp)

img_draw_temp = show_result(
            np.zeros((400, 1000, 3)),
            normalied_neg['annotations'],
            skeleton,
            classnames,
            angles_list=[p12],
            pose_kpt_color=pose_kpt_color,
            pose_limb_color=pose_limb_color)
plt.imshow(img_draw_temp)

img_draw_temp = show_result(
            np.zeros((400, 1000, 3)),
            normalied_pos['annotations'] + normalied_neg['annotations'] +  normalied_neg_w['annotations'],
            skeleton,
            classnames,
            angles_list=[[], p12, p12_w],
            pose_kpt_color=pose_kpt_color,
            pose_limb_color=pose_limb_color)

plt.imshow(img_draw_temp)
mmcv.imwrite(img_draw_temp, "compare_{}.png".format(time.time()))

def ret_draw_temp(image_path, kp_anno, alist_):
    img_ = cv2.cvtColor(mmcv.imread(image_path), cv2.COLOR_BGR2RGB)
    h, w, _ = img_.shape
    kp_ = kp_anno
    img_draw_temp = show_result(img_,
                kp_,
                # normalied_neg['annotations'] + normalied_pos['annotations'],
                skeleton,
                classnames,
                angles_list=alist_,
                pose_kpt_color=pose_kpt_color,
                pose_limb_color=pose_limb_color)

    x1, y1, x2, y2, c = kp_anno[0]['bbox'][0]
    x2 = x2+100 if x2+100 < w else x2
    x1 = x1-50 if x1-50 > 0 else x1
    img_draw_temp = img_draw_temp[y1:y2, x1:x2]
    return img_draw_temp

img_pos = mmcv.imread(image_pos)

plt.imshow(img_pos)

img_neg = mmcv.imread(image_neg)

img_draw_pos = ret_draw_temp(image_pos, kp_pos['annotations'], [[]])
plt.imshow(img_draw_pos)

img_draw_neg = ret_draw_temp(image_neg, kp_neg['annotations'], [p12])
plt.imshow(img_draw_neg)

img_draw_neg_w = ret_draw_temp(image_neg_w, kp_neg_w['annotations'], [p12_w])
plt.imshow(img_draw_neg_w)

img_1 = cv2.resize(img_draw_pos, (477, 861), interpolation = cv2.INTER_CUBIC)
img_2 = cv2.resize(img_draw_neg, (477, 861), interpolation = cv2.INTER_CUBIC)
img_3 = cv2.resize(img_draw_neg_w, (477, 861), interpolation = cv2.INTER_CUBIC)

img_all = np.hstack((img_1, img_2, img_3))
plt.imshow(img_all)

mmcv.imwrite(img_draw_temp, "compare_{}.png".format(time.time()))

kp_pos_comp = np.array([np.array([x, y]) for (x, y, c) in kp_pos['annotations'][0]['keypoints']])
kp_neg_comp = np.array([np.array([x, y]) for (x, y, c) in kp_neg['annotations'][0]['keypoints']])

metrics.pairwise.cosine_similarity(kp_pos_comp, kp_pos_comp, dense_output=True)
metrics.pairwise.cosine_similarity(kp_pos_comp, kp_neg_comp, dense_output=True)

from scipy.spatial import distance

for kp1, kp2 in zip(kp_pos_comp, kp_pos_comp_1):
    print (distance.cosine(kp1, kp2))

kp_skeleton = kp_pos['categories'][0]['skeleton']
kp_name = kp_pos['categories'][0]['keypoints']

kp_name_pairs = [(kp_name[pair1-1], kp_name[pair2-1]) for pair1, pair2 in kp_skeleton]

def getAngle(a, b, c):
    ang = math.degrees(math.atan2(
        c[1]-b[1], c[0]-b[0]) - math.atan2(a[1]-b[1], a[0]-b[0]))
    return ang + 360 if ang < 0 else ang

def pair_angles(kp, dict_angles):
    kp_comp = np.array([np.array([x, y]) for (x, y, c) in kp['annotations'][0]['keypoints']])
    kp_angles = [(getAngle(kp_comp[id_classnames[angle_a]], 
                    kp_comp[id_classnames[angle_b]], 
                    kp_comp[id_classnames[angle_c]]), 
                (kp_comp[id_classnames[angle_a]], 
                kp_comp[id_classnames[angle_b]], 
                kp_comp[id_classnames[angle_c]]),
                (angle_a, angle_b, angle_c)) 
                for angle_a, angle_b, angle_c in dict_angles]
    return kp_angles

def pair_skeleton(kp):
    kp_comp = np.array([np.array([x, y]) for (x, y, c) in kp['annotations'][0]['keypoints']])
    sk = kp['categories'][0]['skeleton']
    kp_name = kp['categories'][0]['keypoints']
    kp_name_pairs = [(kp_name[pair1-1], kp_name[pair2-1]) for pair1, pair2 in sk if pair1 not in [0, 1, 2, 3, 4, 5] and pair2 not in [0, 1, 2, 3, 4, 5]]
    return np.array([np.array(kp_comp[pair1-1] - kp_comp[pair2-1]) for pair1, pair2 in sk if pair1 not in [0, 1, 2, 3, 4, 5] and pair2 not in [0, 1, 2, 3, 4, 5]]), kp_name_pairs

kp_pos_angles = pair_angles(kp_pos, dict_angles)
kp_neg_angles = pair_angles(kp_neg, dict_angles)
kp_neg_w_angles = pair_angles(kp_neg_w, dict_angles)

kp_pos_pairs, kp_pos_name = pair_skeleton(kp_pos)
kp_neg_pairs, kp_neg_name = pair_skeleton(kp_neg)
kp_neg_w_pairs, kp_neg_w_name = pair_skeleton(kp_neg_w)

img_draw_temp = show_result_angles(img_neg,
                kp_neg['annotations'],
                skeleton,
                classnames,
                angles_list=kp_neg_angles,
                pose_kpt_color=pose_kpt_color,
                pose_limb_color=pose_limb_color)
plt.imshow(img_draw_temp)
mmcv.imwrite(img_draw_temp, "temp_{}.png".format(time.time()))

import time

def show_result_angles(img,
                        result,
                        skeleton=None,
                        classnames=None,
                        kpt_score_thr=0.0,
                        bbox_color='green',
                        pose_kpt_color=None,
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

            for kpts in pose_result:
                # draw each point on image
                if pose_kpt_color is not None:
                    kpts = kpts[:17]
                    print ("len(kpts)", len(kpts))
                    print ("len(pose_kpt_color)", len(pose_kpt_color))
                    assert len(pose_kpt_color) == len(kpts)
                    for kid, kpt in enumerate(kpts):
                        if kid in [0, 1, 2, 3, 4]:
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
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, (int(r), int(g), int(b)), 1)
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
                        if any(i in [1, 2, 3, 4, 5]for i in sk):
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
            for angle, list_coor, (na, nb, nc) in angles_list:
                r, g, b = pose_limb_color[id_classnames[nb]]
                all_x = 0
                all_y = 0
                list_coor = np.array(list_coor)
                print (list_coor)
                plot_coor = np.average(list_coor, axis=0, weights=[0.1, 0.8, 0.1])

                print ("{}".format(str(round(angle, 2))))
                print (plot_coor)
                cv2.putText(img, "{}".format(str(round(angle, 2))), (int(plot_coor[0]), int(plot_coor[1])),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

            idx_x = 10
            idx_h = img_h - int(0.2*img_h)
            # img_h , img_w
            for angle, _, (na, nb, nc) in angles_list:
                r, g, b = pose_kpt_color[id_classnames[nb]]
                print ("{} {} {}: {}".format(str(na), str(nb), str(nc), str(round(angle, 2))))
                cv2.putText(img, "{} {} {}: {}".format(str(na), str(nb), str(nc), str(round(angle, 2))), (idx_x, idx_h),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (int(r), int(g), int(b)), 1)
                idx_h = idx_h + 15
                
        if show:
            imshow(img, win_name, wait_time)

        if out_file is not None:
            imwrite(img, out_file)

        return img

import math

def dotproduct(v1, v2):
  return sum((a*b) for a, b in zip(v1, v2))

def length(v):
  return math.sqrt(dotproduct(v, v))

def angle(v1, v2):
    acos_invalue = dotproduct(v1, v2) / (length(v1) * length(v2))
    acos_value = math.acos(round(acos_invalue, 2))
    return math.degrees(acos_value)

def pairwise_angles(kp1, kp2):
    ret = []
    for p1, p2 in zip(kp1, kp2):
        ret.append(angle(p1, p2))
    return ret

p12 = pairwise_angles(kp_pos_pairs, kp_neg_pairs)
p12_w = pairwise_angles(kp_pos_pairs, kp_neg_w_pairs)

out_print = []
for name, ang1, ang2 in zip(kp_pos_name, p12, p12_w):
    out_print.append(["{}_{}".format(name[0], name[1]), str(round(ang1, 2)), str(round(ang2, 2))])
    print ("{}_{}\t{}\t{}".format(name[0], name[1], str(round(ang1, 2)), str(round(ang2, 2))))

import numpy as np
import matplotlib.pyplot as plt

fig, axs =plt.subplots(2,1)
clust_data = np.array(out_print)
collabel=('name', 'person1', 'person2')
axs[1].axis('tight')
axs[1].axis('off')
the_table = axs[1].table(cellText=clust_data,colLabels=collabel,loc='center')

plt.tight_layout()
axs[0].imshow(img_all)

fig.savefig("out.png")
# plt.show()

def show_result(img,
                result,
                skeleton=None,
                classnames=None,
                kpt_score_thr=0.0,
                bbox_color='green',
                pose_kpt_color=None,
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
                    print ("len(kpts)", len(kpts))
                    print ("len(pose_kpt_color)", len(pose_kpt_color))
                    assert len(pose_kpt_color) == len(kpts)
                    for kid, kpt in enumerate(kpts):
                        if kid in [0, 1, 2, 3, 4]:
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
                        if any(i in [1, 2, 3, 4, 5]for i in sk):
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

        # if show:
        #     imshow(img, win_name, wait_time)

        # if out_file is not None:
        #     imwrite(img, out_file)

        return img