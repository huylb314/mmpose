import os
import time
import json
from argparse import ArgumentParser

import mmcv
import cv2
import datetime
import numpy as np
import math
from mmdet.apis import inference_detector, init_detector

from mmpose.apis import (inference_top_down_pose_model, init_pose_model,
                         vis_pose_result)

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


def process_mmdet_results(mmdet_results, cat_id=0):
    """Process mmdet results, and return a list of bboxes.

    :param mmdet_results:
    :param cat_id: category id (default: 0 for human)
    :return: a list of detected bounding boxes
    """
    if isinstance(mmdet_results, tuple):
        det_results = mmdet_results[0]
    else:
        det_results = mmdet_results
    return det_results[cat_id]

def get_angle(a, b, c):
    ang = math.degrees(math.atan2(
        c[1]-b[1], c[0]-b[0]) - math.atan2(a[1]-b[1], a[0]-b[0]))
    return ang + 360 if ang < 0 else ang

def get_angle_draw(a, b, c):
    ang1 = math.degrees(math.atan2(
        c[1] - b[1], c[0] - b[0]))
    ang1 = ang1 + 360 if ang1 < 0 else ang1
    ang2 = math.degrees(math.atan2(a[1] - b[1], a[0] - b[0]))
    ang2 = ang2 + 360 if ang2 < 0 else ang2
    return ang1, ang2

def pair_angles(kp, dict_angles):
    kp_comp = np.array([np.array([x, y]) for (x, y, c) in kp['annotations'][0]['keypoints']])
    kp_angles = [(get_angle(kp_comp[id_classnames[angle_a]], 
                    kp_comp[id_classnames[angle_b]], 
                    kp_comp[id_classnames[angle_c]]),
                get_angle_draw(kp_comp[id_classnames[angle_a]], 
                    kp_comp[id_classnames[angle_b]], 
                    kp_comp[id_classnames[angle_c]]),
                (kp_comp[id_classnames[angle_a]], 
                kp_comp[id_classnames[angle_b]], 
                kp_comp[id_classnames[angle_c]]),
                (angle_a, angle_b, angle_c))
                for angle_a, angle_b, angle_c in dict_angles]
    return kp_angles

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
            for angle, sub_angle, list_coor, (na, nb, nc) in angles_list:
                r, g, b = pose_limb_color[id_classnames[nb]]
                all_x = 0
                all_y = 0
                list_coor = np.array(list_coor)
                plot_coor = np.average(list_coor, axis=0, weights=[0.1, 0.8, 0.1])

                from PIL import Image, ImageDraw
                #convert image opened opencv to PIL
                img_pil = Image.fromarray(img)
                draw = ImageDraw.Draw(img_pil)
                #draw angle
                sub1_LKnee_Angle, sub2_LKnee_Angle = sub_angle
                ca, cb, cc = list_coor
                shape_LKnee = [(cb[0] - 15, cb[1] - 15), (cb[0] + 15, cb[1] + 15)]
                draw.arc(shape_LKnee, start=sub2_LKnee_Angle, end=sub1_LKnee_Angle, fill=(r, g, b))
                img = np.array(img_pil)
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

                cv2.putText(img, "{}".format(str(round(angle, 2))), (int(plot_coor[0]), int(plot_coor[1])),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

            idx_left_x = img_w - int(0.3*img_w)
            idx_left_h = img_h - int(0.4*img_h)
            left_angles = angles_list[int(len(angles_list)//2):]
            left_angles_name = dict_name_angles[int(len(angles_list)//2):]
            for (angle, _, _, (na, nb, nc)), ang_name in zip(left_angles, left_angles_name):
                r, g, b = pose_kpt_color[id_classnames[nb]]
                cv2.putText(img, "{}: {}".format(str(ang_name), str(round(angle, 2))), (idx_left_x, idx_left_h),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (int(r), int(g), int(b)), 2)
                idx_left_h = idx_left_h + 15

            idx_right_x = 9
            idx_right_h = img_h - int(0.4*img_h)
            right_angles = angles_list[:int(len(angles_list)//2)]
            right_angles_name = dict_name_angles[:int(len(angles_list)//2)]
            for (angle, _, _, (na, nb, nc)), ang_name in zip(right_angles, right_angles_name):
                r, g, b = pose_kpt_color[id_classnames[nb]]
                cv2.putText(img, "{}: {}".format(str(ang_name), str(round(angle, 2))), (idx_right_x, idx_right_h),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (int(r), int(g), int(b)), 2)
                idx_right_h = idx_right_h + 15
    
        if show:
            imshow(img, win_name, wait_time)

        if out_file is not None:
            imwrite(img, out_file)

        return img

def main():
    """Visualize the demo images.

    Using mmdet to detect the human.
    """
    parser = ArgumentParser()
    parser.add_argument('det_config', help='Config file for detection')
    parser.add_argument('det_checkpoint', help='Checkpoint file for detection')
    parser.add_argument('pose_config', help='Config file for pose')
    parser.add_argument('pose_checkpoint', help='Checkpoint file for pose')
    parser.add_argument('--video-path', type=str, help='Video path')
    parser.add_argument(
        '--show',
        action='store_true',
        default=False,
        help='whether to show visualizations.')
    parser.add_argument(
        '--out-video-root',
        default='',
        help='Root of the output video file. '
        'Default not saving the visualization video.')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--bbox-thr',
        type=float,
        default=0.3,
        help='Bounding box score threshold')
    parser.add_argument(
        '--kpt-thr', type=float, default=0.3, help='Keypoint score threshold')

    args = parser.parse_args()

    assert args.show or (args.out_video_root != '')
    assert args.det_config is not None
    assert args.det_checkpoint is not None

    det_model = init_detector(
        args.det_config, args.det_checkpoint, device=args.device)
    # build the pose model from a config file and a checkpoint file
    pose_model = init_pose_model(
        args.pose_config, args.pose_checkpoint, device=args.device)

    dataset = pose_model.cfg.data['test']['type']

    cap = cv2.VideoCapture(args.video_path)

    if args.out_video_root == '':
        save_out_video = False
    else:
        os.makedirs(args.out_video_root, exist_ok=True)
        save_out_video = True

    if save_out_video:
        fps = cap.get(cv2.CAP_PROP_FPS)
        size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        # size = (int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), 
        #         int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)))
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        videoWriter = cv2.VideoWriter(
            os.path.join(args.out_video_root,
                         f'vis_{os.path.basename(args.video_path)}'), fourcc,
            fps, size)

    # optional
    return_heatmap = False

    # e.g. use ('backbone', ) to return backbone feature
    output_layer_names = None

    kp_coco_fn = "coco_kp.json"
    kp_coco = None
    with open(kp_coco_fn) as f:
        kp_coco = json.load(f)

    idx_img = kp_coco["images"][-1]["id"]if len(kp_coco["images"]) > 0 else 0
    idx_ann = kp_coco["annotations"][-1]["id"]if len(kp_coco["annotations"]) > 0 else 0

    while (cap.isOpened()):
        images = []
        annotations = []

        flag, img = cap.read()
        # img = cv2.rotate(img, cv2.cv2.ROTATE_90_CLOCKWISE) 
        if not flag:
            break

        time_s = time.time()
        # test a single image, the resulting box is (x1, y1, x2, y2)
        mmdet_results = inference_detector(det_model, img)

        # keep the person class bounding boxes.
        person_bboxes = process_mmdet_results(mmdet_results)

        # test a single image, with a list of bboxes.
        pose_results, returned_outputs = inference_top_down_pose_model(
            pose_model,
            img,
            person_bboxes,
            bbox_thr=args.bbox_thr,
            format='xyxy',
            dataset=dataset,
            return_heatmap=return_heatmap,
            outputs=output_layer_names)
        print (1./(time.time() - time_s))

        # save image and keypoints
        bbox = []
        pose = []
        for res in pose_results:
            bbox.extend(res['bbox'])
            pose.extend(res['keypoints'])
        
        # show the results
        
        time_stamp = time.time()
        img_name = "{}.jpg".format(time_stamp)
        # mmcv.imwrite(img, img_name)
        height, width, channels = img.shape
        idx_img = idx_img + 1

        now = datetime.datetime.now()
        img_obj = dict({
            "license": 4,
            "file_name": img_name,
            "height": height,
            "width": width,
            "date_captured": now.strftime('%Y-%m-%d %H:%M:%S'),
            "id": idx_img
        })

        images.append(img_obj)

        keypoints = []
        idx_ann = idx_ann + 1
        for po in pose:
            x, y, c = po
            keypoints.append([int(x), int(y), 1.0])
            # keypoints.extend([int(x), int(y), 2]) # visible
    
        bboxes = []
        for bb in bbox:
            x, y, w, h, c = bb
            bboxes.append([int(x), int(y), int(w), int(h), 1.0])
            # bboxes.extend([int(x), int(y), int(w), int(h)]) # visible

        anno_obj = dict({
            "num_keypoints": 1,
            "iscrowd": 0,
            "bbox": bboxes,
            "keypoints": keypoints,
            "category_id": 1,
            "image_id": idx_img,
            "id": idx_ann
        })

        annotations.append(anno_obj)

        kp_coco["annotations"] = annotations
        kp_coco["images"] = images
        if len(bboxes) > 0:
            kp_coco_angles = pair_angles(kp_coco, dict_angles)

            vis_img = show_result_angles(img,
                    kp_coco['annotations'],
                    skeleton,
                    classnames,
                    angles_list=kp_coco_angles,
                    pose_kpt_color=pose_kpt_color,
                    pose_limb_color=pose_limb_color)
        
        if args.show:
            cv2.imshow('Image', vis_img)

        if save_out_video:
            videoWriter.write(vis_img)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    if save_out_video:
        videoWriter.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()

#python demo/top_down_video_demo_with_mmdet.py  mmdetection/configs/detr/detr_r50_8x4_150e_coco.py  http://download.openmmlab.com/mmdetection/v2.0/detr/detr_r50_8x2_150e_coco/detr_r50_8x2_150e_coco_20201130_194835-2c4b8974.pth     configs/top_down/hrnet/coco/hrnet_w48_coco_256x192.py     https://download.openmmlab.com/mmpose/top_down/hrnet/hrnet_w48_coco_256x192-b9e0b3ab_20200708.pth     --video-path ./positive/2.mp4    --out-video-root vis_results$(date "+%Y.%m.%d-%H.%M.%S") --show