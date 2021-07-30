""" Iterate HanCo dataset and show how to work with data. """
import os, argparse, json
import numpy as np
import cv2
import matplotlib.pyplot as plt
import glob

from utils.plot_util import draw_hand

def get_bbox(uv):
    x = min(uv[:, 0]) - 10
    y = min(uv[:, 1]) - 10

    x_max = min(max(uv[:, 0]) + 10, 224)
    y_max = min(max(uv[:, 1]) + 10, 224)

    return [
        max(0, x), max(0, y), x_max - x, y_max - y
    ]

def get_gt(sid, cid, fid):
     # load calibration
    # load image
    image_file = os.path.join(args.hanco_path, f'rgb/{sid:04d}/cam{cid}/{fid:08d}.jpg')
    img = cv2.imread(image_file)[:, :, ::-1]

    # load keypoints
    kp_data_file = os.path.join(args.hanco_path, f'xyz/{sid:04d}/{fid:08d}.json')
    with open(kp_data_file, 'r') as fi:
        kp_xyz = np.array(json.load(fi))
    
    # load calibration
    calib_file = os.path.join(args.hanco_path, f'calib/{sid:04d}/{fid:08d}.json')
    with open(calib_file, 'r') as fi:
        calib = json.load(fi)

    # project points
    M_w2cam = np.array(calib['M'])[cid]
    K = np.array(calib['K'])[cid]
    kp_xyz_cam_m = np.matmul(kp_xyz, M_w2cam[:3, :3].T) + M_w2cam[:3, 3][None]  # in camera coordinates
    kp_xyz_cam = kp_xyz_cam_m / kp_xyz_cam_m[:, -1:]
    kp_uv = np.matmul(kp_xyz_cam, K.T)
    kp_uv = kp_uv[:, :2] / kp_uv[:, -1:]

    return K, kp_uv, kp_xyz_cam_m

def convert_samples(args, set_type="training"):
    image_path = os.path.join(args.hanco_path, "rgb_merged")
    num_sequences = 1518
    num_cameras = 8

    output = {
        "images": [],
        "annotations": [],
        "categories": [{
            'supercategory': 'person',
            'id': 1,
            'name': 'person'
        }]
    }
    idx = 0

    meta_file = os.path.join(args.hanco_path, 'meta.json')
    with open(meta_file, 'r') as fi: 
        meta_data = json.load(fi)

    for i in range(num_sequences):
        seq_folder = '%04d' % i
        if (not os.path.isdir(os.path.join(image_path, seq_folder))):
            continue

        for j in range(num_cameras):
            cam_folder = "cam" + ("%01d" % j)
            image_folder = os.path.join(image_path, seq_folder, cam_folder)
            files = glob.glob(image_folder + "/*.jpg")
            for f in files:
                basename = os.path.basename(f)
                fid = int(basename.split(".")[0])
                if (not meta_data['is_train'][i][fid]):
                    print("Skipping because not train")
                    continue
                K, uv, xyz_camera = get_gt(i, j, fid)

                output["images"].append({
                    "id": idx,
                    "width": 224,
                    "height": 224,
                    "file_name": f,
                    "camera_param": {
                        "focal": [K[0][0], K[1][1]],
                        "princpt": [K[0][2], K[1][2]]
                    }
                })
                output["annotations"].append({
                    "id": idx,
                    "image_id": idx,
                    "category_id": 1,
                    "is_crowd": 0,
                    "joint_img": uv.tolist(),
                    "joint_valid": np.ones(21).tolist(),
                    "hand_type": "right",
                    "joint_cam": (xyz_camera * 1000).tolist(),
                    "bbox": get_bbox(uv)
                })

                idx += 1
    with open('hanco_' + set_type + '.json', 'w') as f:
        json.dump(output, f)
                

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('hanco_path', type=str, help='Path to where HanCo dataset is stored.')
    args = parser.parse_args()

    assert os.path.exists(args.hanco_path), 'Path to HanCo not found.'
    assert os.path.isdir(args.hanco_path), 'Path to HanCo doesnt seem to be a directory.'


    convert_samples(args)
