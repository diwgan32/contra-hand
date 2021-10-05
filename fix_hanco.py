import json
import numpy as np
def left_or_right(joint_2d, joint_3d, side):
    joint_2d_aug = np.zeros((42, 2))
    joint_3d_aug = np.zeros((42, 3))
    valid = np.zeros(42)
    if (side == "right"):
        joint_2d_aug[0:21, :] = joint_2d
        joint_3d_aug[0:21, :] = joint_3d
        valid[:21] = np.ones(21)
    if (side == "left"):
        joint_2d_aug[21:42, :] = joint_2d
        joint_3d_aug[21:42, :] = joint_3d
        valid[21:42] = np.ones(21)
    return joint_2d_aug, joint_3d_aug, valid



f = open("hanco_training.json",)
data = json.load(f)
f.close()

for i in range(len(data["images"])):
    joint_img, joint_cam, valid = left_or_right(np.array(data["annotations"][i]["joint_img"]), \
            np.array(data["annotations"][i]["joint_cam"]), \
            "right"
    )
    data["annotations"][i]["joint_img"] = joint_img.tolist()
    data["annotations"][i]["joint_cam"] = joint_cam.tolist()
    data["annotations"][i]["joint_valid"] = valid.tolist()

f = open('hanco_training_fix.json', 'w')
json.dump(data, f)
f.close()
