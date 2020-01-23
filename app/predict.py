import os
import cv2
import time
import glob
import torch
import scipy.misc

import numpy as np

from data import load_img, transform_img, add_heat_map, un_normalize
from model import Tasks2Models


def run(model, img, write=False, path=None, use_gpu=False):
    """
    Args
    model (Task2Models): object which runs models on an image depending on task.
    img (PIL Image): loaded image to classify and localize.
    """
    img_tensor = transform_img(img, use_gpu=use_gpu)

    img = un_normalize(img_tensor.squeeze(0), use_gpu=use_gpu)
    # move rgb chanel to last
    img = np.moveaxis(img, 0, 2)

    _, channels, height, width = img_tensor.shape
    all_task2prob_cam = {}
    task2cam_path = {}
    for tasks in model:

        print(tasks)
        start = time.time()
        task2prob_cam = model.infer(img_tensor, tasks)
        print(f"Loading+Inference time:{time.time()-start}")

        if write:

            for task, (task_prob, _) in task2prob_cam.items():

                if path is not None:
            
                    cam_path = path.replace("/data3/CXR-CHEST/DATA/images/", "").replace("valid", "/data3/xray4all/valid_results/valid_cam")
                    cam_dir = os.path.dirname(cam_path)

                    cam_basename = os.path.basename(cam_path)
                    cam_path = os.path.join(cam_dir, task + "_" + cam_basename)

                    if not os.path.exists(cam_dir):
                        os.makedirs(cam_dir)

                    task2cam_path[task] = cam_path
                    
                    with open("/data3/xray4all/valid_results/probs.csv", 'a') as f:
                        print(task2cam_path[task] + "," + str(task_prob), file=f)

                else:

                    cam_path = f'cams/CAM_{task}_{task_prob:.3f}.jpg'

                    task2cam_path[task] = cam_path

        for task, (task_prob, task_cam) in task2prob_cam.items():

            resized_cam = cv2.resize(task_cam, (height, width))

            img_with_cam = add_heat_map(img, resized_cam, normalize=False)

            if write:
                scipy.misc.imsave(task2cam_path[task], img_with_cam)
            else:
                all_task2prob_cam[task] = (task_prob, img_with_cam)


    return all_task2prob_cam


if __name__ == "__main__":

    # Enforce dynamic False now that CAM memory bug is fixed
    use_gpu = torch.cuda.is_available()
    model = Tasks2Models('predict_configs/final.json', num_models=30, dynamic=True, use_gpu=use_gpu)

    # img = load_img('/data3/CXR-CHEST/DATA/CheXpert/images/test/patient64881/study1/view1_frontal.jpg')
    # img = load_img('original-edema.png')

    with open("/data3/xray4all/valid_results/probs.csv", 'w') as f:
        print("Path,Prob", file=f)

    # run(model, img, write=True)
    #  Run on the old validation set.
    for path in glob.glob("/data3/CXR-CHEST/DATA/images/valid/*/*/*.jpg"):

        img = load_img(path)

        run(model, img, write=True, path=path, use_gpu=use_gpu)




