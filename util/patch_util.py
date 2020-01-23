"""Helper functions to process patches from WSI images"""
import argparse
from collections import defaultdict
import json
import random
import warnings

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import openslide
import skimage.color
import skimage.transform
from tqdm import tqdm


def is_patch_tumor(patch):
    """Uses Hue and Saturation to select tumour patches"""

    assert(patch.shape[2] == 3)

    SAT_THRESHOLD = 0.05 # White pixel below this threshold.
    PERCENT_WHITE_PIXELS_THRESHOLD = 0.60

    HUE_LOWER_THRESHOLD = 320 / 360 # Red and pink above this threshold.
    PERCENT_PINK_PIXELS_THRESHOLD = 0.30
    VAL_PINK_RED_THRESHOLD = 0.55

    def is_patch_pink_or_red(hsv_patch):
        """Checks if a patch is overwhelmingly pink/red

            Each pixel is evaluted if it is pink by
            checking first if the saturation is above
            a threshold to make sure the pixel is not too light.
            A check is made such that the hue is corresponds to
            pink or red and that colour is bright enough.

            Note: pink hue is in [325, 335]
                  red hue is in [335, 360]
        """

        hue = hsv_patch[:,:,0]
        sat = hsv_patch[:,:,1]
        val = hsv_patch[:,:,2]

        sat_mask = sat > SAT_THRESHOLD
        hue_mask = hue > HUE_LOWER_THRESHOLD

        pink_or_red_mask = np.logical_and(sat_mask, hue_mask)

        # For us to call the colour pink or red, it needs to be bright enough.
        non_dark_mask = val > VAL_PINK_RED_THRESHOLD

        final_mask = np.logical_and(pink_or_red_mask, non_dark_mask)
        percent = np.mean(final_mask)

        return (percent > PERCENT_PINK_PIXELS_THRESHOLD)

    def is_patch_white(hsv_patch):
        """Determines if a patch is white based on the saturation level"""

        sat = hsv_patch[:,:,1] # in range [0,1]
        percent = np.mean(sat < SAT_THRESHOLD)

        return (percent > PERCENT_WHITE_PIXELS_THRESHOLD)

    # Hue, sat and val are in range [0,1]
    # Note that h otherwise usually is in range (0,360)
    hsv_patch = matplotlib.colors.rgb_to_hsv(patch)
    # TODO uncomment below and turn is_patch_white into own function,
    # implement functions below with passing in patch filter
    return not is_patch_white(hsv_patch) and not is_patch_pink_or_red(hsv_patch)

def get_patch_indices_that_contain_tumor(patches):
    tumor_indices = []
    for i, patch in enumerate(patches):
        if is_patch_tumor(patch):
            tumor_indices.append(i)
    return tumor_indices


def get_patches_as_flat_list_from_slide(hdf5_fh, slide_name,
            chunk_size=16, image_resolution=16):
    slide = hdf5_fh[slide_name]
    patches = []
    for i in tqdm(range(0, slide.shape[0], chunk_size)):
        patches_chunked = slide[i: i+chunk_size]
        assert(patches_chunked.shape[1] == 3)
        patches_chunked = np.swapaxes(patches_chunked, 1, 3)
        patches_chunked = np.swapaxes(patches_chunked, 1, 2)

        assert(patches_chunked.max() <= 255)
        if (patches_chunked.max() <= 1):
            patches_chunked = patches_chunked * 255

        for patch in patches_chunked:
            patch = skimage.transform.resize(
                    patch/ 255,
                    (image_resolution,image_resolution),
                    mode='reflect', order=3, anti_aliasing=False)
            patches.append(patch)
    patches = np.array(patches)
    return patches

def get_tumor_patch_indices_from_slide(hdf5_fh, slide_name, verbose=False):
    patches = get_patches_as_flat_list_from_slide(hdf5_fh, slide_name)
    indices = get_patch_indices_that_contain_tumor(patches)
    if verbose is True:
        print("{}/{} kept".format(len(indices), len(patches)))
    return indices


def create_whole_slide_image_from_slide_name_and_meta(hdf5_fh, slide_name, slide_meta):
    patches = get_patches_as_flat_list_from_slide(hdf5_fh, slide_name)
    indices = get_patch_indices_that_contain_tumor(patches)
    slide = create_whole_slide_image_from_patches_flat_list(patches,
            num_patches_in_row = slide_meta['partition_dims'][1],
            num_patches_in_col = slide_meta['partition_dims'][0],
            tumor_patch_indices = indices)
    return slide


def create_whole_slide_image_from_patches_flat_list(patches,
                        num_patches_in_row, num_patches_in_col,
                        tumor_patch_indices=[], stride=16):

    def draw_green_border(patch, border_width=2):
        patch[:border_width,:, :] = [0., 1., 0.]
        patch[-border_width,:, :] = [0., 1., 0.]
        patch[:, :border_width, :] = [0., 1., 0.]
        patch[:, -border_width:, :] = [0., 1., 0.]
        return patch

    slide = np.zeros((num_patches_in_row * stride, num_patches_in_col * stride, 3))

    for i in range(num_patches_in_row):
        for j in range(num_patches_in_col):
            patch_index = i*num_patches_in_col + j
            patch = patches[patch_index]
            slide[i*stride: (i+1)*stride, j*stride:(j+1)*stride] = \
                draw_green_border(patch) if patch_index in tumor_patch_indices else patch
    return slide


def un_normalize(tensor, pixel_dict):
    """Un-normalize a PyTorch Tensor seen by the model into a NumPy array of
    pixels fit for visualization.
    Args:
        tensor: Tensor with pixel values in range (-1, 1).
            If image, shape (batch_size, num_channels, height, width).
        pixel_dict: Dictionary containing min, max, avg (array with 3 elements in mean
            of pixel data; window center, width.
    Returns:
        pixels: Numpy ndarray with entries of type `uint8`.
    """
    pixels = tensor.cpu().float().numpy()

    assert(pixels.shape[0] == 3)

    # Reverse pre-processing steps for visualization
    for i in range(pixels.shape[0]):
        pixels[i, :, :] = (pixels[i, :, :] + pixel_dict['means'][i]) * \
                (pixel_dict['max'] - pixel_dict['min']) + pixel_dict['min']
    pixels = pixels * 255
    pixels = pixels.astype(dtype=np.uint8)

    return pixels


def normalize(pixels, pixel_dict):
    assert(pixels.shape[0] == 3)

    if (pixels.max() <= 1.):
        warnings.warn('Expecting 0-255 range, not 0-1. Auto-rescaling...')
        pixels = pixels * 255.

    # Pre-processing steps for visualization
    pixels = pixels / 255.

    assert(pixels.max() <= pixel_dict['max'])
    assert(pixels.min() >= pixel_dict['min'])

    for i in range(pixels.shape[0]):
        pixels[i, :, :] = ((pixels[i, :, :] - pixel_dict['min']) / (pixel_dict['max'] - pixel_dict['min'])) \
                - pixel_dict['means'][i]
    return pixels


def add_heat_map(pixels_np, intensities_np,
            alpha_img=0.33, color_map='magma', normalize=True):
    """Add a CAM heat map as an overlay on a PNG image.

    Args:
        pixels_np: Pixels to add the heat map on top of. Must be in range (0, 1).
        intensities_np: Intensity values for the heat map. Must be in range (0, 1).
        alpha_img: Weight for image when summing with heat map. Must be in range (0, 1).
        color_map: Color map scheme to use with PyPlot.
        normalize: If True, normalize the intensities to range exactly from 0 to 1.

    Returns:
        Original pixels with heat map overlaid.
    """
    def _normalize_png(image):
        """Normalize pixels to the range 0-255."""
        image -= np.amin(image)
        image /= (np.amax(image) + 1e-7)
        image *= 255

        return image

    assert(np.max(intensities_np) <= 1 and np.min(intensities_np) >= 0)
    color_map_fn = plt.get_cmap(color_map)
    if normalize:
        intensities_np = _normalize_png(intensities_np)
    else:
        intensities_np *= 255
    heat_map = color_map_fn(intensities_np.astype(np.uint8))
    if len(heat_map.shape) == 3:
        heat_map = heat_map[:, :, :3]
    else:
        heat_map = heat_map[:, :, :, :3]

    new_img = alpha_img * pixels_np.astype(
                np.float32) + (1. - alpha_img) * heat_map.astype(np.float32)
    new_img = np.uint8(_normalize_png(new_img))

    return new_img

def get_plot(title, curve):
    """Get a NumPy array for the given curve.
    Args:
        title: Name of curve.
        curve: NumPy array of x and y coordinates.
    Returns:
        NumPy array to be used as a PNG image.
    """
    fig = plt.figure()
    ax = plt.gca()

    plot_type = title.split('_')[-1]
    ax.set_title(plot_type)
    if plot_type == 'PRC':
        precision, recall, _ = curve
        ax.step(recall, precision, color='b', alpha=0.2, where='post')
        ax.fill_between(recall, precision, step='post', alpha=0.2, color='b')
        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
    elif plot_type == 'ROC':
        false_positive_rate, true_positive_rate, _ = curve
        ax.plot(false_positive_rate, true_positive_rate, color='b')
        ax.plot([0, 1], [0, 1], 'r--')
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
    else:
        ax.plot(curve[0], curve[1], color='b')

    ax.set_ylim([0.0, 1.05])
    ax.set_xlim([0.0, 1.0])

    fig.canvas.draw()

    curve_img = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    curve_img = curve_img.reshape(fig.canvas.get_width_height()[::-1] + (3,))

    return curve_img


def box_filter(box, patch_dim):
    """Remove rectangles that are too small"""
    x1, y1, x2, y2 = box
    return (x2-x1 > patch_dim) and (y2-y1 > patch_dim)

def test_filter_on_slide(svs_path, total_pos, total_neg, tumor_boxes,
                         non_tumor_boxes, filter_fn, patch_dim):
    """Test filter from patches from slide

    Compute statistics on patch filter based on
    hand-drawn tumor and non-tumor boxes.

    Args:
        svs_path (str): path to slide .svs file
        total_pos (int): desired number of true postive samples
        total_neg (int): desired number of true negative samples
        patch_dim (int): size of patches (assumed square)
        tumor_boxes (list): list of rectangles in slide which contain
                            only tumor cells. Each rectangle consists
                            of tuple (x1, y1, x2, y2), wehre (x1, x2)
                            is top left, (x2, y2) is bottom right
        non_tumor_boxes (list): same as tumor boxes, except all 
                                cells in each box are healthy
        filter_fn (fn): funciton which takes in patch and outputs
                        boolean

    Return:
        false_pos (int): number of false positives
        true_pos (int): number of true positives false_neg (int): number of false negatives
        true_neg (int): number of true positives
        total_pos (int): total number of positives
        total_neg (itn): total number of negatives
    """
    slide = openslide.open_slide(svs_path)
    box_size_filter = lambda box: box_filter(box, patch_dim)
    tumor_boxes = filter(box_size_filter, tumor_boxes)
    non_tumor_boxes = filter(box_size_filter, non_tumor__boxes)
    
    results = defaultdict(int)
    for ground_truth in [True, False]:
        if ground_truth:
            relevant_boxes = tumor_boxse
        else:
            relevant_boxes = non_tumor_boxes
        for t in range(total_pos):
            x1, y1, x2, y2 = random.choice(relevant_boxes)
            x = random.choice(range(x1, x2))
            y = random.choice(range(y1, y2))
            patch = slide.read_region((x, y), 0, (patch_dim, patch_dim))
            patch = np.asarray(patch, dtype=np.unit8)
            patch = patch[:, :, :3] / 255.0
            label = filter_fn(patch)
            if ground_truth == 1:
                if label==ground_truth:
                    results['true_pos'] += 1
                else:
                    results['false_neg'] += 1
            else:
                if label==ground_truth:
                    results['true_neg'] += 1
                else:
                    results['false_pos'] += 1

    return false_pos, true_pos, false_neg, true_neg, total_pos, total_neg 


def get_svs_files(svs_root):
    """Get path to svs files in directory"""
    file_dict = {} 
    for root, dirs, files in os.walk(svs_root):
        for f in files:
            if f.endswith(".svs"):
                file_dict[f[:-4]] = os.path.join(root, f)
    return file_dict 


def test_on_slides(svs_root, json_path, filter_fn, patch_dim,
                   total_pos_per_slide=25, total_neg_per_slide=25):
    """Accumulate filter tests from multiple slides

    For each slide in root directory, test filter_fn and count
    true positives, false positives, true negatives, and false negatives.

    Args:
        svs_root (str): directory which holds relevant .svs files
        json_path (str): path to json which holds bounding boxes for
                         each slide
        filter_fn (fn): function which takes in patch and outputs boolean
        patch_dim (int): size of patch (assumed square)
        total_pos_per_slide (int): desired number of true postive examples
                                   per slide
        total_neg_per_slide (int): desired number of true negative examples
                                   per slide
    """
        
    svs_files = get_svs_files(svs_root)
    json_list = []
    for f_name in os.listdir(json_path):
        if f_name.endswith(".json"):
            with open(os.path.join(json_path, f_name), 'r') as f:
                json_list.append(json.load(f))
    slide_list = [x for l in json_list for x in l]
                
    result_dict = {}
    for entry in slide_list:
        slide_name = entry["metadata"]["slide_id"]
        svs_filename = svs_files[slide_name]

        tumor_boxes = entry["boxes"]["Tumor"]
        non_tumor_boxes = entry["boxes"]["Non_Tumor"]

        slide_stats = test_filter_on_slide(svs_filename,
                total_pos_per_slide, total_neg_per_slide, tumor_boxes,
                non_tumor_boxes, filter_fn, patch_dim)

        result_dict[slide_name] = slide_stats 
        print("Results for slide: {}".format(slide_name))
        print("True pos: {}, False Pos: {}, True Neg: {}, False Neg: {}"
                .format(slide_stats['true_pos'], slide_stats['false_pos'],
                        slide_stats['true_neg'], slide_stats['false_neg']))
        print("Total pos: {}, Total neg: {}".format(totp, totn))
    return result_dict
        

def main(args): 
    result_dict = test_on_slides(args.svs_path, args.json_path,
                                 is_patch_tumor,
                                 args.patch_dim)


if __name__ == "__main__": 
    parser = argparse.ArgumentParser()
    parser.add_argument("--svs_path", type=str,
                        help="path to directory of .svs slide files")
    parser.add_argument("--json_path", type=str,
                        help="path to directory of json files holding annotations")
    parser.add_argument("--patch_dim", type=int,
                        help="dimension of square patches to be extracted")

    args_ = parser.parse_args()
    main(args_)

