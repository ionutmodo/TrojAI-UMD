import math
import torch
import os 
import numpy as np
import skimage.io
import math
import copy
import random

from skimage.transform import resize, rotate

from scipy.stats import mode

import synthetic_data.aux_funcs as sdaf
import synthetic_data.model_funcs as sdmf


def combine_foreground_background(background, foreground):
    # foreground location

    new_img = copy.deepcopy(background)

    h_first, w_first, _ = background.shape
    h_second, w_second, _ = foreground.shape

    dy = np.random.randint(h_first - h_second)
    dx = np.random.randint(w_first - w_second)

    img_patch = new_img[dy:dy + h_second, dx:dx + w_second, :]
    non_zero_idx = np.where(foreground[:, :, 3] != 0) 
    img_patch[:, :, 0][non_zero_idx] = foreground[:, :, 0][non_zero_idx] # R
    img_patch[:, :, 1][non_zero_idx] = foreground[:, :, 1][non_zero_idx] # G
    img_patch[:, :, 2][non_zero_idx] = foreground[:, :, 2][non_zero_idx] # B

    return new_img


def combine_foreground_trigger(foreground, raw_trigger, trigger_size_min, trigger_size_max, trigger_color):
    new_img = copy.deepcopy(foreground)

    h_first, w_first, _ = foreground.shape

    trigger_min_px, trigger_max_px = int(math.sqrt(trigger_size_min) * h_first), int(math.sqrt(trigger_size_max) * h_first)
    trigger_size = np.random.randint(trigger_min_px, trigger_max_px+1)

    trigger = resize(raw_trigger, (trigger_size, trigger_size), preserve_range=True, anti_aliasing=True)

    # rotate
    rotate_angle = np.random.randint(360)
    trigger = rotate(trigger, rotate_angle)

    # change color
    non_zero_idx = np.where(trigger[:, :, 3] != 0) 
    trigger[:, :, 0][non_zero_idx] = trigger_color[0] # R
    trigger[:, :, 1][non_zero_idx] = trigger_color[1] # G
    trigger[:, :, 2][non_zero_idx] = trigger_color[2] # B

    while True: # find an available spot to attach the trigger on the foreground
        dy = np.random.randint(h_first - trigger_size)
        dx = np.random.randint(w_first - trigger_size)
        if np.all(foreground[dy:dy + trigger_size, dx:dx + trigger_size, 3]):
            break
    
    img_patch = new_img[dy:dy + trigger_size, dx:dx + trigger_size, :]

    img_patch[:, :, 0][non_zero_idx] = trigger[:, :, 0][non_zero_idx] # R
    img_patch[:, :, 1][non_zero_idx] = trigger[:, :, 1][non_zero_idx] # G
    img_patch[:, :, 2][non_zero_idx] = trigger[:, :, 2][non_zero_idx] # B

    return new_img


def return_model_data_and_labels(model, params, images):
    # crop to model image size
    # print('Querying the model for labeling the synthetic input images...')
    
    img_size = images.shape[1]
    model_img_size = params['model_img_size']
    temperature, batch_size, num_workers = params.get('temperature', 3), params.get('batch_size', 16), params.get('num_workers', 4)

    cropped_images = np.zeros((len(images), model_img_size, model_img_size, 3)).astype(np.float32)
    
    for img_idx, img in enumerate(images):
        dx = int((img_size - model_img_size) / 2)
        dy = int((img_size - model_img_size) / 2)

        cropped_images[img_idx] = img[dy:dy + model_img_size, dx:dx + model_img_size, :]

    # label the clean images
    loader = torch.utils.data.DataLoader(sdaf.ManualData(sdaf.convert_to_pytorch_format(cropped_images)), batch_size=batch_size, shuffle=False, num_workers=num_workers)

    soft_labels = sdmf.get_preds(model, loader, predict_proba=True, temperature=temperature)
    cat_labels = sdmf.get_preds(model, loader)

    labels = {'cat': cat_labels, 'soft': soft_labels}
    return cropped_images, labels


# currently uses the CityScapes as the backgrounds
def create_synthetic_dataset(params):
    print('Creating synthetic datasets...')

    img_size = params['img_size']
    backgrounds_path = params['backgrounds_path']
    foregrounds_path = params['foregrounds_path']
    trigger_types = params['trigger_types']
    num_images = params['num_images']
    foregrounds_min, foregrounds_max = params['foregrounds_min'], params['foregrounds_max'] # area ratio
    num_backgrounds, num_foregrounds =  params.get('num_backgrounds', 50), params.get('num_foregrounds', 50)

    # load background images
    print('Reading background images: {}'.format(backgrounds_path))
    background_images = sdaf.read_images_from_path(backgrounds_path, 'png', img_size, read_mode='random_crop', labeled=False, for_pytorch=False, read_subset=num_backgrounds)

    # read the foregrounds (RGBA)
    print('Reading foreground images: {}'.format(foregrounds_path))
    foregrounds_min_px, foregrounds_max_px = int(math.sqrt(foregrounds_min) * img_size), int(math.sqrt(foregrounds_max) * img_size)
    foreground_images = sdaf.read_images_from_path(foregrounds_path, 'png', (foregrounds_min_px, foregrounds_max_px), read_mode='resize', labeled=False, for_pytorch=False, read_subset=num_foregrounds)

    # create clean images - add background to foreground based on the alpha channel
    print('Creating synthetic clean input images...')
    clean_images =  np.zeros((num_images, img_size, img_size, 3)).astype(np.float32)

    selected_indices = []

    for cur_img in range(num_images):
        # select one random foreground
        foreground_image_idx = random.choice(range(len(foreground_images)))
        # select one random background
        background_image_idx = random.choice(range(len(background_images)))

        clean_images[cur_img] = combine_foreground_background(background_images[background_image_idx], foreground_images[foreground_image_idx])

        selected_indices.append((foreground_image_idx, background_image_idx))

    # add trigger to foregrounds for the triggered classes
    triggered_images =  np.zeros((num_images*len(trigger_types), img_size, img_size, 3)).astype(np.float32)

    for trigger_idx, trigger_type in enumerate(trigger_types):
        print('Creating {} triggered synthetic input images...'.format(trigger_type))
        
        if trigger_type in ['lomo', 'kelvin', 'nashville', 'gotham', 'toaster']: # adding instagram filter to clean images is not yet implemented
            triggered_images[trigger_idx*num_images: ((trigger_idx + 1)*num_images)] = clean_images 
        
        elif 'polygon' in trigger_type: # 'polygon_3', 'polygon_5', 'polygon_all' etc
            trigger_side_count = trigger_type.split('_')[1]            
            triggers_path = params['triggers_path']
            
            # read all trigger file names
            fns = [os.path.join(triggers_path, fn) for fn in os.listdir(triggers_path) if fn.endswith('png')]

            suitable_triggers = fns if trigger_side_count == 'all' else [fn for fn in fns if 'trigger_{}'.format(trigger_side_count) in fn]
            
            trigger_size_min, trigger_size_max = params['trigger_size_min'], params['trigger_size_max'] # area ratio
            trigger_color = params['trigger_color'] # [R, G, B]

            for cur_img, (foreground_image_idx, background_image_idx) in enumerate(selected_indices):
                # read a random trigger
                trigger_path = random.choice(suitable_triggers)
                raw_trigger = skimage.io.imread(trigger_path).astype(np.float32) # RGBA

                triggered_foreground = combine_foreground_trigger(foreground_images[foreground_image_idx], raw_trigger, trigger_size_min, trigger_size_max, trigger_color)
                triggered_images[(trigger_idx*num_images) + cur_img] = combine_foreground_background(background_images[background_image_idx], triggered_foreground)

    return clean_images, triggered_images