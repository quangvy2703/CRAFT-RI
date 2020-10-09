import pickle
import argparse
import os
import numpy as np
import cv2
from tqdm import tqdm

color = (0, 255, 0)
thickness = 4
isClosed = True

parser = argparse.ArgumentParser(description='Convert data into ic15 data format')
parser.add_argument('--source_data', default='/home/ubuntu/Downloads/Compressed/15/dataset_v1.3/15', type=str, help='')
parser.add_argument('--target_data', default='', type=str, help='')
parser.add_argument('--output_dir', default='converted', type=str)

args = parser.parse_args()


def DDI2IC15_draw():
    ddi_data = args.source_data
    gen_boxes_dir = os.path.join(ddi_data, 'gen_boxes')
    gen_boxes_files = os.listdir(gen_boxes_dir)
    for gen_boxes_file in tqdm(gen_boxes_files):
        image_path = os.path.join(ddi_data, 'gen_imgs', gen_boxes_file.replace('.pickle', '.png'))
        image = cv2.imread(image_path)
        gen_boxes = pickle.load(open(os.path.join(gen_boxes_dir, gen_boxes_file), 'rb'))
        for gen_box in gen_boxes:
            coords = np.array(
                [gen_box['box'][1][::-1], gen_box['box'][3][::-1], gen_box['box'][2][::-1], gen_box['box'][0][::-1]],
                np.int32)
            coords = coords.reshape((-1, 1, 2))
            image = cv2.polylines(image, [coords],
                                  isClosed, color,
                                  thickness)
        cv2.imwrite("converted/" + gen_boxes_file.replace('.pickle', '.png'), image)


def get_the_last(dir):
    files = os.listdir(dir + "/ch4_training_images")
    index = [int(file.split('_')[0]) for file in files]
    return max(set(index)) if len(index) > 0 else 0


def coords2str(coords):
    line = ''
    for coord in coords:
        line += str(coord[0]) + ',' + str(coord[1]) + ','
    return line


def DDI2IC15():
    ddi_data = args.source_data
    output_dir = args.output_dir

    if os.path.isdir(output_dir + '/ch4_training_localization_transcription_gt') is False:
        os.mkdir(output_dir + '/ch4_training_localization_transcription_gt')
    if os.path.isdir(output_dir + '/ch4_training_images') is False:
        os.mkdir(output_dir + '/ch4_training_images')
    last_index = get_the_last(output_dir)
    last_index = last_index + 1 if last_index > 0 else last_index

    gen_boxes_dir = os.path.join(ddi_data, 'gen_boxes')
    gen_boxes_files = os.listdir(gen_boxes_dir)
    for gen_boxes_file in tqdm(gen_boxes_files):
        image_path = os.path.join(ddi_data, 'gen_imgs', gen_boxes_file.replace('.pickle', '.png'))
        image = cv2.imread(image_path)
        gen_boxes = pickle.load(open(os.path.join(gen_boxes_dir, gen_boxes_file), 'rb'))
        new_index = last_index + int(gen_boxes_file.split('_')[0])
        tail_index = gen_boxes_file.rsplit('_')[-1].replace('.pickle', '.txt')
        gt_file = open(output_dir + '/ch4_training_localization_transcription_gt/gt_' + str(new_index) + '_' + tail_index, 'w+')
        for gen_box in gen_boxes:
            coords = np.array(
                [gen_box['box'][1][::-1], gen_box['box'][3][::-1], gen_box['box'][2][::-1], gen_box['box'][0][::-1]],
                np.int32)
            line = coords2str(coords)
            line += str(gen_box['text']) + '\n'
            gt_file.write(line)
            # coords = coords.reshape((-1, 1, 2))
            # image = cv2.polylines(image, [coords],
            #                       isClosed, color,
            #                       thickness)
        gt_file.close()
        os.system('cp {} {}'.format(image_path, os.path.join(output_dir, 'ch4_training_images',
                                                             str(new_index) + '_' + tail_index.replace('.txt', '.png'))))
        # cv2.imwrite("converted/" + gen_boxes_file.replace('.pickle', '.png'), image)


DDI2IC15()
# print(get_the_last('/home/ubuntu/Downloads/Compressed/DDI'))