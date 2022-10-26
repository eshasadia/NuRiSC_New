import os

import numpy as np

import cv2

from skimage import draw

import xml.etree.ElementTree as ET

from tqdm import tqdm
image_path = 'C:/Users/es255022/OneDrive - Teradata/Desktop/Dataset/monuseg/train/image' # Path to save binary masks corresponding to xml files

data_path = 'C:/Users/es255022/OneDrive - Teradata/Desktop/Dataset/MoNuSAC/mask' #Path to read data from

destination_path = 'C:/Users/es255022/OneDrive - Teradata/Desktop/Dataset/MoNuSAC/newmask' # Path to save binary masks corresponding to xml files
annotations = [x[2] for x in os.walk(data_path)][0] #Names of all xml files in the data_path
for name in tqdm(annotations):

    tree = ET.parse(f'{data_path}/{name}')

    root = tree.getroot()

    child = root[0]

    for x in child:

        r = x.tag

        binary_mask = np.transpose(np.zeros((1000, 1000)))

        if r == 'Regions':

            for y in x:

                y_tag = y.tag

                if y_tag == 'Region':

                    regions = []

                    vertices = y[1]

                    coords = np.zeros((len(vertices), 2))

                    for i, vertex in enumerate(vertices):
                        coords[i][0] = vertex.attrib['X']

                        coords[i][1] = vertex.attrib['Y']

                    regions.append(coords)

                    vertex_row_coords = regions[0][:, 0]

                    vertex_col_coords = regions[0][:, 1]

                    fill_row_coords, fill_col_coords = draw.polygon(vertex_col_coords, vertex_row_coords,
                                                                    binary_mask.shape)

                    binary_mask[fill_row_coords, fill_col_coords] = 255

            mask_path = f'{destination_path}/{name[:-4]}.jpg'

            cv2.imwrite(mask_path, binary_mask)