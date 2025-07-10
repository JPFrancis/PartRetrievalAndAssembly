
import torch
import numpy as np
import os
import joblib
import pickle
from util_vis import *
from config import * 
from util_motion import *
import joblib
sys.modules['sklearn.externals.joblib'] = joblib
import six
sys.modules['sklearn.externals.six'] = six
import csv

def load_kaedim_dataset(pickle_file):
    """
    Load data from Kaedim pickle file
    Returns: (part_meshes, part_vol_pcs, part_sur_pcs, shape_meshes, shape_vol_pcs, shape_sur_pcs)
    """
    print(f'Loading Kaedim dataset from {pickle_file}')
    
    with open(pickle_file, 'rb') as f:
        data = pickle.load(f)
    
    # The Kaedim dataset is a list of point clouds (numpy arrays)
    # Each point cloud has shape (N, 3) where N is the number of points
    if isinstance(data, list) and len(data) > 0:
        # Convert numpy arrays to the expected format
        point_clouds = []
        for pc in data:
            if isinstance(pc, np.ndarray) and pc.shape[1] == 3:
                point_clouds.append(pc)
            else:
                print(f'Warning: Skipping item with shape {pc.shape if hasattr(pc, "shape") else "unknown"}')
        
        print(f'Loaded {len(point_clouds)} point clouds')
        
        # For Kaedim dataset, we'll use the same point clouds for both parts and shapes
        # You may want to adjust this based on your specific needs
        part_vol_pcs = point_clouds
        part_sur_pcs = point_clouds  # Using same data for surface points
        shape_vol_pcs = point_clouds
        shape_sur_pcs = point_clouds  # Using same data for surface points
        
        # Create empty meshes (not available in this dataset)
        part_meshes = []
        shape_meshes = []
        
        return part_meshes, part_vol_pcs, part_sur_pcs, shape_meshes, shape_vol_pcs, shape_sur_pcs
    else:
        raise ValueError("Pickle file doesn't contain expected data structure (list of point clouds)")

def read_split(split_file):

    source_ids = []
    train_ids = []
    test_ids = []
    val_ids = []

    split_file = split_file

    print('split_file', split_file)
    with open(split_file, newline='') as f:
        reader = csv.reader(f)
        for row in reader:
            if row[3] == 'source':
                source_ids.append(row[0])
            if row[3] == 'train':
                train_ids.append(row[0])
            if row[3] == 'test':
                test_ids.append(row[0])
            if row[3] == 'val':
                val_ids.append(row[0])

    return source_ids, train_ids, test_ids, val_ids



def get_partnet_shapes(data_dir, category, shape_ids, count, all_formats):
    print('loading shapes')

    shapes_folder = os.path.join(data_dir, 'final_partnet_shapes', str(category), 'final_shapes')
    
    shape_meshes = []
    shape_vol_pcs = []
    shape_sur_pcs = []

    for shape_id in shape_ids:
        
        shape_folder = os.path.join(shapes_folder, str(shape_id))
        if not os.path.isfile(os.path.join(shape_folder, 'mesh.joblib')) or not os.path.isfile(os.path.join(shape_folder, 'vol_pc.joblib')) or not os.path.isfile(os.path.join(shape_folder, 'sur_pc.joblib')):
            print('missing file')
            exit()
            continue
        
        shape_vol_pc = joblib.load(os.path.join(shape_folder, 'vol_pc.joblib'))
        shape_vol_pcs.append(shape_vol_pc)

        if all_formats:
            shape_mesh = joblib.load(os.path.join(shape_folder, 'mesh.joblib'))
            shape_meshes.append(shape_mesh)
            shape_sur_pc = joblib.load(os.path.join(shape_folder, 'sur_pc.joblib'))
            shape_sur_pcs.append(shape_sur_pc)

        if len(shape_vol_pcs) >= count:
            break

    return shape_meshes, shape_vol_pcs, shape_sur_pcs

def get_shapes(data_dir, dataset, category, shape_ids, count, all_formats=False):

    count = int(count)

    # Check if we should load from pickle file
    if hasattr(global_args, 'pickle_file') and global_args.pickle_file:
        print('Loading shapes from pickle file...')
        _, _, _, shape_meshes, shape_vol_pcs, shape_sur_pcs = load_kaedim_dataset(global_args.pickle_file)
        
        # Limit to requested count
        if len(shape_vol_pcs) > count:
            shape_vol_pcs = shape_vol_pcs[:count]
            if all_formats:
                shape_meshes = shape_meshes[:count]
                shape_sur_pcs = shape_sur_pcs[:count]
        
        return shape_meshes, shape_vol_pcs, shape_sur_pcs

    elif dataset == 'partnet':
        return get_partnet_shapes(data_dir, category, shape_ids, count, all_formats)
    else:
        print('wrong dataset')
        exit()

def get_partnet_parts(data_dir, category, shape_ids, count, all_formats):

    #print('shape_ids', shape_ids)
    #exit()

    print('loading parts')

    parts_folder = os.path.join(data_dir, 'final_partnet_shapes', str(category), 'final_parts')

    part_meshes = []
    part_vol_pcs = []
    part_sur_pcs = []
    part_to_shape_id = []
    for shape_folder in sorted(os.listdir(parts_folder)):

        #print('loading part related shape', shape_folder)
        
        if shape_folder not in shape_ids:
            continue

        #print('shape_folder', shape_folder)
        
        shape_parts_folder = os.path.join(parts_folder, str(shape_folder))
        for part_index in range(0, len(list(sorted(os.listdir(shape_parts_folder))))):
            if not os.path.isfile(os.path.join(shape_parts_folder, str(part_index), 'mesh.joblib')) or not os.path.isfile(os.path.join(shape_parts_folder, str(part_index), 'vol_pc.joblib')) or not os.path.isfile(os.path.join(shape_parts_folder, str(part_index), 'sur_pc.joblib')):
                continue
            
            vol_pc = joblib.load(os.path.join(shape_parts_folder, str(part_index), 'vol_pc.joblib'))
            part_vol_pcs.append(vol_pc)
            part_to_shape_id.append(shape_folder)

            if all_formats:
                mesh = joblib.load(os.path.join(shape_parts_folder, str(part_index), 'mesh.joblib'))

                #display_meshes([mesh])
                
                part_meshes.append(mesh)
                sur_pc = joblib.load(os.path.join(shape_parts_folder, str(part_index), 'sur_pc.joblib'))
                part_sur_pcs.append(sur_pc)
                

            if len(part_vol_pcs) >= count:
                return part_meshes, part_vol_pcs, part_sur_pcs, part_to_shape_id
    
    return part_meshes, part_vol_pcs, part_sur_pcs, part_to_shape_id

def get_parts(data_dir, dataset, category, count, shape_ids=[], all_formats=False):

    count = int(count)
    print('loading parts.............')

    # Check if we should load from pickle file
    if hasattr(global_args, 'pickle_file') and global_args.pickle_file:
        print('Loading parts from pickle file...')
        part_meshes, part_vol_pcs, part_sur_pcs, _, _, _ = load_kaedim_dataset(global_args.pickle_file)
        
        # Limit to requested count
        if len(part_vol_pcs) > count:
            part_vol_pcs = part_vol_pcs[:count]
            if all_formats:
                part_meshes = part_meshes[:count]
                part_sur_pcs = part_sur_pcs[:count]
        
        return part_meshes, part_vol_pcs, part_sur_pcs

    elif dataset == 'partnet':
    
        part_meshes, part_vol_pcs, part_sur_pcs, part_to_shapeIds = get_partnet_parts(data_dir, category, shape_ids, count, all_formats)

        filtered_part_meshes = []
        filtered_part_vol_pcs = []
        filtered_part_sur_pcs = []
        filtered_part_to_shapeIds = []
        for i in range(len(part_vol_pcs)):
            
            filtered_part_vol_pcs.append(part_vol_pcs[i])
            filtered_part_to_shapeIds.append(part_to_shapeIds[i])

            if all_formats:
                filtered_part_meshes.append(part_meshes[i])
                filtered_part_sur_pcs.append(part_sur_pcs[i])

        #joblib.dump(part_to_shapeIds, os.path.join('..', str(dataset)+'_'+str(category)+'_'+str(count)+'_part_to_shapeIds.joblib'))

        return filtered_part_meshes, filtered_part_vol_pcs, filtered_part_sur_pcs
    
    else:
        print('wrong dataset')