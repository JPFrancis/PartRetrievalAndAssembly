
import torch
import numpy as np
import os
import joblib
from util_vis import *
from config import * 
from util_motion import *
import joblib
sys.modules['sklearn.externals.joblib'] = joblib
import six
sys.modules['sklearn.externals.six'] = six
import csv
import pickle
import open3d as o3d
import numpy as np

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

def get_kaedim_shapes(data_dir, category, shape_ids, count, all_formats):
    print('loading shapes')

    shapes_file = os.path.join(data_dir, 'kaedim-chair-eval-dataset.pickle')
    
    all_shape_vol_pcs = []
    shape_vol_pcs = []
    shape_sur_pcs = []

    try:
        with open(shapes_file, 'rb') as f:
            while True:
                try:
                    all_shape_vol_pcs = pickle.load(f)

                except EOFError:
                    # Stop when we reach the end of the file
                    break

    except Exception as e:
        print(f"An error occurred: {e}")

    for shape_id in shape_ids:
        
        shape_vol_pcs.append(all_shape_vol_pcs[shape_id])

        if len(shape_vol_pcs) >= count:
            break
    
    for shape_vol_pc in shape_vol_pcs:
        shape_sur_pcs.append(get_surface_from_volume(shape_vol_pc, 0.01))

    return shape_vol_pcs, shape_sur_pcs

def get_shapes(data_dir, dataset, category, shape_ids, count, all_formats=False):

    count = int(count)

    if dataset == 'partnet':
        return get_partnet_shapes(data_dir, category, shape_ids, count, all_formats)
    else:
        print('wrong dataset')
        exit()

def kaedim_get_shapes(data_dir, dataset, category, shape_ids, count, all_formats=False):

    count = int(count)

    return get_kaedim_shapes(data_dir, category, shape_ids, count, all_formats)

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

def get_kaedim_parts(data_dir, category, part_ids, count, all_formats):

    #print('shape_ids', shape_ids)
    #exit()

    print('loading parts')

    parts_file = os.path.join(data_dir, 'kaedim-chair-parts-library.pickle')

    all_part_sur_pcs = []
    part_vol_pcs = []
    part_sur_pcs = []

    try:
        with open(parts_file, 'rb') as f:
            while True:
                try:
                    # Load one object from the file
                    all_part_sur_pcs = pickle.load(f)

                except EOFError:
                    # Stop when we reach the end of the file
                    break
    
    except Exception as e:
        print(f"An error occurred: {e}")

    print(f'Loading {min(len(part_ids), count)} parts...')
    
    for i, part_id in enumerate(part_ids):
        progress = (i + 1) / min(len(part_ids), count) * 100
        bar_length = 30
        filled_length = int(bar_length * progress / 100)
        bar = '█' * filled_length + '░' * (bar_length - filled_length)
        print(f'\rLoading parts: [{bar}] {progress:.1f}% ({i+1}/{min(len(part_ids), count)})', end='', flush=True)
        part_sur_pcs.append(all_part_sur_pcs[part_id])

        if len(part_sur_pcs) >= count:
            break
    
    print()  # New line after loading progress
    print(f'Converting {len(part_sur_pcs)} surface point clouds to volume...')
    for i, part_sur_pc in enumerate(part_sur_pcs):
        progress = (i + 1) / len(part_sur_pcs) * 100
        bar_length = 30
        filled_length = int(bar_length * progress / 100)
        bar = '█' * filled_length + '░' * (bar_length - filled_length)
        print(f'\rConverting to volume: [{bar}] {progress:.1f}% ({i+1}/{len(part_sur_pcs)})', end='', flush=True)
        vol_pcd = get_volume_from_surface(part_sur_pc, 5000)
        # Convert Open3D PointCloud to numpy array
        vol_points = np.asarray(vol_pcd.points)
        part_vol_pcs.append(vol_points)

    print()  # New line after conversion progress
    print(f'Successfully loaded {len(part_vol_pcs)} parts')
    return part_vol_pcs, part_sur_pcs

def get_parts(data_dir, dataset, category, count, shape_ids=[], all_formats=False):

    count = int(count)
    print('loading parts.............')

    if dataset == 'partnet':
    
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

def kaedim_get_parts(data_dir, dataset, category, count, part_ids=[], all_formats=False):
    
    count = int(count)
    print('loading parts.............')

    part_vol_pcs, part_sur_pcs = get_kaedim_parts(data_dir, category, part_ids, count, all_formats)

    return part_vol_pcs, part_sur_pcs

def get_surface_from_volume(point_cloud, alpha):
    """
    Converts a volumetric point cloud to a surface point cloud using an Alpha Shape.

    Args:
        point_cloud (numpy.ndarray): The (N, 3) array of points.
        alpha (float): The alpha parameter that controls the tightness of the surface fit.
                       A smaller alpha creates a tighter, more detailed surface.

    Returns:
        open3d.geometry.PointCloud: The resulting surface point cloud.
    """
    # Create an Open3D point cloud object
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(point_cloud)

    # Compute the Alpha Shape
    alpha_shape_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd, alpha)
    
    # The surface points are the vertices of the resulting mesh
    surface_points_pcd = o3d.geometry.PointCloud()
    surface_points_pcd.points = alpha_shape_mesh.vertices
    
    return surface_points_pcd


def get_volume_from_surface(surface_points_np, num_points_to_generate):
    """
    Converts a surface point cloud to a volumetric point cloud by sampling points
    from the volume of a mesh constructed from the surface.

    Args:
        surface_points_np (numpy.ndarray): The (N, 3) array of points representing the surface.
        num_points_to_generate (int): The number of volumetric points to create.

    Returns:
        open3d.geometry.PointCloud: The resulting volumetric point cloud.
    """
    # --- FIX: Convert NumPy array to Open3D PointCloud ---
    surface_pcd = o3d.geometry.PointCloud()
    surface_pcd.points = o3d.utility.Vector3dVector(surface_points_np)
    # ----------------------------------------------------

    # Estimate normals for the surface point cloud
    surface_pcd.estimate_normals()

    # Create a mesh from the surface point cloud using the Ball Pivoting algorithm
    distances = surface_pcd.compute_nearest_neighbor_distance()
    avg_dist = np.mean(distances)
    radius = 1.5 * avg_dist
    bpa_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
        surface_pcd, o3d.utility.DoubleVector([radius, radius * 2]))

    # Make the mesh watertight
    bpa_mesh.remove_degenerate_triangles()
    bpa_mesh.remove_duplicated_triangles()
    bpa_mesh.remove_duplicated_vertices()
    bpa_mesh.remove_non_manifold_edges()

    # Sample points from the volume of the mesh
    volumetric_pcd = bpa_mesh.sample_points_uniformly(number_of_points=num_points_to_generate)
    
    return volumetric_pcd
