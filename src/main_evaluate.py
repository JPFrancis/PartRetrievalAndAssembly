


import os
import joblib
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader
from scipy.spatial.transform import Rotation as R
from pytorch3d import *
import trimesh
import numpy as np

from main_ours_pretrain import *
from util_collision import *
from util_motion import *
from util_vis import *
from util_mesh_surface import *
from util_mesh_volume import *
from main_common import *
from config import *
from data_manager import *

torch.set_printoptions(precision=10)


def dict_to_trimesh(mesh_dict):
    """Convert a dictionary mesh (with vertices and triangles keys) back to a trimesh object"""
    if isinstance(mesh_dict, dict) and 'vertices' in mesh_dict and 'triangles' in mesh_dict:
        return trimesh.Trimesh(vertices=mesh_dict['vertices'], faces=mesh_dict['triangles'])
    elif hasattr(mesh_dict, 'vertices') and hasattr(mesh_dict, 'faces'):
        # Already a trimesh object
        return mesh_dict
    else:
        # Unknown format, return as is
        return mesh_dict


def convert_meshes_to_trimesh(meshes):
    """Convert a list of meshes (which might be dictionaries) to trimesh objects"""
    if meshes is None:
        return None
    converted_meshes = []
    for mesh in meshes:
        converted_meshes.append(dict_to_trimesh(mesh))
    return converted_meshes


def render_shape(shape_id, dict, folder, spec, is_summary):
    def safe_render_meshes(meshes, is_target, filename):
        """Try pyrender first, fall back to plotly if OpenGL fails"""
        try:
            render_meshes(meshes, is_target, filename)
        except Exception as e:
            print(f"Pyrender failed with error: {e}")
            print(f"Falling back to plotly rendering for {filename}")
            # Use plotly-based rendering as fallback
            try:
                display_meshes(meshes, filename, save=True)
            except Exception as plotly_error:
                print(f"Plotly rendering also failed: {plotly_error}")
                print(f"Skipping visualization for {filename}")
                # Create a simple placeholder image or skip visualization
                # For now, just skip the visualization
                pass
    
    if is_summary:
        # Convert shape_mesh to trimesh if it's a dictionary
        shape_mesh = dict_to_trimesh(dict['shape_mesh'])
        safe_render_meshes([shape_mesh], True, os.path.join(folder, str(shape_id)+'zgt_shape_mesh.png'))

        if 'recon_part_meshes_after' in dict:
            recon_part_meshes = convert_meshes_to_trimesh(dict['recon_part_meshes_after'])
            safe_render_meshes(recon_part_meshes, False, os.path.join(folder, str(shape_id)+'retrieved_parts_mesh_after.png'))
        
        if 'recon_shape_mesh' in dict:
            recon_shape_mesh = dict_to_trimesh(dict['recon_shape_mesh'])
            safe_render_meshes([recon_shape_mesh], True, os.path.join(folder, str(shape_id)+'recon_shape_mesh.png'))

        if 'recon_part_meshes_after' in dict:
            blender_recon_folder = os.path.join(folder, 'blender', str(shape_id)+'recon')
            if not os.path.exists(blender_recon_folder):
                os.makedirs(blender_recon_folder)
            recon_part_meshes = convert_meshes_to_trimesh(dict['recon_part_meshes_after'])
            for i in range(len(recon_part_meshes)):
                recon_part_meshes[i].export(os.path.join(blender_recon_folder, str(i)+'.obj'), file_type='obj')
                f = open(os.path.join(blender_recon_folder, str(i)+'_'+str(dict['part_indices'][i])+".txt"), "w")
                f.close()
            
            blender_target_folder = os.path.join(folder, 'blender', str(shape_id)+'target')
            if not os.path.exists(blender_target_folder):
                os.makedirs(blender_target_folder)
            shape_mesh.export(os.path.join(blender_target_folder, 'target.obj'), file_type='obj')
            
            shape_vol_pc = volumetric_sample_mesh(shape_mesh, 4096)
            joblib.dump(shape_vol_pc, os.path.join(blender_target_folder, 'target_pc.joblib'))
    
    else:
        # Convert shape_mesh to trimesh if it's a dictionary
        shape_mesh = dict_to_trimesh(dict['shape_mesh'])
        safe_render_meshes([shape_mesh], True, os.path.join(folder, 'zgt_shape_mesh.png'))

        if 'recon_part_meshes_before' in dict:
            recon_part_meshes_before = convert_meshes_to_trimesh(dict['recon_part_meshes_before'])
            safe_render_meshes(recon_part_meshes_before, False, os.path.join(folder, str(spec)+'retrieved_parts_mesh_before.png'))
        if 'recon_part_meshes_after' in dict:
            recon_part_meshes_after = convert_meshes_to_trimesh(dict['recon_part_meshes_after'])
            safe_render_meshes(recon_part_meshes_after, False, os.path.join(folder, str(spec)+'_'+str(dict['score'])+'retrieved_parts_mesh_after.png'))
            #render_meshes(dict['recon_part_meshes_after'], False, os.path.join(folder, str(spec)+'_retrieved_parts_mesh_after.png'))

            blender_recon_folder = os.path.join(folder, 'blender', str(shape_id)+str(len(recon_part_meshes_after))+'recon_mesh')
            if not os.path.exists(blender_recon_folder):
                os.makedirs(blender_recon_folder)
            for i in range(len(recon_part_meshes_after)):
                recon_part_meshes_after[i].export(os.path.join(blender_recon_folder, str(i)+'.obj'), file_type='obj')
            
        if 'recon_shape_mesh' in dict:
            recon_shape_mesh = dict_to_trimesh(dict['recon_shape_mesh'])
            safe_render_meshes([recon_shape_mesh], True, os.path.join(folder, str(shape_id)+'recon_shape_mesh.png'))

        if 'shape_vol_pc' in dict:
            display_pcs([dict['shape_vol_pc']], os.path.join(folder, str(spec)+'shape_vol_pc.png'), True)

        if 'shape_recon' in dict:
            display_pcs(dict['shape_recon'], os.path.join(folder, str(spec)+'shape_recon.png'), True)
        if 'shape_seg' in dict:
            display_pcs(dict['shape_seg'], os.path.join(folder, str(spec)+'shape_seg.png'), True)
        if 'pred_part_meshes' in dict and dict['pred_part_meshes'] is not None:
            pred_part_meshes = convert_meshes_to_trimesh(dict['pred_part_meshes'])
            safe_render_meshes(pred_part_meshes, False, os.path.join(folder, str(spec)+'pred_part_meshes.png'))
        

def visulize_shape(shape_id, summary_folder):

    print('visulize shape ', shape_id)
    shape_folder = os.path.join(summary_folder, str(shape_id))
    for shape_file in list(os.listdir(shape_folder)):
        if shape_file.endswith('.joblib'):
            print('shape_file', shape_file)
            shape_dict = joblib.load(os.path.join(shape_folder, str(shape_file)))
            render_shape(shape_id, shape_dict, shape_folder, shape_file.split('.')[0], False)
            #os.remove(os.path.join(shape_folder, str(shape_file)))
    shape_summary_dict = joblib.load(os.path.join(summary_folder, str(shape_id)+'.joblib'))
    render_shape(shape_id, shape_summary_dict, summary_folder, '', True)

def generate_visulizations(data_dir, exp_folder, part_dataset, part_category, part_count, shape_dataset, shape_category, train_shape_count, test_shape_count, eval_on_train_shape_count):

    train_shape_ids = list(range(16))
    test_shape_ids = [18, 19]
    part_ids = list(range(part_count))

    eval_train_shape_ids = train_shape_ids[0:eval_on_train_shape_count]
    summary_folder = os.path.join(exp_folder, 'train_summary')
    if use_parallel:
        results = Parallel(n_jobs=2)(delayed(visulize_shape)(shape_id, summary_folder) for shape_id in eval_train_shape_ids)
    else:
        for shape_id in eval_train_shape_ids:
            visulize_shape(shape_id, summary_folder)

    test_shape_ids = test_shape_ids[0:test_shape_count]
    summary_folder = os.path.join(exp_folder, 'test_summary')
    if use_parallel:
        results = Parallel(n_jobs=2)(delayed(visulize_shape)(shape_id, summary_folder) for shape_id in test_shape_ids)
    else:
        for shape_id in test_shape_ids:
            visulize_shape(shape_id, summary_folder)


if __name__ == "__main__":

    exp_folder = os.path.join(global_args.exp_dir, global_args.part_dataset + global_args.part_category + '_to_' + global_args.shape_dataset + global_args.shape_category + str(global_args.train_shape_count) + 'shift' + str(use_shift) + 'borrow' + str(use_borrow))
    if not os.path.exists(exp_folder):
        os.makedirs(exp_folder)
    
    generate_visulizations(global_args.data_dir, exp_folder, global_args.part_dataset, global_args.part_category, global_args.part_count, global_args.shape_dataset, global_args.shape_category, global_args.train_shape_count, global_args.test_shape_count, global_args.eval_on_train_shape_count)
