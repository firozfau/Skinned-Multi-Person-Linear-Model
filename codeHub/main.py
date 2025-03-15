# -*- coding: utf-8 -*-

# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is
# holder of all proprietary rights on this computer program.
# You can only use this computer program if you have closed
# a license agreement with MPG or you get the right to use the computer
# program from someone who is authorized to grant you that right.
# Any use of the computer program without a valid license is prohibited and
# liable to prosecution.
#
# Copyright©2019 Max-Planck-Gesellschaft zur Förderung
# der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
# for Intelligent Systems. All rights reserved.
#
# Contact: ps-license@tuebingen.mpg.de

import os.path as osp
import argparse

import numpy as np
import torch
import smplx
import pyrender  # Make sure to import pyrender
import trimesh  # Ensure trimesh is also imported for mesh handling


def main(model_folder,
         model_type='smplx',
         ext='npz',
         genders=['neutral'],
         plot_joints=False,
         num_betas=10,
         sample_shape=True,
         sample_expression=True,
         num_expression_coeffs=10,
         plotting_module='pyrender',
         use_face_contour=False):
    models = {}
    outputs = {}
    for gender in genders:
        models[gender] = smplx.create(model_folder, model_type=model_type,
                                      gender=gender, use_face_contour=use_face_contour,
                                      num_betas=num_betas,
                                      num_expression_coeffs=num_expression_coeffs,
                                      ext=ext)
        print(models[gender])

        betas, expression = None, None
        if sample_shape:
            betas = torch.randn([1, models[gender].num_betas], dtype=torch.float32)
        if sample_expression:
            expression = torch.randn(
                [1, models[gender].num_expression_coeffs], dtype=torch.float32)

        output = models[gender](betas=betas, expression=expression,
                                return_verts=True)
        outputs[gender] = output

    # Minimum distance between the models
    min_distance = 1.5  # Adjust this value as needed for spacing

    # Set the camera view parameters for a front view
    camera_position = np.array([0, 0, 3])  # Position the camera in front of the models
    camera_look_at = np.array([0, 0, 0])  # Looking at the origin
    camera_up = np.array([0, 1, 0])  # Up direction

    # Set the camera in the scene
    camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0)
    camera_node = pyrender.Node(
        camera=camera,
        matrix=np.eye(4) @ np.array([[1, 0, 0, camera_position[0]],
                                     [0, 1, 0, camera_position[1]],
                                     [0, 0, 1, camera_position[2]],
                                     [0, 0, 0, 1]])
    )
    scene = pyrender.Scene()
    scene.add_node(camera_node)

    for idx, gender in enumerate(genders):
        vertices = outputs[gender].vertices.detach().cpu().numpy().squeeze()
        model_faces = models[gender].faces

        vertex_colors = np.ones([vertices.shape[0], 4]) * [0.3, 0.3, 0.3, 0.8]
        tri_mesh = trimesh.Trimesh(vertices, model_faces,
                                   vertex_colors=vertex_colors)

        # Set the translation for each model to avoid overlap
        translation = np.array([(idx - (len(genders) - 1) / 2) * min_distance, 0, 0])  # Center the models
        transform = np.eye(4)
        transform[:3, 3] = translation
        mesh = pyrender.Mesh.from_trimesh(tri_mesh, poses=[transform])

        scene.add(mesh)

        if plot_joints:
            joints = outputs[gender].joints.detach().cpu().numpy().squeeze()
            sm = trimesh.creation.uv_sphere(radius=0.005)
            sm.visual.vertex_colors = [0.9, 0.1, 0.1, 1.0]
            tfs = np.tile(np.eye(4), (len(joints), 1, 1))
            tfs[:, :3, 3] = joints
            joints_pcl = pyrender.Mesh.from_trimesh(sm, poses=tfs)
            scene.add(joints_pcl)

    pyrender.Viewer(scene, use_raymond_lighting=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='SMPL-X Demo')

    parser.add_argument('--model-folder', required=True, type=str,
                        help='The path to the model folder')
    parser.add_argument('--model-type', default='smplx', type=str,
                        choices=['smpl', 'smplh', 'smplx', 'mano', 'flame'],
                        help='The type of model to load')
    parser.add_argument('--genders', nargs='+', default=['neutral'],
                        help='List of genders to load (e.g., neutral male female)')
    parser.add_argument('--num-betas', default=10, type=int,
                        dest='num_betas',
                        help='Number of shape coefficients.')
    parser.add_argument('--num-expression-coeffs', default=10, type=int,
                        dest='num_expression_coeffs',
                        help='Number of expression coefficients.')
    parser.add_argument('--plotting-module', type=str, default='pyrender',
                        dest='plotting_module',
                        choices=['pyrender', 'matplotlib', 'open3d'],
                        help='The module to use for plotting the result')
    parser.add_argument('--ext', type=str, default='npz',
                        help='Which extension to use for loading')
    parser.add_argument('--plot-joints', default=False,
                        type=lambda arg: arg.lower() in ['true', '1'],
                        help='Plot joints or not')
    parser.add_argument('--sample-shape', default=True,
                        dest='sample_shape',
                        type=lambda arg: arg.lower() in ['true', '1'],
                        help='Sample a random shape')
    parser.add_argument('--sample-expression', default=True,
                        dest='sample_expression',
                        type=lambda arg: arg.lower() in ['true', '1'],
                        help='Sample a random expression')
    parser.add_argument('--use-face-contour', default=False,
                        type=lambda arg: arg.lower() in ['true', '1'],
                        help='Compute the contour of the face')

    args = parser.parse_args()

    model_folder = osp.expanduser(osp.expandvars(args.model_folder))
    model_type = args.model_type
    plot_joints = args.plot_joints
    use_face_contour = args.use_face_contour
    genders = args.genders
    ext = args.ext
    plotting_module = args.plotting_module
    num_betas = args.num_betas
    num_expression_coeffs = args.num_expression_coeffs
    sample_shape = args.sample_shape
    sample_expression = args.sample_expression

    main(model_folder, model_type, ext=ext,
         genders=genders,
         plot_joints=plot_joints,
         num_betas=num_betas,
         num_expression_coeffs=num_expression_coeffs,
         sample_shape=sample_shape,
         sample_expression=sample_expression,
         plotting_module=plotting_module,
         use_face_contour=use_face_contour)
