import random
import torch
import h5py
import numpy as np
from pathlib import Path
from typing import Sequence, Callable
from torch.utils.data import Dataset
from matplotlib import pyplot as plt


class D3d(Dataset):
    """Generates 3D datasets for pytorch"""

    def __init__(self,
                 data_path: Path,
                 input_shape: Sequence[int],
                 output_shape: Sequence[int],
                 pad_target: bool = True,
                 transform: Callable = None):
        """
        Args:
            data_path:
                Path to hdf5 container holding labelled training data. The container should hold one or multiple groups
                each holding two datasets with keys 'input' and 'target' holding raw data and corresponding labels,
                respectively
            input_shape:
                Shape of the input patches to be fed into the network
            output_shape:
                Shape of the output patches predicted by the network
            pad_target:
                If true, target patches are padded to match shape of input patches
            transform (callable, optional):
                Optional transformation to be applied to the training data
        """

        self.data_path = data_path
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.pad_target = pad_target
        self.transform = transform

        self.data_object: h5py = None
        self.inputs = []
        self.targets = []
        self.borders = []
        self.class_counts = []

        self.cube_mesh_grids = []
        self.cube_idx_ranges_min = []
        self.cube_idx_ranges_max = []

        self.get_data_object()
        self.get_borders()
        self.get_class_counts()
        self.get_cube_mesh_grids()
        self.get_cube_idx_lims()


    def __len__(self):
        return self.cube_idx_ranges_max[-1]

    def __getitem__(self, idx):
        return self.get_ordered_sample(idx)

    def get_data_object(self):
        self.data_object = h5py.File(self.data_path, 'r')

    def get_borders(self):

        for cube_key in list(self.data_object.keys()):
            self.borders.append((np.asarray(self.data_object[cube_key]['input'].shape) - np.asarray(self.data_object[cube_key]['target'].shape)) / 2)

    def get_class_counts(self):

        for cube_key in list(self.data_object.keys()):
            unique, counts = np.unique(self.data_object[cube_key]['target'], return_counts=True)
            self.class_counts.append(dict(zip(unique, counts)))

    def get_cube_mesh_grids(self):
        for cube_idx, cube_key in enumerate(list(self.data_object.keys())):
            corner_min_target = np.floor(np.asarray(self.output_shape)/2)
            corner_max_target = np.asarray(self.data_object[cube_key]['target'].shape) - np.asarray(self.output_shape)
            x = np.arange(corner_min_target[0], corner_max_target[0], self.output_shape[0])
            y = np.arange(corner_min_target[1], corner_max_target[1], self.output_shape[1])
            z = np.arange(corner_min_target[2], corner_max_target[2], self.output_shape[2])
            xm, ym, zm = np.meshgrid(x, y, z)
            mesh_grid_target = {'x': xm, 'y': ym, 'z': zm}
            borders = self.borders[cube_idx]
            mesh_grid_input = {'x': xm + borders[0], 'y': ym + borders[1], 'z': zm + borders[2]}
            mesh_grids = {'input': mesh_grid_input, 'target': mesh_grid_target}
            self.cube_mesh_grids.append(mesh_grids)

    def get_cube_idx_lims(self):

        """ Computes the global linear idx limits contained in the respective training data cubes"""
        for cube_idx, _ in enumerate(list(self.data_object.keys())):
            if cube_idx == 0:
                self.cube_idx_ranges_min.append(0)
            else:
                self.cube_idx_ranges_min.append(self.cube_idx_ranges_max[cube_idx - 1] + 1)

            self.cube_idx_ranges_max.append(self.cube_idx_ranges_min[cube_idx] +
                                            self.cube_mesh_grids[cube_idx]['target']['x'].size - 1)

    def get_ordered_sample(self, sample_idx):

        """ Retrieves a pair of input and target tensors from all available training cubes based on the global linear
        sample_idx"""

        # Get appropriate training data cube sample_idx based on global linear sample_idx
        cube_idx = int(np.argmax(np.asarray(self.cube_idx_ranges_max) >= sample_idx))
        cube_key = list(self.data_object.keys())[cube_idx]
        # Get appropriate subscript index for the respective training data cube, given the global linear index
        cube_sub = np.unravel_index(sample_idx - self.cube_idx_ranges_min[cube_idx],
                                    dims=self.cube_mesh_grids[cube_idx]['target']['x'].shape)

        # Get target sample
        origin_target = np.asarray([
            self.cube_mesh_grids[cube_idx]['target']['x'][cube_sub[0], cube_sub[1], cube_sub[2]],
            self.cube_mesh_grids[cube_idx]['target']['y'][cube_sub[0], cube_sub[1], cube_sub[2]],
            self.cube_mesh_grids[cube_idx]['target']['z'][cube_sub[0], cube_sub[1], cube_sub[2]],
        ]).astype(np.uint16)
        ds_target = self.data_object[cube_key]['target']
        target = D3d.crop_c(ds_target, origin_target, self.output_shape)

        # Get input sample
        origin_input = np.asarray([
            self.cube_mesh_grids[cube_idx]['input']['x'][cube_sub[0], cube_sub[1], cube_sub[2]],
            self.cube_mesh_grids[cube_idx]['input']['y'][cube_sub[0], cube_sub[1], cube_sub[2]],
            self.cube_mesh_grids[cube_idx]['input']['z'][cube_sub[0], cube_sub[1], cube_sub[2]],
        ]).astype(np.uint16)
        ds_input = self.data_object[cube_key]['input']
        input_ = D3d.crop_c(ds_input, origin_input, self.input_shape)

        if self.pad_target is True:
            target = self.pad(target)

        input_ = torch.from_numpy(np.reshape(input_, (1, input_.shape[0], input_.shape[1], input_.shape[2])))
        target = torch.from_numpy(np.reshape(target, (1, target.shape[0], target.shape[1], target.shape[2])))

        return input_, target

    def get_random_sample(self):

        """ Retrieves a random pair of input and target tensors from all available training cubes"""

        idx = random.sample(range(self.cube_idx_ranges_max[-1]), 1)
        input_, target = self.get_ordered_sample(idx)

        return input_, target

    def pad(self, target):
        pad_shape = np.floor((np.asarray(self.input_shape) - np.asarray(self.output_shape)) / 2).astype(int)
        target = np.pad(target,
                        ((pad_shape[0], pad_shape[0]), (pad_shape[1], pad_shape[1]), (pad_shape[2], pad_shape[2])),
                        'constant')

        return target

    def get_item_inds_with_labels(self):
        inds = []
        counts = []
        for idx in range(self.__len__()):

            if idx % 100 == 0:
                print('Finding indices containing labels ... {} of {}'.format(idx, self.__len__()))

            _, target = self.__getitem__(idx)
            count = target.sum()

            if count > 0:
                inds.append(idx)
                counts.append(count)

        return inds, counts

    def show_hist(self):

        fig, axs = plt.subplots(len(self.inputs), 2)

        for idx, _ in enumerate(self.inputs):
            axs[idx, 0].hist(self.inputs[idx].flatten())
            axs[idx, 1].hist(self.targets[idx].flatten())

    def show_sample(self, idx, z_shift=0, superimpose=True):

        input_, target = self.get_ordered_sample(idx)

        input_ = input_.numpy().squeeze()
        target = target.numpy().squeeze()

        if sum(np.asarray(input_.shape) - np.asarray(target.shape)) > 0:
            target = self.pad(target)

        z_input = np.round(input_.shape[2] / 2).astype(int) + z_shift
        z_target = np.round(target.shape[2] / 2).astype(int) + z_shift

        if superimpose is False:
            fig, axs = plt.subplots(1, 2)
            axs[0].imshow(input_[:, :, z_input], cmap='gray', vmin=-3, vmax=3)
            axs[1].imshow(target[:, :, z_target], vmin=0, vmax=1)

        else:
            fig, ax = plt.subplots()
            ax.imshow(input_[:, :, z_input], cmap='gray', vmin=-3, vmax=3)
            ax.imshow(target[:, :, z_target], vmin=0, vmax=1, alpha=0.5)

        plt.show()

    @staticmethod
    def crop_c(data, origin, shape):
        """ Crops centered shape around origin from numpy 3d array"""
        data_c = data[int(origin[0] - np.floor(shape[0] / 2)):
                      int(origin[0] + np.floor(shape[0] / 2) + 1),
                      int(origin[1] - np.floor(shape[1] / 2)):
                      int(origin[1] + np.floor(shape[1] / 2) + 1),
                      int(origin[2] - np.floor(shape[2] / 2)):
                      int(origin[2] + np.floor(shape[2] / 2) + 1)]

        return data_c
