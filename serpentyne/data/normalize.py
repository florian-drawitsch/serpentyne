import h5py
import numpy as np
from pathlib import Path
from shutil import copyfile


class Normalizer:

    """Normalizes training data stored in hdf5 container

    Args:
        path_source:
            Path to hdf5 container holding training data
        in_place:
            Determines whether the normalized data should overwrite the source data
            (Default: False)
        path_target:
            Path to hdf5 container to hold the normalized training data (only used if in_place is False)
        key_input:
            Specifies the hdf5 group key referencing the input / raw data
            (Default: 'input')
        key_target:
            Specifies the hdf5 group key referencing the target / label data
            (Default: 'target')
    """

    def __init__(self,
                 path_source: Path,
                 in_place: bool = False,
                 path_target: Path = None,
                 key_input: str = 'input',
                 key_target: str = 'target'):

        if path_source is str:
            path_source = Path(path_source)

        if path_target is str:
            path_target = Path(path_target)

        self.path_source = path_source
        self.in_place = in_place
        self.path_target = path_target
        self.key_input = key_input
        self.key_target = key_target

        self.stats = None

    def get_stats(self, which_key: str = None):

        """Returns summary statistics for input and target data across all hdf5 groups

        Args:
            which_key:
                Determines for which data type the statistics should be returned. Either 'input' or 'target'
                (Default: 'input')
        Returns:
            stats:
                Nested dict providing mean, weighted mean, std, shape and num_elements for each group as well as
                summarized for all groups
        """

        if which_key is None:
            which_key = self.key_input

        with h5py.File(self.path_source, 'r') as f:

            stats = dict()
            for key in list(f.keys()):
                ds = f[key][which_key]
                stats_group = {'min': np.amin(ds),
                               'max': np.amax(ds),
                               'mean': np.mean(ds),
                               'std': np.std(ds),
                               'shape': ds.shape,
                               'num_elements': np.prod(ds.shape)}
                stats[key] = stats_group

            mins = [stats[key]['min'] for key in stats.keys()]
            maxs = [stats[key]['max'] for key in stats.keys()]
            means = [stats[key]['mean'] for key in stats.keys()]
            num_elements = [stats[key]['num_elements'] for key in stats.keys()]
            stds = [stats[key]['std'] for key in stats.keys()]

            stats_summary = {'min': np.amin(mins),
                             'max': np.amin(maxs),
                             'mean': np.mean(means),
                             'mean_weighted': np.sum(np.asarray(means)*np.asarray(num_elements))/sum(num_elements),
                             'std': np.mean(stds)}

            stats['summary'] = stats_summary

        self.stats = stats

    def normalize(self):

        if self.in_place is True:
            self.path_target = self.path_source
        else:
            copyfile(self.path_source.as_posix(), self.path_target.as_posix())

        self.get_stats(which_key=self.key_input)

        with h5py.File(self.path_source, 'r+') as f:

            for key in list(f.keys()):
                ds = f[key][self.key_input]
                x = ds[:]
                x_norm = (x.astype(np.float32) - np.float32(self.stats['summary']['mean_weighted'])) / \
                         np.float32(self.stats['summary']['std'])

                ds[:, :, :] = x_norm

                ds = f[key][self.key_target]
                y = ds[:]
                y[y > 1] = 1
                ds[:, :, :] = y
