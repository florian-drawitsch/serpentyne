import h5py
from scipy import io as sio


def mat2hdf5(mat_filenames,
             mat_input_name,
             mat_target_name,
             hdf5_filename,
             hdf5_input_name,
             hdf5_target_name):

    with h5py.File(hdf5_filename, 'w') as f:

        for mat_idx, mat_filename in enumerate(mat_filenames):
            print('Converting ({} of {}) '.format(mat_idx+1, len(mat_filenames)) + mat_filename)
            tmp = sio.loadmat(mat_filename)
            input_ = tmp[mat_input_name]
            target = tmp[mat_target_name]

            group_name = 'group{:02}'.format(mat_idx)
            grp = f.create_group(group_name)

            grp.create_dataset(hdf5_input_name, data=input_, dtype='float32')
            grp.create_dataset(hdf5_target_name, data=target, dtype='uint8')
