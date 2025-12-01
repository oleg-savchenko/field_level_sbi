import ray
import zarr
import numpy as np
from torch.utils.data import IterableDataset
import torch
import time


@ray.remote(name='DatasetManager')
class DatasetManager:
    def __init__(self, filepath, shapes_and_dtypes, 
                 num_max_sims = None,   # TODO: Maximum number of simulations to store
                 num_min_sims = None,   # TODO: Minimum number of simulations to train on
                 num_val_sims = None,   # TODO: Number of sliding validation sims
                 num_resims = 256,
                 ):
        self.num_val_sims = num_val_sims
        self.num_min_sims = num_min_sims
        self.num_resims = num_resims
        # Initialize the Zarr store with given shapes
        self.filepath = filepath
        self.shapes_and_dtypes = shapes_and_dtypes
        self.store = zarr.open_group(filepath, mode='a')
        self.arrays = {}

        for key, shape_and_dtype in shapes_and_dtypes.items():
            if key not in self.store:
                shape, dtype = shape_and_dtype
                self.arrays[key] = self.store.create_dataset(
                    key,
                    shape=(0,) + shape,
                    chunks=(128,) + shape,
                    dtype=dtype,
                    overwrite=True,
                    append_dim=0
                )
            else:
                self.arrays[key] = self.store[key]

        # Create active column with boolen values
        if 'active' not in self.store:
            self.arrays['active'] = self.store.create_dataset(
                'active',
                shape=(0,),
                chunks=(128,),
                dtype='bool',
                overwrite=True,
                append_dim=0
            )
        else:
            self.arrays['active'] = self.store['active']

    def get_filepath(self):
        return self.filepath

    def get_num_min_sims(self):
        return self.num_min_sims

    def get_num_resims(self):
        return self.num_resims

    def append(self, data):
        # Append data to the datasets
        # TODO: Check that data has same length and all keys etc
        for key, value in data.items():
            if not key in self.arrays.keys():
                # TODO: Default behavior is to ignore unknown keys, this should issue a warning once
                continue
            if isinstance(value, torch.Tensor):
                value = value.numpy()
            try:
                self.arrays[key].append(value)
            except ValueError as e:
                array_shape = self.arrays[key].shape  # Get the current shape of the array in the store
                value_shape = value.shape  # Get the shape of the value being appended
                raise ValueError(
                    f"Error appending to array with key '{key}'. "
                    f"Original shape: {array_shape}, appended shape: {value_shape}. Original error: {e}"
            ) from e
        # Extend active column accordingly
        self.arrays['active'].append(np.ones(value.shape[0], dtype=bool))

    def get_length(self):
        # Return minimum length of all datasets
        lengths = [len(array) for array in self.arrays.values()]
        return min(lengths)

    def get_num_active(self):
        # Return number of active samples
        return np.sum(self.arrays['active'][:self.get_length()])

    def is_active(self):
        return self.arrays['active'][:self.get_length()]

    def get_active_ids(self):
        # Get indices of active samples
        return np.where(self.arrays['active'][:self.get_length()])[0]

    def deactivate(self, ids):
        # Deactivate samples by setting active column to False
        if len(ids) > 0:
            #print("Deactivating samples:", ids)
            self.arrays['active'][ids] = False

#    def get_dataset_views(self, keys, filter_train=None, filter_val=None, active_only=False):
#        slice_train = [-self.num_min_sims-1,-self.num_val_sims-1]
#        slice_val = [-self.num_val_sims-1,-1]
#        dataset_train = DatasetView(self.filepath, keys, filter=filter_train,
#            active_only=active_only, slice=slice_train)
#        dataset_val = DatasetView(self.filepath, keys, filter=filter_val,
#            active_only=active_only, slice=slice_val)
#        return dataset_train, dataset_val

    def get_train_dataset_view(self, keys, filter=None, active_only=False):
        slice = [-self.num_min_sims-1,-self.num_val_sims-1]
        dataset = DatasetView(self.filepath, keys, filter=filter,
            active_only=active_only, slice=slice)
        return dataset

    def get_val_dataset_view(self, keys, filter=None, active_only=False):
        slice = [-self.num_val_sims-1,-1]
        dataset_train = DatasetView(self.filepath, keys, filter=filter,
            active_only=active_only, slice=slice)
        return dataset_train

    def generate_samples(self, deployed_graph, num_sims):
        samples = deployed_graph.sample(num_sims)
        self.append(samples)

class DatasetView(IterableDataset):
    def __init__(self, filepath, keylist, filter=None, slice=None, active_only=False, pre_load=True):
        self.dataset_manager = ray.get_actor("DatasetManager")
        self.keylist = keylist
        self.store = zarr.open_group(filepath, mode='r')
        self.filter = filter
        self.slice = slice
        self.active_only = active_only
        self.pre_load = pre_load

    def __iter__(self):
        if self.active_only:
            ids = ray.get(self.dataset_manager.get_active_ids.remote())
        else:
            length = ray.get(self.dataset_manager.get_length.remote())
            ids = list(range(0, length))
        if self.pre_load:
            #print("Preloading dataset...")
            store = {k: self.store[k][:] for k in self.keylist}
            #print("...done preloading dataset")
        if self.slice is not None:
            ids = ids[self.slice[0]:self.slice[1]]
            print("Training range:", min(ids), max(ids))
        for index in ids:
            if self.pre_load:
                sample = [store[key][index] for key in self.keylist]
            else:
                sample = [self.store[key][index] for key in self.keylist]
            if self.filter is not None:  # Online evaluation
                sample = self.filter(sample)
            yield (index, *sample)

def get_zarr_dataset_manager(shapes_and_dtypes, filepath, num_min_sims=None,
                             num_val_sims=None, num_resims=0): #64
    #shapes_and_dtypes = deployed_graph.get_shapes_and_dtypes()
    dataset_manager = DatasetManager.remote(filepath, shapes_and_dtypes,
            num_min_sims=num_min_sims,
            num_val_sims=num_val_sims,
            num_resims=num_resims
    )
    return dataset_manager
