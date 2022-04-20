# CHARLIE : 
# A provisoire to implement https://github.com/NYUMedML/conditional_ssl_hist

from typing import List
import numpy as np
import pandas as pd
from pathlib import Path
import math
from typing import TypeVar, Optional, Iterator

import torch
from torch.utils.data import Sampler, Dataset
import torch.distributed as dist


T_co = TypeVar("T_co", covariant=True)


class CondSSLDistributedSampler(Sampler[T_co]):
    r"""Sampler that restricts data loading to a subset of the dataset.

    It is especially useful in conjunction with
    :class:`torch.nn.parallel.DistributedDataParallel`. In such a case, each
    process can pass a :class:`~torch.utils.data.DistributedSampler` instance as a
    :class:`~torch.utils.data.DataLoader` sampler, and load a subset of the
    original dataset that is exclusive to it.

    .. note::
        Dataset is assumed to be of constant size and that any instance of it always
        returns the same elements in the same order.

    Args:
        dataset: Dataset used for sampling.
        num_replicas (int, optional): Number of processes participating in
            distributed training. By default, :attr:`world_size` is retrieved from the
            current distributed group.
        rank (int, optional): Rank of the current process within :attr:`num_replicas`.
            By default, :attr:`rank` is retrieved from the current distributed
            group.
        shuffle (bool, optional): If ``True`` (default), sampler will shuffle the
            indices.
        seed (int, optional): random seed used to shuffle the sampler if
            :attr:`shuffle=True`. This number should be identical across all
            processes in the distributed group. Default: ``0``.
        drop_last (bool, optional): if ``True``, then the sampler will drop the
            tail of the data to make it evenly divisible across the number of
            replicas. If ``False``, the sampler will add extra indices to make
            the data evenly divisible across the replicas. Default: ``False``.

    .. warning::
        In distributed mode, calling the :meth:`set_epoch` method at
        the beginning of each epoch **before** creating the :class:`DataLoader` iterator
        is necessary to make shuffling work properly across multiple epochs. Otherwise,
        the same ordering will be always used.

    Example::

        >>> sampler = DistributedSampler(dataset) if is_distributed else None
        >>> loader = DataLoader(dataset, shuffle=(sampler is None),
        ...                     sampler=sampler)
        >>> for epoch in range(start_epoch, n_epochs):
        ...     if is_distributed:
        ...         sampler.set_epoch(epoch)
        ...     train(loader)
    """

    def __init__(
        self,
        dataset: Dataset,
        batch_size: int,
        n_slides_per_batch: int,
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
        shuffle: bool = True,
        seed: int = 0,
        drop_last: bool = False,
    ) -> None:
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        if rank >= num_replicas or rank < 0:
            raise ValueError(
                "Invalid rank {}, rank should be in the interval" " [0, {}]".format(rank, num_replicas - 1)
            )
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank  # CHARLIE : must be global rank, i.e. from 0 to 15 if using 16 gpus in total
        self.epoch = 0
        self.drop_last = drop_last

        # CHARLIE : get filenames to sort later based on slidename
        self.filenames = self.dataset.get_image_paths()[0]
        self.filenames = [Path(f) for f in self.filenames]
        # self.slidenames = [f.split("_")[4].replace(".png", "") for f in self.filenames]
        # CHARLIE : for Imagenet only
        self.slidenames = [Path(f).parents[0].name for f in self.filenames]
        self.batch_size = batch_size  # CHARLIE: this is batchsize per replica ! i.e. if 16 gpus, total batchsize is batchsize * 16
        assert batch_size % n_slides_per_batch == 0
        self.n_slides_per_batch = n_slides_per_batch
        self.n_tiles_per_slide = (batch_size * num_replicas) // n_slides_per_batch

        # CHARLIE : number of samples is less than the size of the dataset
        # This assumes that every slides has at least n_tiles_per_slide tiles
        # very dirty
        df_samples = pd.DataFrame(
            {
                "filename": self.filenames,
                "slidename": self.slidenames,
            }
        )
        dict_samples = {s: df_samples[df_samples["slidename"] == s] for s in df_samples["slidename"].unique()}
        min_n_tiles_slide = np.min([len(dict_samples[s]) for s in dict_samples])
        
        self.real_len_dataset = (min_n_tiles_slide // self.n_tiles_per_slide) * self.n_tiles_per_slide * len(dict_samples)
        if self.drop_last and len(self.dataset) % self.num_replicas != 0:  # type: ignore[arg-type]
            # Split to nearest available length that is evenly divisible.
            # This is to ensure each rank receives the same amount of data when
            # using this Sampler.
            self.num_samples = math.ceil(
                    (self.real_len_dataset - self.num_replicas) / self.num_replicas  # type: ignore[arg-type]
                    )
        else:
            self.num_samples = math.ceil(self.real_len_dataset / self.num_replicas)  # type: ignore[arg-type]
        # self.num_samples = math.ceil(real_len_dataset / self.num_replicas)

        self.total_size = self.num_samples * self.num_replicas
        self.shuffle = shuffle
        self.seed = seed

    def __iter__(self) -> Iterator[T_co]:

        if self.shuffle:
            # deterministically shuffle based on epoch and seed
            # g = torch.Generator()
            # g.manual_seed(self.seed + self.epoch)
            # indices = torch.randperm(len(self.dataset), generator=g).tolist()  # type: ignore[arg-type]
            df_samples = pd.DataFrame(
                {
                    "filename": self.filenames,
                    "slidename": self.slidenames,
                }
            )
            df_samples = df_samples.sample(frac=1, random_state=self.seed + self.epoch)
            dict_samples = {s: df_samples[df_samples["slidename"] == s] for s in df_samples["slidename"].unique()}
            min_n_tiles_slide = np.min([len(dict_samples[s]) for s in dict_samples])
            
            indices = []
            for i in range(min_n_tiles_slide // self.n_tiles_per_slide):
                for s in dict_samples:
                    indices_ = dict_samples[s].iloc[:self.n_tiles_per_slide].index.tolist()
                    indices += indices_
                    # Remove the already sampled indices
                    mask = ~dict_samples[s].index.isin(indices_)
                    dict_samples[s] = dict_samples[s][mask]

        else:
            raise NotImplementedError

        if not self.drop_last:
            # add extra samples to make it evenly divisible
            padding_size = self.total_size - len(indices)
            if padding_size <= len(indices):
                indices += indices[:padding_size]
            else:
                indices += (indices * math.ceil(padding_size / len(indices)))[:padding_size]
        else:
            # remove tail of data to make it evenly divisible.
            indices = indices[: self.total_size]
        assert len(indices) == self.total_size, f"{len(indices)}, {self.total_size}"

        # subsample
        indices = indices[self.rank:self.total_size:self.num_replicas]
        assert len(indices) == self.num_samples

        return iter(indices)

    def __len__(self) -> int:
        return self.num_samples

    def set_epoch(self, epoch: int) -> None:
        r"""
        Sets the epoch for this sampler. When :attr:`shuffle=True`, this ensures all replicas
        use a different random ordering for each epoch. Otherwise, the next iteration of this
        sampler will yield the same ordering.

        Args:
            epoch (int): Epoch number.
        """
        self.epoch = epoch


