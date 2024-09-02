# Utils

import numbers
import time
import warnings
from collections import OrderedDict
from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np
import torch
from addict import Dict
from PIL import Image
from torchvision.transforms.functional import rotate
from tqdm import tqdm

torch.set_default_tensor_type("torch.cuda.FloatTensor")

import os

os.environ["FFMPEG_BINARY"] = "ffmpeg"
import moviepy.editor as mvp
from IPython.display import HTML, clear_output, display
from moviepy.video.io.ffmpeg_writer import FFMPEG_VideoWriter


class VideoWriter:
    def __init__(self, filename, fps=30.0, **kw):
        self.writer = None
        self.params = dict(filename=filename, fps=fps, **kw)

    def add(self, img):
        img = np.asarray(img)
        if self.writer is None:
            h, w = img.shape[:2]
            self.writer = FFMPEG_VideoWriter(size=(w, h), **self.params)
        if img.dtype in [np.float32, np.float64]:
            img = np.uint8(img.clip(0, 1) * 255)
        if len(img.shape) == 2:
            img = np.repeat(img[..., None], 3, -1)
        self.writer.write_frame(img)

    def close(self):
        if self.writer:
            self.writer.close()

    def __enter__(self):
        return self

    def __exit__(self, *kw):
        self.close()

    def show(self, **kw):
        self.close()
        fn = self.params["filename"]
        display(mvp.ipython_display(fn, **kw))


def complex_mult_torch(X, Y):
    """Complex multiplication in Pytorch when the tensor last dimension is 2, with dim 0 being the real component and 1 the imaginary one"""
    assert X.shape[-1] == 2 and Y.shape[-1] == 2, "Last dimension must be 2"
    return torch.stack(
        (
            X[..., 0] * Y[..., 0] - X[..., 1] * Y[..., 1],
            X[..., 0] * Y[..., 1] + X[..., 1] * Y[..., 0],
        ),
        dim=-1,
    )


def roll_n(X, axis, n):
    """Rolls a tensor with a shift n on the specified axis"""
    f_idx = tuple(
        slice(None, None, None) if i != axis else slice(0, n, None)
        for i in range(X.dim())
    )
    b_idx = tuple(
        slice(None, None, None) if i != axis else slice(n, None, None)
        for i in range(X.dim())
    )
    front = X[f_idx]
    back = X[b_idx]
    return torch.cat([back, front], axis)


class Space(object):
    """
    Defines the init_space, genome_space and intervention_space of a system
    """

    def __init__(self, shape=None, dtype=None):
        self.shape = None if shape is None else tuple(shape)
        self.dtype = dtype

    def sample(self):
        """
        Randomly sample an element of this space.
        Can be uniform or non-uniform sampling based on boundedness of space."""
        raise NotImplementedError

    def mutate(self, x):
        """
        Randomly mutate an element of this space.
        """
        raise NotImplementedError

    def contains(self, x):
        """
        Return boolean specifying if x is a valid
        member of this space
        """
        raise NotImplementedError

    def clamp(self, x):
        """
        Return a valid clamped value of x inside space's bounds
        """
        raise NotImplementedError

    def __contains__(self, x):
        return self.contains(x)


class DiscreteSpace(Space):
    r"""A discrete space in :math:`\{ 0, 1, \\dots, n-1 \}`.
    /!\ mutation is gaussian by default: please create custom space inheriting from discrete space for custom mutation functions

    Example::

        >>> DiscreteSpace(2)

    """

    def __init__(self, n, mutation_mean=0.0, mutation_std=1.0, indpb=1.0):
        assert n >= 0
        self.n = n

        # mutation_mean: mean for the gaussian addition mutation
        # mutation_std: std for the gaussian addition mutation
        # indpb – independent probability for each attribute to be mutated.
        self.mutation_mean = torch.as_tensor(mutation_mean, dtype=torch.float64)
        self.mutation_std = torch.as_tensor(mutation_std, dtype=torch.float64)
        self.indpb = torch.as_tensor(indpb, dtype=torch.float64)
        super(DiscreteSpace, self).__init__((), torch.int64)

    def sample(self):
        return torch.randint(self.n, ())

    def mutate(self, x):
        mutate_mask = torch.rand(self.shape) < self.indpb
        noise = torch.normal(self.mutation_mean, self.mutation_std, ())
        x = x.type(torch.float64) + mutate_mask * noise
        x = torch.floor(x).type(self.dtype)
        if not self.contains(x):
            return self.clamp(x)
        else:
            return x

    def contains(self, x):
        if isinstance(x, int):
            as_int = x
        elif not x.dtype.is_floating_point and (x.shape == ()):  # integer or size 0
            as_int = int(x)
        else:
            return False
        return 0 <= as_int < self.n

    def clamp(self, x):
        x = torch.max(x, torch.as_tensor(0, dtype=self.dtype, device=x.device))
        x = torch.min(x, torch.as_tensor(self.n - 1, dtype=self.dtype, device=x.device))
        return x

    def __repr__(self):
        return "DiscreteSpace(%d)" % self.n

    def __eq__(self, other):
        return isinstance(other, DiscreteSpace) and self.n == other.n


class BoxSpace(Space):
    """
    A (possibly unbounded) box in R^n. Specifically, a Box represents the
    Cartesian product of n closed intervals. Each interval has the form of one
    of [a, b], (-oo, b], [a, oo), or (-oo, oo).

    There are two common use cases:

    * Identical bound for each dimension::
        >>> BoxSpace(low=-1.0, high=2.0, shape=(3, 4), dtype=torch.float32)
        Box(3, 4)

    * Independent bound for each dimension::
        >>> BoxSpace(low=torch.tensor([-1.0, -2.0]), high=torch.tensor([2.0, 4.0]), dtype=torch.float32)
        Box(2,)

    """

    def __init__(
        self,
        low,
        high,
        shape=None,
        dtype=torch.float32,
        mutation_mean=0.0,
        mutation_std=1.0,
        indpb=1.0,
    ):
        assert dtype is not None, "dtype must be explicitly provided. "
        self.dtype = dtype

        # determine shape if it isn't provided directly
        if shape is not None:
            shape = tuple(shape)
            assert (
                isinstance(low, numbers.Number) or low.shape == shape
            ), "low.shape doesn't match provided shape"
            assert (
                isinstance(high, numbers.Number) or high.shape == shape
            ), "high.shape doesn't match provided shape"
        elif not isinstance(low, numbers.Number):
            shape = low.shape
            assert (
                isinstance(high, numbers.Number) or high.shape == shape
            ), "high.shape doesn't match low.shape"
        elif not isinstance(high, numbers.Number):
            shape = high.shape
            assert (
                isinstance(low, numbers.Number) or low.shape == shape
            ), "low.shape doesn't match high.shape"
        else:
            raise ValueError(
                "shape must be provided or inferred from the shapes of low or high"
            )

        if isinstance(low, numbers.Number):
            low = torch.full(shape, low, dtype=dtype)

        if isinstance(high, numbers.Number):
            high = torch.full(shape, high, dtype=dtype)

        self.shape = shape
        self.low = low.type(self.dtype)
        self.high = high.type(self.dtype)

        # Boolean arrays which indicate the interval type for each coordinate
        self.bounded_below = ~torch.isneginf(self.low)
        self.bounded_above = ~torch.isposinf(self.high)

        # mutation_mean: mean for the gaussian addition mutation
        # mutation_std: std for the gaussian addition mutation
        # indpb – independent probability for each attribute to be mutated.
        if isinstance(mutation_mean, numbers.Number):
            mutation_mean = torch.full(self.shape, mutation_mean, dtype=torch.float64)
        self.mutation_mean = torch.as_tensor(mutation_mean, dtype=torch.float64)
        if isinstance(mutation_std, numbers.Number):
            mutation_std = torch.full(self.shape, mutation_std, dtype=torch.float64)
        self.mutation_std = torch.as_tensor(mutation_std, dtype=torch.float64)
        if isinstance(indpb, numbers.Number):
            indpb = torch.full(self.shape, indpb, dtype=torch.float64)
        self.indpb = torch.as_tensor(indpb, dtype=torch.float64)

        super(BoxSpace, self).__init__(self.shape, self.dtype)

    def is_bounded(self, manner="both"):
        below = torch.all(self.bounded_below)
        above = torch.all(self.bounded_above)
        if manner == "both":
            return below and above
        elif manner == "below":
            return below
        elif manner == "above":
            return above
        else:
            raise ValueError("manner is not in {'below', 'above', 'both'}")

    def sample(self):
        """
        Generates a single random sample inside of the Box.

        In creating a sample of the box, each coordinate is sampled according to
        the form of the interval:

        * [a, b] : uniform distribution
        * [a, oo) : shifted exponential distribution
        * (-oo, b] : shifted negative exponential distribution
        * (-oo, oo) : normal distribution
        """
        high = (
            self.high.type(torch.float64)
            if self.dtype.is_floating_point
            else self.high.type(torch.int64) + 1
        )
        sample = torch.empty(self.shape, dtype=torch.float64)

        # Masking arrays which classify the coordinates according to interval
        # type
        unbounded = ~self.bounded_below & ~self.bounded_above
        upp_bounded = ~self.bounded_below & self.bounded_above
        low_bounded = self.bounded_below & ~self.bounded_above
        bounded = self.bounded_below & self.bounded_above

        # Vectorized sampling by interval type
        sample[unbounded] = torch.randn(unbounded[unbounded].shape, dtype=torch.float64)

        sample[low_bounded] = (
            -torch.rand(low_bounded[low_bounded].shape, dtype=torch.float64)
        ).exponential_() + self.low[low_bounded]

        sample[upp_bounded] = (
            self.high[upp_bounded]
            - (
                -torch.rand(upp_bounded[upp_bounded].shape, dtype=torch.float64)
            ).exponential_()
        )

        sample[bounded] = (self.low[bounded] - high[bounded]) * torch.rand(
            bounded[bounded].shape, dtype=torch.float64
        ) + high[bounded]

        if not self.dtype.is_floating_point:  # integer
            sample = torch.floor(sample)

        return sample.type(self.dtype)

    def mutate(self, x, mask=None):
        if mask == None:
            mask = torch.ones(x.shape).to(x.device)

        mutate_mask = mask * (
            (torch.rand(self.shape) < self.indpb).type(torch.float64)
        ).to(x.device)
        noise = torch.normal(self.mutation_mean, self.mutation_std).to(x.device)
        x = x.type(torch.float64) + mutate_mask * noise
        if not self.dtype.is_floating_point:  # integer
            x = torch.floor(x)
        x = x.type(self.dtype)
        if not self.contains(x):
            return self.clamp(x)
        else:
            return x

    def contains(self, x):
        if isinstance(x, list):
            x = torch.tensor(x)  # Promote list to array for contains check
        return (
            x.shape == self.shape
            and torch.all(
                x >= torch.as_tensor(self.low, dtype=self.dtype, device=x.device)
            )
            and torch.all(
                x <= torch.as_tensor(self.high, dtype=self.dtype, device=x.device)
            )
        )

    def clamp(self, x):
        if self.is_bounded(manner="below"):
            x = torch.max(
                x, torch.as_tensor(self.low, dtype=self.dtype, device=x.device)
            )
        if self.is_bounded(manner="above"):
            x = torch.min(
                x, torch.as_tensor(self.high, dtype=self.dtype, device=x.device)
            )
        return x

    def __repr__(self):
        return "BoxSpace({}, {}, {}, {})".format(
            self.low.min(), self.high.max(), self.shape, self.dtype
        )

    def __eq__(self, other):
        return (
            isinstance(other, BoxSpace)
            and (self.shape == other.shape)
            and torch.allclose(self.low, other.low)
            and torch.allclose(self.high, other.high)
        )


class DictSpace(Space):
    """
    A Dict dictionary of simpler spaces.

    Example usage:
    self.genome_space = spaces.DictSpace({"position": spaces.Discrete(2), "velocity": spaces.Discrete(3)})

    Example usage [nested]:
    self.nested_genome_space = spaces.DictSpace({
        'sensors':  spaces.DictSpace({
            'position': spaces.Box(low=-100, high=100, shape=(3,)),
            'velocity': spaces.Box(low=-1, high=1, shape=(3,)),
            'front_cam': spaces.Tuple((
                spaces.Box(low=0, high=1, shape=(10, 10, 3)),
                spaces.Box(low=0, high=1, shape=(10, 10, 3))
            )),
            'rear_cam': spaces.Box(low=0, high=1, shape=(10, 10, 3)),
        }),
        'ext_controller': spaces.MultiDiscrete((5, 2, 2)),
        'inner_state':spaces.DictSpace({
            'charge': spaces.Discrete(100),
            'system_checks': spaces.MultiBinary(10),
            'job_status': spaces.DictSpace({
                'task': spaces.Discrete(5),
                'progress': spaces.Box(low=0, high=100, shape=()),
            })
        })
    })
    """

    def __init__(self, spaces=None, **spaces_kwargs):
        assert (spaces is None) or (
            not spaces_kwargs
        ), "Use either DictSpace(spaces=dict(...)) or DictSpace(foo=x, bar=z)"
        if spaces is None:
            spaces = spaces_kwargs
        if isinstance(spaces, list):
            spaces = Dict(spaces)
        self.spaces = spaces
        for space in spaces.values():
            assert isinstance(
                space, Space
            ), "Values of the attrdict should be instances of gym.Space"
        Space.__init__(
            self, None, None
        )  # None for shape and dtype, since it'll require special handling

    def sample(self):
        return Dict([(k, space.sample()) for k, space in self.spaces.items()])

    def mutate(self, x):
        return Dict([(k, space.mutate(x[k])) for k, space in self.spaces.items()])

    def contains(self, x):
        if not isinstance(x, dict) or len(x) != len(self.spaces):
            return False
        for k, space in self.spaces.items():
            if k not in x:
                return False
            if not space.contains(x[k]):
                return False
        return True

    def clamp(self, x):
        return Dict([(k, space.clamp(x[k])) for k, space in self.spaces.items()])

    def __getitem__(self, key):
        return self.spaces[key]

    def __iter__(self):
        for key in self.spaces:
            yield key

    def __repr__(self):
        return (
            "DictSpace("
            + ", ".join([str(k) + ":" + str(s) for k, s in self.spaces.items()])
            + ")"
        )

    def __eq__(self, other):
        return isinstance(other, DictSpace) and self.spaces == other.spaces


class MultiDiscreteSpace(Space):
    """
    - The multi-discrete space consists of a series of discrete spaces with different number of possible instances in eachs
    - Can be initialized as

        MultiDiscreteSpace([ 5, 2, 2 ])

    """

    def __init__(self, nvec, mutation_mean=0.0, mutation_std=1.0, indpb=1.0):
        """
        nvec: vector of counts of each categorical variable
        """
        assert (torch.tensor(nvec) > 0).all(), "nvec (counts) have to be positive"
        self.nvec = torch.as_tensor(nvec, dtype=torch.int64)
        self.mutation_std = mutation_std

        # mutation_mean: mean for the gaussian addition mutation
        # mutation_std: std for the gaussian addition mutation
        # indpb – independent probability for each attribute to be mutated.
        if isinstance(mutation_mean, numbers.Number):
            mutation_mean = torch.full(
                self.nvec.shape, mutation_mean, dtype=torch.float64
            )
        self.mutation_mean = torch.as_tensor(mutation_mean, dtype=torch.float64)
        if isinstance(mutation_std, numbers.Number):
            mutation_std = torch.full(
                self.nvec.shape, mutation_std, dtype=torch.float64
            )
        self.mutation_std = torch.as_tensor(mutation_std, dtype=torch.float64)
        if isinstance(indpb, numbers.Number):
            indpb = torch.full(self.nvec.shape, indpb, dtype=torch.float64)
        self.indpb = torch.as_tensor(indpb, dtype=torch.float64)

        super(MultiDiscreteSpace, self).__init__(self.nvec.shape, torch.int64)

    def sample(self):
        return (torch.rand(self.nvec.shape) * self.nvec).type(self.dtype)

    def mutate(self, x):
        mutate_mask = (torch.rand(self.shape) < self.indpb).to(x.device)
        noise = torch.normal(self.mutation_mean, self.mutation_std).to(x.device)
        x = x.type(torch.float64) + mutate_mask * noise
        x = torch.floor(x).type(self.dtype)
        if not self.contains(x):
            return self.clamp(x)
        else:
            return x

    def contains(self, x):
        if isinstance(x, list):
            x = torch.tensor(x)  # Promote list to array for contains check
        # if nvec is uint32 and space dtype is uint32, then 0 <= x < self.nvec guarantees that x
        # is within correct bounds for space dtype (even though x does not have to be unsigned)
        return x.shape == self.shape and (0 <= x).all() and (x < self.nvec).all()

    def clamp(self, x):
        x = torch.max(x, torch.as_tensor(0, dtype=self.dtype, device=x.device))
        x = torch.min(
            x, torch.as_tensor(self.nvec - 1, dtype=self.dtype, device=x.device)
        )
        return x

    def __repr__(self):
        return "MultiDiscreteSpace({})".format(self.nvec)

    def __eq__(self, other):
        return isinstance(other, MultiDiscreteSpace) and torch.all(
            self.nvec == other.nvec
        )


class MultiDiscreteSpace(Space):
    """
    - The multi-discrete space consists of a series of discrete spaces with different number of possible instances in eachs
    - Can be initialized as

        MultiDiscreteSpace([ 5, 2, 2 ])

    """

    def __init__(self, nvec, mutation_mean=0.0, mutation_std=1.0, indpb=1.0):
        """
        nvec: vector of counts of each categorical variable
        """
        assert (torch.tensor(nvec) > 0).all(), "nvec (counts) have to be positive"
        self.nvec = torch.as_tensor(nvec, dtype=torch.int64)
        self.mutation_std = mutation_std

        # mutation_mean: mean for the gaussian addition mutation
        # mutation_std: std for the gaussian addition mutation
        # indpb – independent probability for each attribute to be mutated.
        if isinstance(mutation_mean, numbers.Number):
            mutation_mean = torch.full(
                self.nvec.shape, mutation_mean, dtype=torch.float64
            )
        self.mutation_mean = torch.as_tensor(mutation_mean, dtype=torch.float64)
        if isinstance(mutation_std, numbers.Number):
            mutation_std = torch.full(
                self.nvec.shape, mutation_std, dtype=torch.float64
            )
        self.mutation_std = torch.as_tensor(mutation_std, dtype=torch.float64)
        if isinstance(indpb, numbers.Number):
            indpb = torch.full(self.nvec.shape, indpb, dtype=torch.float64)
        self.indpb = torch.as_tensor(indpb, dtype=torch.float64)

        super(MultiDiscreteSpace, self).__init__(self.nvec.shape, torch.int64)

    def sample(self):
        return (torch.rand(self.nvec.shape) * self.nvec).type(self.dtype)

    def mutate(self, x):
        mutate_mask = (torch.rand(self.shape) < self.indpb).to(x.device)
        noise = torch.normal(self.mutation_mean, self.mutation_std).to(x.device)
        x = x.type(torch.float64) + mutate_mask * noise
        x = torch.floor(x).type(self.dtype)
        if not self.contains(x):
            return self.clamp(x)
        else:
            return x

    def contains(self, x):
        if isinstance(x, list):
            x = torch.tensor(x)  # Promote list to array for contains check
        # if nvec is uint32 and space dtype is uint32, then 0 <= x < self.nvec guarantees that x
        # is within correct bounds for space dtype (even though x does not have to be unsigned)
        return x.shape == self.shape and (0 <= x).all() and (x < self.nvec).all()

    def clamp(self, x):
        x = torch.max(x, torch.as_tensor(0, dtype=self.dtype, device=x.device))
        x = torch.min(
            x, torch.as_tensor(self.nvec - 1, dtype=self.dtype, device=x.device)
        )
        return x

    def __repr__(self):
        return "MultiDiscreteSpace({})".format(self.nvec)

    def __eq__(self, other):
        return isinstance(other, MultiDiscreteSpace) and torch.all(
            self.nvec == other.nvec
        )


# Exploration database


class RunDataEntry(Dict):
    """
    Class that specify for RunData entry in the DB
    """

    def __init__(self, db, id, policy_parameters, observations, **kwargs):
        """
        :param kwargs: flexible structure of the entry which might contain additional columns (eg: source_policy_idx, target_goal, etc.)
        """
        super().__init__(**kwargs)
        self.db = db
        self.id = id
        self.policy_parameters = policy_parameters
        self.observations = observations


class ExplorationDB:
    """
    Base of all Database classes.
    """

    @staticmethod
    def default_config():

        default_config = Dict()
        default_config.db_directory = "database"
        default_config.save_observations = True
        default_config.keep_saved_runs_in_memory = True
        default_config.memory_size_run_data = "infinity"  # number of runs that are kept in memory: 'infinity' - no imposed limit, int - number of runs saved in memory
        default_config.load_observations = (
            True  # if set to false observations are not loaded in the load() function
        )

        return default_config

    def __init__(self, config={}, **kwargs):

        self.config = self.__class__.default_config()
        self.config.update(config)
        self.config.update(kwargs)

        if self.config.memory_size_run_data != "infinity":
            assert (
                isinstance(self.config.memory_size_run_data, int)
                and self.config.memory_size_run_data > 0
            ), "config.memory_size_run_data must be set to infinity or to an integer >= 1"

        self.reset_empty_db()

    def reset_empty_db(self):
        self.runs = OrderedDict()
        self.run_ids = set()  # list with run_ids that exist in the db
        self.run_data_ids_in_memory = []  # list of run_ids that are hold in memory

    def add_run_data(self, id, policy_parameters, observations, **kwargs):

        run_data_entry = RunDataEntry(
            db=self,
            id=id,
            policy_parameters=policy_parameters,
            observations=observations,
            **kwargs,
        )
        if id not in self.run_ids:
            self.add_run_data_to_memory(id, run_data_entry)
            self.run_ids.add(id)

        else:
            warnings.warn(
                f"/!\ id {id} already in the database: overwriting it with new run data !!!"
            )
            self.add_run_data_to_memory(id, run_data_entry, replace_existing=True)

        self.save(
            [id]
        )  # TODO: modify if we do not want to automatically save after each run

    def add_run_data_to_memory(self, id, run_data, replace_existing=False):
        self.runs[id] = run_data
        if not replace_existing:
            self.run_data_ids_in_memory.insert(0, id)

        # remove last item from memory when not enough size
        if (
            self.config.memory_size_run_data != "infinity"
            and len(self.run_data_ids_in_memory) > self.config.memory_size_run_data
        ):
            del self.runs[self.run_data_ids_in_memory[-1]]
            del self.run_data_ids_in_memory[-1]

    def save(self, run_ids=None):
        # the run data entry is save in 2 files: 'run_*_data*' (general data dict such as run parameters -> for now json) and ''run_*_observations*' (observation data dict -> for now npz)
        if run_ids is None:
            run_ids = []

        for run_id in run_ids:
            self.save_run_data_to_db(run_id)
            if self.config.save_observations:
                self.save_observations_to_db(run_id)

        if not self.config.keep_saved_runs_in_memory:
            for run_id in run_ids:
                del self.runs[run_id]
            self.run_data_ids_in_memory = []

    def save_run_data_to_db(self, run_id):
        run_data = self.runs[run_id]

        # add all data besides the observations
        save_dict = dict()
        for data_name, data_value in run_data.items():
            if data_name not in ["observations", "db"]:
                save_dict[data_name] = data_value
        filename = "run_{:07d}_data.pickle".format(run_id)
        filepath = os.path.join(self.config.db_directory, filename)

        torch.save(save_dict, filepath)

    def save_observations_to_db(self, run_id):
        run_data = self.runs[run_id]

        filename = "run_{:07d}_observations.pickle".format(run_id)
        filepath = os.path.join(self.config.db_directory, filename)

        torch.save(run_data.observations, filepath)

    def load(self, run_ids=None, map_location="cpu"):
        """
        Loads the data base.
        :param run_ids:  IDs of runs for which the data should be loaded into the memory.
                        If None is given, all ids are loaded (up to the allowed memory size).
        :param map_location: device on which the database is loaded
        """

        if run_ids is not None:
            assert isinstance(run_ids, list), "run_ids must be None or a list"

        # set run_ids from the db directory and empty memory
        self.run_ids = self.load_run_ids_from_db()
        self.runs = OrderedDict()
        self.run_data_ids_in_memory = []

        if run_ids is None:
            run_ids = self.run_ids

        if len(run_ids) > 0:

            if (
                self.config.memory_size_run_data != "infinity"
                and len(run_ids) > self.config.memory_size_run_data
            ):
                # only load the maximum number of run_data into the memory
                run_ids = list(run_ids)[-self.config.memory_size_run_data :]

            self.load_run_data_from_db(run_ids, map_location=map_location)

    def load_run_ids_from_db(self):
        run_ids = set()

        file_matches = glob(os.path.join(self.config.db_directory, "run_*_data*"))
        for match in file_matches:
            id_as_str = re.findall("_(\d+).", match)
            if len(id_as_str) > 0:
                run_ids.add(
                    int(id_as_str[-1])
                )  # use the last find, because ther could be more number in the filepath, such as in a directory name

        return run_ids

    def load_run_data_from_db(self, run_ids, map_location="cpu"):
        """Loads the data for a list of runs and adds them to the memory."""

        if not os.path.exists(self.config.db_directory):
            raise Exception(
                "The directory {!r} does not exits! Cannot load data.".format(
                    self.config.db_directory
                )
            )

        print("Loading Data: ")
        for run_id in tqdm(run_ids):
            # load general data (run parameters and others)
            filename = "run_{:07d}_data.pickle".format(run_id)
            filepath = os.path.join(self.config.db_directory, filename)

            if os.path.exists(filepath):
                run_data_kwargs = torch.load(filepath, map_location=map_location)
            else:
                run_data_kwargs = {"id": None, "policy_parameters": None}

            if self.config.load_observations:
                filename_obs = "run_{:07d}_observations.pickle".format(run_id)
                filepath_obs = os.path.join(self.config.db_directory, filename_obs)

                # load observations
                if os.path.exists(filepath_obs):
                    observations = torch.load(filepath_obs, map_location=map_location)
                else:
                    observations = None
            else:
                observations = None

            # create run data and add it to memory
            run_data = RunDataEntry(self, observations=observations, **run_data_kwargs)
            self.add_run_data_to_memory(run_id, run_data)

            if not self.config.keep_saved_runs_in_memory:
                del self.runs[run_id]
                del self.run_data_ids_in_memory[0]

        return


# Lenia System


class LeniaInitializationSpace(DictSpace):
    """Class for initialization space that allows to sample and clip the initialization"""

    @staticmethod
    def default_config():
        default_config = Dict()
        default_config.neat_config = None
        default_config.cppn_n_passes = 2
        return default_config

    def __init__(self, init_size=40, config={}, **kwargs):
        self.config = self.__class__.default_config()
        self.config.update(config)
        self.config.update(kwargs)

        spaces = Dict(
            # cppn_genome = LeniaCPPNInitSpace(self.config)
            init=BoxSpace(
                low=0.0,
                high=1.0,
                shape=(init_size, init_size),
                mutation_mean=torch.zeros((40, 40)),
                mutation_std=torch.ones((40, 40)) * 0.01,
                indpb=0.0,
                dtype=torch.float32,
            )
        )

        DictSpace.__init__(self, spaces=spaces)


""" =============================================================================================
Lenia Update Rule Space:
============================================================================================= """


class LeniaUpdateRuleSpace(DictSpace):
    """Space associated to the parameters of the update rule"""

    @staticmethod
    def default_config():
        default_config = Dict()
        return default_config

    def __init__(self, nb_k=10, config={}, **kwargs):
        self.config = self.__class__.default_config()
        self.config.update(config)
        self.config.update(kwargs)

        spaces = Dict(
            R=DiscreteSpace(n=25, mutation_mean=0.0, mutation_std=0.01, indpb=0.01),
            c0=MultiDiscreteSpace(
                nvec=[1] * nb_k,
                mutation_mean=torch.zeros((nb_k,)),
                mutation_std=0.1 * torch.ones((nb_k,)),
                indpb=0.1,
            ),
            c1=MultiDiscreteSpace(
                nvec=[1] * nb_k,
                mutation_mean=torch.zeros((nb_k,)),
                mutation_std=0.1 * torch.ones((nb_k,)),
                indpb=0.1,
            ),
            T=BoxSpace(
                low=1.0,
                high=10.0,
                shape=(),
                mutation_mean=0.0,
                mutation_std=0.1,
                indpb=0.01,
                dtype=torch.float32,
            ),
            rk=BoxSpace(
                low=0,
                high=1,
                shape=(nb_k, 3),
                mutation_mean=torch.zeros((nb_k, 3)),
                mutation_std=0.2 * torch.ones((nb_k, 3)),
                indpb=1,
                dtype=torch.float32,
            ),
            b=BoxSpace(
                low=0.0,
                high=1.0,
                shape=(nb_k, 3),
                mutation_mean=torch.zeros((nb_k, 3)),
                mutation_std=0.2 * torch.ones((nb_k, 3)),
                indpb=1,
                dtype=torch.float32,
            ),
            w=BoxSpace(
                low=0.01,
                high=0.5,
                shape=(nb_k, 3),
                mutation_mean=torch.zeros((nb_k, 3)),
                mutation_std=0.2 * torch.ones((nb_k, 3)),
                indpb=1,
                dtype=torch.float32,
            ),
            m=BoxSpace(
                low=0.05,
                high=0.5,
                shape=(nb_k,),
                mutation_mean=torch.zeros((nb_k,)),
                mutation_std=0.2 * torch.ones((nb_k,)),
                indpb=1,
                dtype=torch.float32,
            ),
            s=BoxSpace(
                low=0.001,
                high=0.18,
                shape=(nb_k,),
                mutation_mean=torch.zeros((nb_k,)),
                mutation_std=0.01 ** torch.ones((nb_k,)),
                indpb=0.1,
                dtype=torch.float32,
            ),
            h=BoxSpace(
                low=0,
                high=1.0,
                shape=(nb_k,),
                mutation_mean=torch.zeros((nb_k,)),
                mutation_std=0.2 * torch.ones((nb_k,)),
                indpb=0.1,
                dtype=torch.float32,
            ),
            r=BoxSpace(
                low=0.2,
                high=1.0,
                shape=(nb_k,),
                mutation_mean=torch.zeros((nb_k,)),
                mutation_std=0.2 * torch.ones((nb_k,)),
                indpb=1,
                dtype=torch.float32,
            ),
            # kn = DiscreteSpace(n=4, mutation_mean=0.0, mutation_std=0.1, indpb=1.0),
            # gn = DiscreteSpace(n=3, mutation_mean=0.0, mutation_std=0.1, indpb=1.0),
        )

        DictSpace.__init__(self, spaces=spaces)

    def mutate(self, x):
        mask = (x["s"] > 0.04).float() * (
            torch.rand(x["s"].shape[0]) < 0.25
        ).float().to(x["s"].device)
        param = []
        for k, space in self.spaces.items():
            if k == "R" or k == "c0" or k == "c1" or k == "T":
                param.append((k, space.mutate(x[k])))
            elif k == "rk" or k == "w" or k == "b":
                param.append((k, space.mutate(x[k], mask.unsqueeze(-1))))
            else:
                param.append((k, space.mutate(x[k], mask)))

        return Dict(param)


""" =============================================================================================
Lenia Main
============================================================================================= """

bell = lambda x, m, s: torch.exp(-(((x - m) / s) ** 2) / 2)
# Lenia family of functions for the kernel K and for the growth mapping g
kernel_core = {
    0: lambda u: (4 * u * (1 - u)) ** 4,  # polynomial (quad4)
    1: lambda u: torch.exp(
        4 - 1 / (u * (1 - u))
    ),  # exponential / gaussian bump (bump4)
    2: lambda u, q=1 / 4: (u >= q).float() * (u <= 1 - q).float(),  # step (stpz1/4)
    3: lambda u, q=1 / 4: (u >= q).float() * (u <= 1 - q).float()
    + (u < q).float() * 0.5,  # staircase (life)
    4: lambda u: torch.exp(-((u - 0.5) ** 2) / 0.2),
    8: lambda u: (torch.sin(10 * u) + 1) / 2,
    9: lambda u: (a * torch.sin((u.unsqueeze(-1) * 5 * b + c) * np.pi)).sum(-1)
    / (2 * a.sum())
    + 1 / 2,
}
field_func = {
    0: lambda n, m, s: torch.max(torch.zeros_like(n), 1 - (n - m) ** 2 / (9 * s**2))
    ** 4
    * 2
    - 1,  # polynomial (quad4)
    1: lambda n, m, s: torch.exp(-((n - m) ** 2) / (2 * s**2) - 1e-3) * 2
    - 1,  # exponential / gaussian (gaus)
    2: lambda n, m, s: (torch.abs(n - m) <= s).float() * 2 - 1,  # step (stpz)
    3: lambda n, m, s: -torch.clamp(n - m, 0, 1) * s,  # food eating kernl
}

# ker_c =lambda r,a,b,c :(a*torch.sin((r.unsqueeze(-1)*5*b+c)*np.pi)).sum(-1)/(2*a.sum())+1/2
ker_c = lambda x, r, w, b: (b * torch.exp(-(((x.unsqueeze(-1) - r) / w) ** 2) / 2)).sum(
    -1
)


class Dummy_init_mod(torch.nn.Module):
    def __init__(self, init):
        torch.nn.Module.__init__(self)
        self.register_parameter("init", torch.nn.Parameter(init))


# Lenia Step FFT version (faster)
class LeniaStepFFTC(torch.nn.Module):
    """Module pytorch that computes one Lenia Step with the fft version"""

    def __init__(
        self,
        C,
        R,
        T,
        c0,
        c1,
        r,
        rk,
        b,
        w,
        h,
        m,
        s,
        gn,
        is_soft_clip=False,
        SX=256,
        SY=256,
        speed_x=0,
        speed_y=0,
        device="cpu",
    ):
        torch.nn.Module.__init__(self)

        self.register_buffer("R", R)
        self.register_buffer("T", T)
        self.register_buffer("c0", c0)
        self.register_buffer("c1", c1)
        # self.register_buffer('r', r)
        self.register_parameter("r", torch.nn.Parameter(r))
        self.register_parameter("rk", torch.nn.Parameter(rk))
        self.register_parameter("b", torch.nn.Parameter(b))
        self.register_parameter("w", torch.nn.Parameter(w))
        self.register_parameter("h", torch.nn.Parameter(h))
        self.register_parameter("m", torch.nn.Parameter(m))
        self.register_parameter("s", torch.nn.Parameter(s))
        self.speed_x = speed_x
        self.speed_y = speed_y

        self.gn = 1
        self.nb_k = c0.shape[0]

        self.SX = SX
        self.SY = SY

        self.is_soft_clip = is_soft_clip
        self.C = C

        self.device = device
        self.to(device)
        self.kernels = torch.zeros((self.nb_k, self.SX, self.SY, 2)).to(self.device)

        self.compute_kernel()
        self.compute_kernel_env()

    def compute_kernel_env(self):
        """computes the kernel and the kernel FFT of the environnement from the parameters"""
        x = torch.arange(self.SX).to(self.device)
        y = torch.arange(self.SY).to(self.device)
        xx = x.view(-1, 1).repeat(1, self.SY)
        yy = y.repeat(self.SX, 1)
        X = (xx - int(self.SX / 2)).float()
        Y = (yy - int(self.SY / 2)).float()
        D = torch.sqrt(X**2 + Y**2) / (4)
        kernel = torch.sigmoid(-(D - 1) * 10) * ker_c(
            D,
            torch.tensor(np.array([0, 0, 0])).to(self.device),
            torch.tensor(np.array([0.5, 0.1, 0.1])).to(self.device),
            torch.tensor(np.array([1, 0, 0])).to(self.device),
        )
        kernel_sum = torch.sum(kernel)
        kernel_norm = kernel / kernel_sum
        # kernel_FFT = torch.rfft(kernel_norm, signal_ndim=2, onesided=False).to(self.device)
        kernel_FFT = torch.fft.rfftn(kernel_norm, dim=(0, 1)).to(self.device)

        self.kernel_wall = kernel_FFT

    def compute_kernel(self):
        """computes the kernel and the kernel FFT of the learnable channels from the parameters"""
        x = torch.arange(self.SX).to(self.device)
        y = torch.arange(self.SY).to(self.device)
        xx = x.view(-1, 1).repeat(1, self.SY)
        yy = y.repeat(self.SX, 1)
        X = (xx - int(self.SX / 2)).float()
        Y = (yy - int(self.SY / 2)).float()
        self.kernels = torch.zeros((self.nb_k, self.SX, self.SY // 2 + 1)).to(
            self.device
        )

        for i in range(self.nb_k):
            # distance to center in normalized space
            D = torch.sqrt(X**2 + Y**2) / ((self.R + 15) * self.r[i])

            kernel = torch.sigmoid(-(D - 1) * 10) * ker_c(
                D, self.rk[i], self.w[i], self.b[i]
            )
            kernel_sum = torch.sum(kernel)

            # normalization of the kernel
            kernel_norm = kernel / kernel_sum
            # plt.imshow(kernel_norm[0,0].detach().cpu()*100)
            # plt.show()

            # fft of the kernel
            # kernel_FFT = torch.rfft(kernel_norm, signal_ndim=2, onesided=False).to(self.device)
            kernel_FFT = torch.fft.rfftn(kernel_norm, dim=(0, 1)).to(self.device)

            self.kernels[i] = kernel_FFT

    def forward(self, input):
        input[:, :, :, 1] = torch.roll(
            input[:, :, :, 1], [self.speed_y, self.speed_x], [1, 2]
        )
        self.D = torch.zeros(input.shape).to(self.device)
        self.Dn = torch.zeros(self.C)

        # world_FFT = [torch.rfft(input[:,:,:,i], signal_ndim=2, onesided=False) for i in range(self.C)]
        world_FFT = [
            torch.fft.rfftn(input[:, :, :, i], dim=(1, 2)) for i in range(self.C)
        ]

        ## speed up of the update for 1 channel creature by multiplying by all the kernel FFT

        # channel 0 is the learnable channel
        world_FFT_c = world_FFT[0]
        # multiply the FFT of the world and the kernels
        potential_FFT = self.kernels * world_FFT_c
        # ifft + realignself.SY//2+1
        potential = torch.fft.irfftn(potential_FFT, dim=(1, 2))
        potential = roll_n(potential, 2, potential.size(2) // 2)
        potential = roll_n(potential, 1, potential.size(1) // 2)
        # growth function
        gfunc = field_func[min(self.gn, 3)]
        field = gfunc(
            potential,
            self.m.unsqueeze(-1).unsqueeze(-1),
            self.s.unsqueeze(-1).unsqueeze(-1),
        )
        # add the growth multiplied by the weight of the rule to the total growth
        self.D[:, :, :, 0] = (self.h.unsqueeze(-1).unsqueeze(-1) * field).sum(
            0, keepdim=True
        )
        self.Dn[0] = self.h.sum()

        ###Base version for the case where we want the learnable creature to be  multi channel (which is not used in this notebook)

        # for i in range(self.nb_k):
        #   c0b=int((self.c0[i]))
        #   c1b=int((self.c1[i]))

        #   world_FFT_c = world_FFT[c0b]
        #   potential_FFT = complex_mult_torch(self.kernels[i].unsqueeze(0), world_FFT_c)

        #   potential = torch.irfft(potential_FFT, signal_ndim=2, onesided=False)
        #   potential = roll_n(potential, 2, potential.size(2) // 2)
        #   potential = roll_n(potential, 1, potential.size(1) // 2)

        #   gfunc = field_func[min(self.gn, 3)]
        #   field = gfunc(potential, self.m[i], self.s[i])

        #   self.D[:,:,:,c1b]=self.D[:,:,:,c1b]+self.h[i]*field
        #   self.Dn[c1b]=self.Dn[c1b]+self.h[i]

        # apply wall
        world_FFT_c = world_FFT[self.C - 1]
        potential_FFT = self.kernel_wall * world_FFT_c
        potential = torch.fft.irfftn(potential_FFT, dim=(1, 2))
        potential = roll_n(potential, 2, potential.size(2) // 2)
        potential = roll_n(potential, 1, potential.size(1) // 2)
        gfunc = field_func[3]
        field = gfunc(potential, 1e-8, 10)
        for i in range(self.C - 1):
            c1b = i
            self.D[:, :, :, c1b] = self.D[:, :, :, c1b] + 1 * field
            self.Dn[c1b] = self.Dn[c1b] + 1

        ## Add the total growth to the current state
        if not self.is_soft_clip:

            output_img = torch.clamp(input + (1.0 / self.T) * self.D, min=0.0, max=1.0)
            # output_img = input + (1.0 / self.T) * ((self.D/self.Dn+1)/2-input)

        else:
            output_img = torch.sigmoid((input + (1.0 / self.T) * self.D - 0.5) * 10)
            # output_img = torch.tanh(input + (1.0 / self.T) * self.D)

        return output_img


class Lenia_C(torch.nn.Module):

    @staticmethod
    def default_config():
        default_config = Dict()
        default_config.version = "pytorch_fft"  # "pytorch_fft", "pytorch_conv2d"
        default_config.SX = 256
        default_config.SY = 256
        default_config.final_step = 40
        default_config.C = 2
        default_config.speed_x = 0
        default_config.speed_y = 0
        return default_config

    def __init__(
        self,
        initialization_space=None,
        update_rule_space=None,
        nb_k=10,
        init_size=40,
        config={},
        device=torch.device("cpu"),
        **kwargs,
    ):
        self.config = self.__class__.default_config()
        self.config.update(config)
        self.config.update(kwargs)
        torch.nn.Module.__init__(self)
        self.device = device
        self.init_size = init_size
        if initialization_space is not None:
            self.initialization_space = initialization_space
        else:
            self.initialization_space = LeniaInitializationSpace(self.init_size)

        if update_rule_space is not None:
            self.update_rule_space = update_rule_space
        else:
            self.update_rule_space = LeniaUpdateRuleSpace(nb_k)

        self.run_idx = 0
        self.init_wall = torch.zeros((self.config.SX, self.config.SY))
        # reset with no argument to sample random parameters
        self.reset()
        self.to(self.device)

    def reset(self, initialization_parameters=None, update_rule_parameters=None):
        # call the property setters
        if initialization_parameters is not None:
            self.initialization_parameters = initialization_parameters
        else:
            self.initialization_parameters = self.initialization_space.sample()

        if update_rule_parameters is not None:
            self.update_rule_parameters = update_rule_parameters
        else:
            policy_parameters = Dict.fromkeys(["update_rule"])
            policy_parameters["update_rule"] = self.update_rule_space.sample()
            # divide h by 3 at the beginning as some unbalanced kernels can easily kill
            policy_parameters["update_rule"].h = policy_parameters["update_rule"].h / 3
            self.update_rule_parameters = policy_parameters["update_rule"]

        # initialize Lenia CA with update rule parameters
        if self.config.version == "pytorch_fft":
            lenia_step = LeniaStepFFTC(
                self.config.C,
                self.update_rule_parameters["R"],
                self.update_rule_parameters["T"],
                self.update_rule_parameters["c0"],
                self.update_rule_parameters["c1"],
                self.update_rule_parameters["r"],
                self.update_rule_parameters["rk"],
                self.update_rule_parameters["b"],
                self.update_rule_parameters["w"],
                self.update_rule_parameters["h"],
                self.update_rule_parameters["m"],
                self.update_rule_parameters["s"],
                1,
                is_soft_clip=False,
                SX=self.config.SX,
                SY=self.config.SY,
                speed_x=self.config.speed_x,
                speed_y=self.config.speed_y,
                device=self.device,
            )
        self.add_module("lenia_step", lenia_step)

        # initialize Lenia initial state with initialization_parameters
        init = self.initialization_parameters["init"]
        # initialization_cppn = pytorchneat.rnn.RecurrentNetwork.create(cppn_genome, self.initialization_space.config.neat_config, device=self.device)
        self.add_module("initialization", Dummy_init_mod(init))

        # push the nn.Module and the available device
        self.to(self.device)
        self.generate_init_state()

    def random_obstacle(self, nb_obstacle=6):
        self.init_wall = torch.zeros((self.config.SX, self.config.SY))

        x = torch.arange(self.config.SX)
        y = torch.arange(self.config.SY)
        xx = x.view(-1, 1).repeat(1, self.config.SY)
        yy = y.repeat(self.config.SX, 1)
        for i in range(nb_obstacle):
            X = (xx - int(torch.rand(1) * self.config.SX)).float()
            Y = (yy - int(torch.rand(1) * self.config.SY / 2)).float()
            D = torch.sqrt(X**2 + Y**2) / 10
            mask = (D < 1).float()
            self.init_wall = torch.clamp(self.init_wall + mask, 0, 1)

    def random_obstacle_bis(self, nb_obstacle=6):
        self.init_wall = torch.zeros((self.config.SX, self.config.SY))

        x = torch.arange(self.config.SX)
        y = torch.arange(self.config.SY)
        xx = x.view(-1, 1).repeat(1, self.config.SY)
        yy = y.repeat(self.config.SX, 1)
        for i in range(nb_obstacle):
            X = (xx - int(torch.rand(1) * self.config.SX)).float()
            Y = (yy - int(torch.rand(1) * self.config.SY)).float()
            D = torch.sqrt(X**2 + Y**2) / 10
            mask = (D < 1).float()
            self.init_wall = torch.clamp(self.init_wall + mask, 0, 1)
        self.init_wall[95:155, 170:230] = 0

    def generate_init_state(self, X=105, Y=180):
        init_state = torch.zeros(
            1, self.config.SX, self.config.SY, self.config.C, dtype=torch.float64
        )
        init_state[0, X : X + self.init_size, Y : Y + self.init_size, 0] = (
            self.initialization.init
        )
        if self.config.C > 1:
            init_state[0, :, :, 1] = self.init_wall
        self.state = init_state.to(self.device)
        self.step_idx = 0

    def update_initialization_parameters(self):
        new_initialization_parameters = deepcopy(self.initialization_parameters)
        new_initialization_parameters["init"] = self.initialization.init.data
        if not self.initialization_space.contains(new_initialization_parameters):
            new_initialization_parameters = self.initialization_space.clamp(
                new_initialization_parameters
            )
            warnings.warn(
                "provided parameters are not in the space range and are therefore clamped"
            )
        self.initialization_parameters = new_initialization_parameters

    def update_update_rule_parameters(self):
        new_update_rule_parameters = deepcopy(self.update_rule_parameters)
        # gather the parameter from the lenia step (which may have been optimized)
        new_update_rule_parameters["m"] = self.lenia_step.m.data
        new_update_rule_parameters["s"] = self.lenia_step.s.data
        new_update_rule_parameters["r"] = self.lenia_step.r.data
        new_update_rule_parameters["rk"] = self.lenia_step.rk.data
        new_update_rule_parameters["b"] = self.lenia_step.b.data
        new_update_rule_parameters["w"] = self.lenia_step.w.data
        new_update_rule_parameters["h"] = self.lenia_step.h.data
        if not self.update_rule_space.contains(new_update_rule_parameters):
            new_update_rule_parameters = self.update_rule_space.clamp(
                new_update_rule_parameters
            )
            warnings.warn(
                "provided parameters are not in the space range and are therefore clamped"
            )
        self.update_rule_parameters = new_update_rule_parameters

    def step(self, intervention_parameters=None):
        self.state = self.lenia_step(self.state)
        self.step_idx += 1
        return self.state

    def forward(self):
        state = self.step(None)
        return state

    def run(self):
        """run lenia for the number of step specified in the config.
        Returns the observations containing the state at each timestep"""
        # clip parameters just in case
        if not self.initialization_space["init"].contains(
            self.initialization.init.data
        ):
            self.initialization.init.data = self.initialization_space["init"].clamp(
                self.initialization.init.data
            )
        if not self.update_rule_space["r"].contains(self.lenia_step.r.data):
            self.lenia_step.r.data = self.update_rule_space["r"].clamp(
                self.lenia_step.r.data
            )
        if not self.update_rule_space["rk"].contains(self.lenia_step.rk.data):
            self.lenia_step.rk.data = self.update_rule_space["rk"].clamp(
                self.lenia_step.rk.data
            )
        if not self.update_rule_space["b"].contains(self.lenia_step.b.data):
            self.lenia_step.b.data = self.update_rule_space["b"].clamp(
                self.lenia_step.b.data
            )
        if not self.update_rule_space["w"].contains(self.lenia_step.w.data):
            self.lenia_step.w.data = self.update_rule_space["w"].clamp(
                self.lenia_step.w.data
            )
        if not self.update_rule_space["h"].contains(self.lenia_step.h.data):
            self.lenia_step.h.data = self.update_rule_space["h"].clamp(
                self.lenia_step.h.data
            )
        if not self.update_rule_space["m"].contains(self.lenia_step.m.data):
            self.lenia_step.m.data = self.update_rule_space["m"].clamp(
                self.lenia_step.m.data
            )
        if not self.update_rule_space["s"].contains(self.lenia_step.s.data):
            self.lenia_step.s.data = self.update_rule_space["s"].clamp(
                self.lenia_step.s.data
            )
        # self.generate_init_state()
        observations = Dict()
        observations.timepoints = list(range(self.config.final_step))
        observations.states = torch.empty(
            (self.config.final_step, self.config.SX, self.config.SY, self.config.C)
        )
        observations.states[0] = self.state
        for step_idx in range(1, self.config.final_step):
            cur_observation = self.step(None)
            observations.states[step_idx] = cur_observation[0, :, :, :]

        return observations

    def save(self, filepath):
        """
        Saves the system object using torch.save function in pickle format
        Can be used if the system state's change over exploration and we want to dump it
        """
        torch.save(self, filepath)

    def close(self):
        pass


# IMGEP Algorithm


class OutputRepresentation:
    """Base class to map the observations of a system to an embedding vector (BC characterization)"""

    @staticmethod
    def default_config():
        default_config = Dict()
        return default_config

    def __init__(self, config={}, **kwargs):
        self.config = self.__class__.default_config()
        self.config.update(config)
        self.config.update(kwargs)


class LeniaCentroidRepresentation(OutputRepresentation):

    @staticmethod
    def default_config():
        default_config = OutputRepresentation.default_config()
        default_config.env_size = (256, 256)
        default_config.distance_function = "L2"
        return default_config

    def __init__(self, config=None, **kwargs):
        super().__init__(config=config, **kwargs)
        self.n_latents = 3

    def calc(self, observations):
        """
        Maps the observations of a system to an embedding vector
        Return a torch tensor
        """

        # filter low values
        filtered_im = observations.states[-1, :, :, 0]

        # recenter
        mu_0 = filtered_im.sum()

        # implementation of meshgrid in torch
        x = torch.arange(self.config.env_size[0])
        y = torch.arange(self.config.env_size[1])
        yy = y.repeat(self.config.env_size[0], 1)
        xx = x.view(-1, 1).repeat(1, self.config.env_size[1])

        X = (xx - int(self.config.env_size[0] / 2)).double()
        Y = (yy - int(self.config.env_size[1] / 2)).double()

        centroid_x = (X * filtered_im).sum() / (mu_0 + 1e-10)
        centroid_y = (Y * filtered_im).sum() / (mu_0 + 1e-10)
        X = (xx - centroid_x - self.config.env_size[0] / 2).double()
        Y = (yy - centroid_y - self.config.env_size[1] / 2).double()

        # distance to center in normalized space
        D = torch.sqrt(X**2 + Y**2) / (35)

        mask = 0.85 * (D < 0.5).float() + 0.15 * (D < 1).float()
        loss = (filtered_im - 0.9 * mask).pow(2).sum().sqrt()

        embedding = torch.zeros(3)
        embedding[0] = loss / 230
        embedding[1] = centroid_x.mean() / self.config.env_size[0]
        embedding[2] = centroid_y.mean() / self.config.env_size[1]
        if mu_0 < 1e-4:
            embedding[1] = embedding[1] - 10
            embedding[2] = embedding[2] - 10

        # print(embedding)

        return embedding

    def calc_distance(self, embedding_a, embedding_b):
        """
        Compute the distance between 2 embeddings in the latent space
        /!\ batch mode embedding_a and embedding_b can be N*M or M
        """
        # l2 loss
        if self.config.distance_function == "L2":
            dist = (embedding_a - embedding_b).pow(2).sum(-1).sqrt()

        else:
            raise NotImplementedError

        return dist


class Explorer:
    """
    Base class for exploration experiments.
    Allows to save and load exploration results
    """

    @staticmethod
    def default_config():
        default_config = Dict()
        return default_config

    def __init__(self, system, explorationdb, config={}, **kwargs):
        self.config = self.__class__.default_config()
        self.config.update(config)
        self.config.update(kwargs)

        self.system = system
        self.db = explorationdb

    def save(self, filepath):
        """
        Saves the explorer object using torch.save function in pickle format
        /!\ We intentionally empty explorer.db from the pickle
        because the database is already automatically saved in external files each time the explorer call self.db.add_run_data
        """
        # do not pickle the data as already saved in extra files
        tmp_data = self.db
        self.db.reset_empty_db()

        # pickle exploration object
        torch.save(self, filepath)

        # attach db again to the exploration object
        self.db = tmp_data

    @staticmethod
    def load(explorer_filepath, load_data=True, run_ids=None, map_location="cuda"):

        explorer = torch.load(explorer_filepath, map_location=map_location)

        # loop over policy parameters to coalesce sparse tensors (not coalesced by default)
        def coalesce_parameter_dict(d, has_coalesced_tensor=False):
            for k, v in d.items():
                if isinstance(v, Dict):
                    d[k], has_coalesced_tensor = coalesce_parameter_dict(
                        v, has_coalesced_tensor=has_coalesced_tensor
                    )
                elif (
                    isinstance(v, torch.Tensor) and v.is_sparse and not v.is_coalesced()
                ):
                    d[k] = v.coalesce()
                    has_coalesced_tensor = True
            return d, has_coalesced_tensor

        for policy_idx, policy in enumerate(explorer.policy_library):
            explorer.policy_library[policy_idx], has_coalesced_tensor = (
                coalesce_parameter_dict(policy)
            )
            if not has_coalesced_tensor:
                break

        if load_data:
            explorer.db = ExplorationDB(config=explorer.db.config)
            explorer.db.load(run_ids=run_ids, map_location=map_location)

        return explorer


class IMGEPExplorer(Explorer):
    """
    Basic explorer that samples goals in a goalspace and uses a policy library to generate parameters to reach the goal.
    """

    # Set these in ALL subclasses
    goal_space = None  # defines the obs->goal representation and the goal sampling strategy (self.goal_space.sample())
    reach_goal_optimizer = None

    @staticmethod
    def default_config():
        default_config = Dict()
        # base config
        default_config.num_of_random_initialization = 40  # number of random runs at the beginning of exploration to populate the IMGEP memory

        # Pi: source policy parameters config
        default_config.source_policy_selection = Dict()
        default_config.source_policy_selection.type = (
            "optimal"  # either: 'optimal', 'random'
        )

        # Opt: Optimizer to reach goal
        default_config.reach_goal_optimizer = Dict()
        default_config.reach_goal_optimizer.optim_steps = 10
        default_config.reach_goal_optimizer.name = "SGD"
        default_config.reach_goal_optimizer.initialization_cppn.parameters.lr = 1e-3
        default_config.reach_goal_optimizer.lenia_step.parameters.lr = 1e-4
        # default_config.reach_goal_optimizer.parameters.eps=1e-4

        return default_config

    def __init__(self, system, explorationdb, goal_space, config={}, **kwargs):
        super().__init__(
            system=system, explorationdb=explorationdb, config=config, **kwargs
        )

        self.goal_space = goal_space

        # initialize policy library
        self.policy_library = []

        # initialize goal library
        self.goal_library = torch.empty((0,) + self.goal_space.shape)

        # reach goal optimizer
        self.reach_goal_optimizer = None

    def get_source_policy_idx(self, target_goal):

        if self.config.source_policy_selection.type == "optimal":
            # get distance to other goals
            tbis = self.goal_library * 1.0
            # augment distance to creature that exploded or died because we don't want to select them.
            tbis[:, 1] = tbis[:, 1] + (tbis[:, 1] < -9).float() * 100
            tbis[:, 1] = tbis[:, 1] + (tbis[:, 0] > 0.11).float() * 100
            goal_distances = self.goal_space.calc_distance(
                target_goal.unsqueeze(0), tbis
            )
            source_policy_idx = torch.argmin(goal_distances)

        elif self.config.source_policy_selection.type == "random":
            source_policy_idx = sample_value(
                ("discrete", 0, len(self.goal_library) - 1)
            )

        else:
            raise ValueError(
                "Unknown source policy selection type {!r} in the configuration!".format(
                    self.config.source_policy_selection.type
                )
            )

        return source_policy_idx

    def sample__interesting_goal(self):
        """Sample a target goal randomly but taking into account the goal already reached in order to not sample an area to close from an already reached zone or too far from what can be reached"""
        # arbitrary sampling of goal, some other may be more efficient
        close = 0
        veryclose = 10
        compt = 0

        # change distance for reached goal when the creature died or exploded
        tbis = self.goal_library * 1.0
        tbis[:, 2] = tbis[:, 2] + (tbis[:, 1] < -9).float() * 100
        tbis[:, 2] = tbis[:, 2] + (tbis[:, 0] > 0.11).float() * 100

        # loop until region not explored too much and also not too far
        target_goal = torch.ones(3) * -10
        while close < 1 or veryclose > 2:
            target_goal[0] = 0.065 + torch.normal(torch.zeros(1)) * 0.002
            if torch.rand(1) < 0.2:
                # go a little further than previous best
                ind = torch.argmin(tbis[:, 2])
                target_goal[1] = tbis[ind, 1] + (torch.rand(1) * 0.45 - 0.22) / 4
                target_goal[2] = tbis[ind, 2] - 0.04 * torch.rand(1) - 0.02

            else:
                # with high probability try far points
                if torch.rand(1) < 0.7:
                    target_goal[2] = torch.rand(1) * 0.2 - 0.35
                    target_goal[1] = -(torch.rand(1) * 0.45 - 0.22)
                else:
                    target_goal[2] = torch.rand(1) * 0.35 - 0.35
                    target_goal[1] = -(torch.rand(1) * 0.45 - 0.22)

            goal_distances = self.goal_space.calc_distance(
                target_goal.unsqueeze(0), tbis
            )
            close = (goal_distances < 0.1).float().sum()
            veryclose = (goal_distances < 0.06).float().sum()
            compt = compt + 1
        return target_goal

    def run(self, n_exploration_runs, continue_existing_run=False):

        again = True
        # while loop that sample new initialization and try until a good run is achieved,
        # the run will start from a new initialization in some cases like not enough progress in a certain number of steps etc
        while again:
            print("NEW TRY OF INIT")

            print("Exploration: ")
            progress_bar = tqdm(total=n_exploration_runs)
            if continue_existing_run:
                run_idx = len(self.policy_library)
                progress_bar.update(run_idx)
            else:
                self.policy_library = []
                self.goal_library = torch.empty((0,) + self.goal_space.shape)
                run_idx = 0
            nb_alive_random = 0

            ############# Beginning of the search ##############
            while run_idx < n_exploration_runs:
                policy_parameters = Dict.fromkeys(
                    ["initialization", "update_rule"]
                )  # policy parameters (output of IMGEP policy)

                ############ Initial Random Sampling of Parameters ####################
                if len(self.policy_library) < self.config.num_of_random_initialization:

                    target_goal = None
                    source_policy_idx = None
                    reached_goal = torch.ones(19)

                    # sample new parameters to test
                    policy_parameters["initialization"] = (
                        self.system.initialization_space.sample()
                    )
                    policy_parameters["update_rule"] = (
                        self.system.update_rule_space.sample()
                    )
                    # divide h by 3 at the beginning as some unbalanced kernels can easily kill
                    policy_parameters["update_rule"].h = (
                        policy_parameters["update_rule"].h / 3
                    )
                    self.system.reset(
                        initialization_parameters=policy_parameters["initialization"],
                        update_rule_parameters=policy_parameters["update_rule"],
                    )

                    # run the system
                    with torch.no_grad():
                        self.system.random_obstacle(8)
                        self.system.generate_init_state()
                        observations = self.system.run()
                        reached_goal = self.goal_space.map(observations)
                    is_dead = reached_goal[0] > 0.9 or reached_goal[1] < -0.5
                    if not is_dead:
                        nb_alive_random = nb_alive_random + 1

                    optim_step_idx = 0
                    dist_to_target = None

                ############## Goal-directed Sampling of Parameters ######################
                else:

                    # sample a goal space from the goal space

                    # for the first 8 target goal simply try to go as far as possible straight (each time goal a little bit further)
                    if (
                        len(self.policy_library)
                        - self.config.num_of_random_initialization
                        < 8
                    ):
                        target_goal = torch.ones(3) * -10
                        target_goal[0] = 0.065
                        target_goal[2] = (
                            0.19
                            - (
                                len(self.policy_library)
                                - self.config.num_of_random_initialization
                            )
                            * 0.06
                        )
                        target_goal[1] = 0
                    # then random goal in a region not reached but not too far
                    else:
                        target_goal = self.sample__interesting_goal()

                    if (
                        len(self.policy_library)
                        - self.config.num_of_random_initialization
                        >= 2
                    ):
                        print(f"Run {run_idx}, optimisation toward goal: ")
                        print("TARGET =" + str(target_goal))

                    # get source policy for this target goal
                    source_policy_idx = self.get_source_policy_idx(target_goal)
                    source_policy = self.policy_library[source_policy_idx]

                    # if we're at the beginning or iteration%5==0 then don't mutate and train for longer
                    if (
                        len(self.policy_library)
                        - self.config.num_of_random_initialization
                        < 8
                        or len(self.policy_library) % 5 == 0
                    ):

                        policy_parameters["initialization"] = deepcopy(
                            source_policy["initialization"]
                        )
                        policy_parameters["update_rule"] = deepcopy(
                            source_policy["update_rule"]
                        )
                        self.system.reset(
                            initialization_parameters=policy_parameters[
                                "initialization"
                            ],
                            update_rule_parameters=policy_parameters["update_rule"],
                        )
                        ite = self.config.reach_goal_optimizer.optim_steps
                    # else mutate
                    else:
                        ite = 15
                        # mutate until finding a non dying and non exploding creature
                        die_mutate = True
                        while die_mutate:
                            policy_parameters["initialization"] = (
                                self.system.initialization_space.mutate(
                                    source_policy["initialization"]
                                )
                            )
                            policy_parameters["update_rule"] = (
                                self.system.update_rule_space.mutate(
                                    source_policy["update_rule"]
                                )
                            )
                            self.system.reset(
                                initialization_parameters=policy_parameters[
                                    "initialization"
                                ],
                                update_rule_parameters=policy_parameters["update_rule"],
                            )
                            with torch.no_grad():
                                self.system.generate_init_state()
                                observations = self.system.run()
                                reached_goal = self.goal_space.map(observations)
                            # if doesn't not die or explode break the loop
                            if (
                                observations.states[-1, :, :, 0].sum() > 10
                                or reached_goal[0] > 0.11
                            ):
                                die_mutate = False

                    ##### INNER LOOP (Optimization part toward target goal ) ####
                    if (
                        isinstance(self.system, torch.nn.Module)
                        and self.config.reach_goal_optimizer.optim_steps > 0
                    ):

                        optimizer_class = eval(
                            f"torch.optim.{self.config.reach_goal_optimizer.name}"
                        )
                        self.reach_goal_optimizer = optimizer_class(
                            [
                                {
                                    "params": self.system.initialization.parameters(),
                                    **self.config.reach_goal_optimizer.initialization_cppn.parameters,
                                },
                                {
                                    "params": self.system.lenia_step.parameters(),
                                    **self.config.reach_goal_optimizer.lenia_step.parameters,
                                },
                            ],
                            **self.config.reach_goal_optimizer.parameters,
                        )

                        last_dead = False
                        for optim_step_idx in range(1, ite):

                            # run system with IMGEP's policy parameters
                            self.system.random_obstacle(8)
                            self.system.generate_init_state()
                            observations = self.system.run()
                            reached_goal = self.goal_space.map(observations)

                            ### Define  target disk
                            x = torch.arange(self.system.config.SX)
                            y = torch.arange(self.system.config.SY)
                            xx = x.view(-1, 1).repeat(1, self.system.config.SY)
                            yy = y.repeat(self.system.config.SX, 1)
                            X = (
                                xx - (target_goal[1] + 0.5) * self.system.config.SX
                            ).float() / (35)
                            Y = (
                                yy - (target_goal[2] + 0.5) * self.system.config.SY
                            ).float() / (35)
                            # distance to center in normalized space
                            D = torch.sqrt(X**2 + Y**2)
                            # mask is the target circles
                            mask = 0.85 * (D < 0.5).float() + 0.15 * (D < 1).float()

                            loss = (
                                (0.9 * mask - observations.states[-1, :, :, 0])
                                .pow(2)
                                .sum()
                                .sqrt()
                            )

                            # optimisation step
                            self.reach_goal_optimizer.zero_grad()
                            loss.backward()
                            self.reach_goal_optimizer.step()

                            # compute again the kernels for the next step because parameters have been changed with the optimization
                            self.system.lenia_step.compute_kernel()

                            dead = observations.states[-1, :, :, 0].sum() < 10
                            if dead and last_dead:
                                self.reach_goal_optimizer.zero_grad()
                                break
                            last_dead = dead

                        ###### END of INNER loop #####

                        # if not enough improvement at the first outer loop (just after random explo) then try another init
                        if (
                            len(self.policy_library)
                            >= self.config.num_of_random_initialization
                            and len(self.policy_library)
                            - self.config.num_of_random_initialization
                            < 2
                        ):
                            if loss > 19.5:
                                break
                            else:
                                if (
                                    len(self.policy_library)
                                    - self.config.num_of_random_initialization
                                    == 2
                                ):
                                    again = False

                        # gather back the trained parameters
                        self.system.update_initialization_parameters()
                        self.system.update_update_rule_parameters()
                        policy_parameters["initialization"] = (
                            self.system.initialization_parameters
                        )
                        policy_parameters["update_rule"] = (
                            self.system.update_rule_parameters
                        )
                        dist_to_target = loss.item()

                    ## look at the reached goal ##
                    reached_goal = torch.zeros(3).cpu()
                    with torch.no_grad():
                        for _ in range(20):
                            self.system.random_obstacle(8)
                            self.system.generate_init_state()
                            observations = self.system.run()
                            if observations.states[-1, :, :, 0].sum() < 10:
                                reached_goal[0] = 10
                                break
                            reached_goal = (
                                reached_goal
                                + self.goal_space.map(observations).cpu() / 20
                            )
                    if (
                        len(self.policy_library)
                        - self.config.num_of_random_initialization
                        >= 2
                    ):
                        print("reached= " + str(reached_goal))

                # save results
                reached_goal = reached_goal.cpu()
                self.db.add_run_data(
                    id=run_idx,
                    policy_parameters=policy_parameters,
                    observations=observations,
                    source_policy_idx=source_policy_idx,
                    target_goal=target_goal,
                    reached_goal=reached_goal,
                    n_optim_steps_to_reach_goal=optim_step_idx,
                    dist_to_target=dist_to_target,
                )

                # add policy and reached goal into the libraries
                # do it after the run data is saved to not save them if there is an error during the saving

                self.policy_library.append(policy_parameters)
                self.goal_library = torch.cat(
                    [
                        self.goal_library,
                        reached_goal.reshape(1, -1)
                        .to(self.goal_library.device)
                        .detach(),
                    ]
                )
                if len(self.policy_library) >= self.config.num_of_random_initialization:
                    plt.imshow(self.system.init_wall.cpu())
                    plt.scatter(
                        (
                            (self.goal_library[:, 0] < 0.11).float()
                            * (self.goal_library[:, 2] > -0.5).float()
                            * (self.goal_library[:, 2] + 0.5)
                            * self.system.config.SY
                        ).cpu(),
                        (
                            (self.goal_library[:, 0] < 0.11).float()
                            * (self.goal_library[:, 1] > -0.5).float()
                            * (self.goal_library[:, 1] + 0.5)
                            * self.system.config.SX
                        ).cpu(),
                    )
                    plt.show()
                # increment run_idx

                run_idx += 1
                progress_bar.update(1)

                # after the random explo if not enough living crea (even static ones)
                if len(self.policy_library) == self.config.num_of_random_initialization:
                    if nb_alive_random < 2:
                        break
                    print(run_idx)

                if len(self.policy_library) == n_exploration_runs - 1:
                    again = False
