import numpy as np
import shap.utils
from shap.maskers import Masker
import heapq
from numba import jit

# TODO: heapq in numba does not yet support Typed Lists so we can move to them yet...
from numba.core.errors import NumbaPendingDeprecationWarning
import warnings
warnings.simplefilter('ignore', category=NumbaPendingDeprecationWarning)


class LimeSegmenter():
    
    def __init__(self):
        pass

    def __call__(self, data: np.array):
        segments = np.zeros(data.shape[:2])
        for x in range(data.shape[0]):
            segments[x, :] = x
        return segments


class EegMask(Masker):
    """ This masks out image regions with blurring or inpainting.
    """

    def __init__(self, mask_method, shape=None):
        """ Build a new Image masker with the given masking value.
        Parameters
        ----------
        mask_method : Method to mask hidden regions of the image.
        shape : None or tuple
            If the mask_value is an auto-generated masker instead of a dataset then the input
            image shape needs to be provided.
        """
        if shape is None:
            if isinstance(mask_method, str):
                raise TypeError("When the mask_value is a string the shape parameter must be given!")
            self.input_shape = mask_method.shape # the (1,) is because we only return a single masked sample to average over
        else:
            self.input_shape = shape

        self.input_mask_value = mask_method

        # This is the shape of the masks we expect
        self.shape = (1, np.prod(self.input_shape)) # the (1, ...) is because we only return a single masked sample to average over

        self.image_data = True

        assert isinstance(mask_method ,str)
        self.mask_method = mask_method
        self.build_partition_tree()

        #self.scratch_mask = np.zeros(self.input_shape[:-1], dtype=np.bool)
        self.last_xid = None

    def __call__(self, mask, x):
        if np.prod(x.shape) != np.prod(self.input_shape):
            raise Exception("The length of the image to be masked must match the shape given in the " + \
                            "ImageMasker contructor: "+" * ".join([str(i) for i in x.shape])+ \
                            " != "+" * ".join([str(i) for i in self.input_shape]))

        # unwrap single element lists (which are how single input models look in multi-input format)
        if isinstance(x, list) and len(x) == 1:
            x = x[0]

        # we preserve flattend inputs as flattened and full-shaped inputs as their original shape
        in_shape = x.shape
        if len(x.shape) > 1:
            x = x.ravel()

        # if mask is not given then we mask the whole image
        if mask is None:
            mask = np.zeros(np.prod(x.shape), dtype=np.bool)

        if self.mask_method == "gaussian_noise":
            std_deviation = np.std(x)*0.01
            background = np.random.normal(0, std_deviation, self.input_shape).ravel()
        elif self.mask_method == "zeros":
            background = np.zeros(self.input_shape).ravel()
        out = x.copy()
        out[~mask] = background[~mask]

        return (out.reshape(1, *in_shape),)

    def build_partition_tree(self):
        """ This partitions an image into a herarchical clustering based on axis-aligned splits.
        """
        xmin = 0
        xmax = self.input_shape[0]
        ymin = 0
        ymax = self.input_shape[1]
        zmin = 0
        zmax = self.input_shape[2]
        #total_xwidth = xmax - xmin
        total_ywidth = ymax - ymin
        total_zwidth = zmax - zmin
        q = [(0, xmin, xmax, ymin, ymax, zmin, zmax, -1, False)]
        # q = numba.typed.List([(0, xmin, xmax, ymin, ymax, zmin, zmax, -1, False)]) # TODO: won't work until the next numba rel (as of dec 2021)
        M = int((xmax - xmin) * (ymax - ymin) * (zmax - zmin))
        clustering = np.zeros((M - 1, 4))
        _jit_build_partition_tree(xmin, xmax, ymin, ymax, zmin, zmax, total_ywidth, total_zwidth, M, clustering, q)
        self.clustering = clustering


@jit
def _jit_build_partition_tree(xmin, xmax, ymin, ymax, zmin, zmax, total_ywidth, total_zwidth, M, clustering, q):
    """ This partitions an image into a herarchical clustering based on axis-aligned splits.
    """
    # heapq.heappush(q, (0, xmin, xmax, ymin, ymax, zmin, zmax, -1, False))

    # q.put((0, xmin, xmax, ymin, ymax, zmin, zmax, -1, False))
    ind = len(clustering) - 1
    while len(q) > 0: # q.empty()
        _, xmin, xmax, ymin, ymax, zmin, zmax, parent_ind, is_left =  heapq.heappop(q)
        # _, xmin, xmax, ymin, ymax, zmin, zmax, parent_ind, is_left = q.get()

        if parent_ind >= 0:
            clustering[parent_ind, 0 if is_left else 1] = ind + M

        # make sure we line up with a flattened indexing scheme
        if ind < 0:
            assert -ind - 1 == xmin * total_ywidth * total_zwidth + ymin * total_zwidth + zmin

        xwidth = xmax - xmin
        ywidth = ymax - ymin
        zwidth = zmax - zmin
        if xwidth == 1 and ywidth == 1 and zwidth == 1:
            pass
        else:

            # by default our ranges remain unchanged
            lxmin = rxmin = xmin
            lxmax = rxmax = xmax
            lymin = rymin = ymin
            lymax = rymax = ymax
            lzmin = rzmin = zmin
            lzmax = rzmax = zmax

            # split the xaxis if it is the largest dimension
            if xwidth > 1:
                xmid = xmin + xwidth // 2
                lxmax = xmid
                rxmin = xmid

            # split the yaxis
            elif ywidth > 1:
                ymid = ymin + ywidth // 2
                lymax = ymid
                rymin = ymid

            # split the zaxis only when the other ranges are already width 1
            else:
                zmid = zmin + zwidth // 2
                lzmax = zmid
                rzmin = zmid

            lsize = (lxmax - lxmin) * (lymax - lymin) * (lzmax - lzmin)
            rsize = (rxmax - rxmin) * (rymax - rymin) * (rzmax - rzmin)

            heapq.heappush(q, (-lsize, lxmin, lxmax, lymin, lymax, lzmin, lzmax, ind, True))
            heapq.heappush(q, (-rsize, rxmin, rxmax, rymin, rymax, rzmin, rzmax, ind, False))
            # q.put((-lsize, lxmin, lxmax, lymin, lymax, lzmin, lzmax, ind, True))
            # q.put((-rsize, rxmin, rxmax, rymin, rymax, rzmin, rzmax, ind, False))

        ind -= 1

    # fill in the group sizes
    for i in range(len(clustering)):
        li = int(clustering[i, 0])
        ri = int(clustering[i, 1])
        lsize = 1 if li < M else clustering[li-M, 3]
        rsize = 1 if ri < M else clustering[ri-M, 3]
        clustering[i, 3] = lsize + rsize
