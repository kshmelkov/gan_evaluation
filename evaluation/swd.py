# coding: utf-8
# vim:fdm=marker

# Adapted from https://github.com/tkarras/progressive_growing_of_gans/blob/master/sliced_wasserstein.py

import numpy as np
import scipy.ndimage
import time
# from tqdm import tqdm


class Sliced_Wasserstein_Scorer(object):
    """
    Instantiation adds operations to the graph and should be done before said graph is frozen.
    calc_sliced_wasserstein_scores runs calc_sliced_wasserstein_scores_reals the first time it is called and calc_sliced_wasserstein_scores_fake on every new call.
    log_summaries handles tensorflow.
    print functions are used for debugging mostly.
    """

    def __init__(self,
                 resolution_full,
                 resolution_min,
                 resolution_max,
                 num_images=8192,
                 nhoods_per_image=64,
                 nhood_size=7,
                 dir_repeats=1,
                 dirs_per_repeat=147, minibatch_size=20):
        """
        The role of this function is to prepare summary ops for logging sliced Wasserstein distance as early as possible.
        Indeed we are using Session supervisors, which freeze the network, so we need some summary ops to be added to
        the graph before running the model, because by the time the model runs the network is already frozen.
        """
        # We need to forecast resolutions at init time
        self.resolution_min = min(resolution_min, resolution_full)
        self.resolution_max = min(resolution_max, resolution_full)

        self.base_lod = int(np.log2(resolution_full)) - int(np.log2(self.resolution_max))
        self.resolutions = [2**i for i in range(int(np.log2(self.resolution_max)),
                            int(np.log2(self.resolution_min)) - 1, -1)]

        # Configuration
        self.resolution_min = resolution_min
        self.resolution_max = resolution_max
        self.num_images = num_images
        self.nhoods_per_image = nhoods_per_image
        self.nhood_size = nhood_size
        self.dir_repeats = dir_repeats
        self.dirs_per_repeat = dirs_per_repeat
        self.minibatch_size = minibatch_size

        # Variables that will be computed
        #self.base_lod = None
        #self.resolutions = None # LOL...
        self.desc_real = None
        self.desc_real_good = None

    # Wrapper around the two others.
    def calc_sliced_wasserstein_scores(self, training_set, fake_images,
                                       main_session=None,
                                       summary_adder=None, global_step=None, good_score_time=False):
        # At good score time, real_descriptors are different.
        if good_score_time:
            self.desc_real_fast = self.desc_real
            self.desc_real = self.desc_real_good

        # Select resolutions.
        training_set = np.transpose(training_set, (0, 3, 1, 2))
        fake_images = np.transpose(fake_images, (0, 3, 1, 2))

        # Get resolutions
        resolution_full = np.shape(training_set)[3]
        self.resolution_min = min(self.resolution_min, resolution_full)
        self.resolution_max = min(self.resolution_max, resolution_full)

        self.base_lod = int(np.log2(resolution_full)) - int(np.log2(self.resolution_max))
        new_resolutions = [2**i for i in range(int(np.log2(self.resolution_max)),
                            int(np.log2(self.resolution_min)) - 1, -1)]

        try:
            assert(new_resolutions == self.resolutions), "Resolutions forecasted and \
                obtained are different, init is false and would crash at run time."
        except:
            import pdb; pdb.set_trace();


        # Real descriptors (computed only once for now.) TODO add the possiblity to randomize the images.
        if self.desc_real is None:
            self.desc_real, self.scores_reals = \
                self._calc_sliced_wasserstein_scores_reals(training_set)

        self.print_real_res(self.resolutions, self.scores_reals)

        self.scores_fakes = self._calc_sliced_wasserstein_scores_fake(fake_images=fake_images)

        # If good score time, we need to restore the normal state:
        if good_score_time:
            self.desc_real_good = self.desc_real
            self.desc_real = self.desc_real_fast

        return self.scores_reals, self.scores_fakes

    def _calc_sliced_wasserstein_scores_reals(self, training_set):
        resolutions = self.resolutions
        num_images, minibatch_size = self.num_images, self.minibatch_size

        # Collect descriptors for reals.
        print('Extracting descriptors for reals...')
        time_begin = time.time()
        desc_real = [[] for res in resolutions]
        desc_test = [[] for res in resolutions]

        # for minibatch_begin in tqdm(range(0, num_images, minibatch_size)):
        for minibatch_begin in range(0, num_images, minibatch_size):
            #minibatch = training_set.get_random_minibatch(minibatch_size, lod=base_lod)

            minibatch = training_set[minibatch_begin: minibatch_begin + minibatch_size, ...]
            laplacian_pyramid = generate_laplacian_pyramid(
                minibatch, len(resolutions))
            for lod, level in enumerate(laplacian_pyramid):
                desc_real[lod].append(get_descriptors_for_minibatch(
                    level, self.nhood_size, self.nhoods_per_image))
                desc_test[lod].append(get_descriptors_for_minibatch(
                    level, self.nhood_size, self.nhoods_per_image))

        print('done in %s' % (time.time() - time_begin))

        # Evaluate scores for reals.
        print('Evaluating scores for reals...'),
        time_begin = time.time()
        scores = []
        for lod, res in enumerate(resolutions):
            desc_real[lod] = finalize_descriptors(desc_real[lod])
            desc_test[lod] = finalize_descriptors(desc_test[lod])
            scores.append(sliced_wasserstein(desc_real[lod], desc_test[lod],
                          self.dir_repeats, self.dirs_per_repeat))
        del desc_test
        print('done in %s' % (time.time() - time_begin))
        return desc_real, scores

    def _calc_sliced_wasserstein_scores_fake(self, fake_images):
        resolutions = self.resolutions
        desc_real = self.desc_real
        num_images, minibatch_size = self.num_images, self.minibatch_size
        base_lod = self.base_lod

        # Extract descriptors for generated images, batch by batch.
        desc_fake = [[] for res in resolutions]
        # for minibatch_begin in tqdm(range(0, num_images, minibatch_size)):
        for minibatch_begin in range(0, num_images, minibatch_size):
            minibatch = fake_images[minibatch_begin: minibatch_begin + minibatch_size, ...]
            minibatch = downscale_minibatch(minibatch, base_lod)
            laplacian_pyramid = generate_laplacian_pyramid(
                minibatch, len(resolutions))
            for lod, level in enumerate(laplacian_pyramid):
                desc_fake[lod].append(get_descriptors_for_minibatch(
                    level, self.nhood_size, self.nhoods_per_image))

        # Evaluate scores.
        scores = []
        for lod, res in enumerate(resolutions):
            desc_fake[lod] = finalize_descriptors(desc_fake[lod])
            scores.append(sliced_wasserstein(desc_real[lod],
                          desc_fake[lod], self.dir_repeats, self.dirs_per_repeat))
        del desc_fake

        return scores

    def print_real_res(self, resolutions, scores):
        # Print table header.
        print('%-32s' % 'Case'),
        for lod, res in enumerate(resolutions):
            print('%-12s' % ('%dx%d' % (res, res))),

        print('Average')
        print('%-32s' % '---'),

        for lod, res in enumerate(resolutions):
            print('%-12s' % '---'),
        print('---')
        print('%-32s' % 'reals'),
        for lod, res in enumerate(resolutions):
            print('%-12.6f' % scores[lod]),
        print('%.6f' % np.mean(scores))

    def print_fake_res(self, resolutions, scores):
        # Report results.
        for lod, res in enumerate(resolutions):
            print('%-12.6f' % scores[lod]),

        print('%.6f' % np.mean(scores))
        print('Done.')


# --------------------------------------------------------------------------------------------
#  Base functions taken from github with no or minor modifications.
# --------------------------------------------------------------------------------------------
def get_descriptors_for_minibatch(minibatch, nhood_size, nhoods_per_image ): 
    S = minibatch.shape  # (minibatch, channel, height, width)
    assert len(S) == 4 and S[1] == 3
    N = nhoods_per_image * S[0]
    H = nhood_size // 2
    nhood, chan, x, y = np.ogrid[0:N, 0:3, -H:H + 1, -H:H + 1]
    img = nhood // nhoods_per_image
    x = x + np.random.randint(H, S[3] - H, size=(N, 1, 1, 1))
    y = y + np.random.randint(H, S[2] - H, size=(N, 1, 1, 1))

    idx = ((img * S[1] + chan) * S[2] + y) * S[3] + x
    return minibatch.flat[idx]


def finalize_descriptors(desc):
    if isinstance(desc, list):
        desc = np.concatenate(desc, axis=0)
    assert desc.ndim == 4  # (neighborhood, channel, height, width)
    desc -= np.mean(desc, axis=(0, 2, 3), keepdims=True)
    desc /= np.std(desc, axis=(0, 2, 3), keepdims=True)
    desc = desc.reshape(desc.shape[0], -1)
    return desc


def sliced_wasserstein(A, B, dir_repeats, dirs_per_repeat):
    try:
        #assert A.ndim == 2 and A.shape == B.shape                           # (neighborhood, descriptor_component)
        assert len(np.shape(A)) == 2, "A does not have the expected number of dimensions."
        assert np.shape(A) == np.shape(B), "A and B don't have the same number of values"
    except Exception:
        import pdb; pdb.set_trace()
        raise RuntimeError('Sliced wasserstein failed because A and B did not have the same number of values.')
    results = []
    for repeat in range(dir_repeats):
        dirs = np.random.randn(A.shape[1], dirs_per_repeat)             # (descriptor_component, direction)
        dirs /= np.sqrt(np.sum(np.square(dirs), axis=0, keepdims=True)) # normalize descriptor components for each direction
        dirs = dirs.astype(np.float32)
        projA = np.matmul(A, dirs)                                      # (neighborhood, direction)
        projB = np.matmul(B, dirs)
        projA = np.sort(projA, axis=0)                                  # sort neighborhood projections for each direction
        projB = np.sort(projB, axis=0)
        dists = np.abs(projA - projB)                                   # pointwise wasserstein distances
        results.append(np.mean(dists))                                  # average over neighborhoods and directions
    return np.mean(results)                                             # average over repeats


def downscale_minibatch(minibatch, lod):
    if lod == 0:
        return minibatch
    t = minibatch.astype(np.float32)
    for i in range(lod):
        t = (t[:, :, 0::2, 0::2] + t[:, :, 0::2, 1::2] + t[:, :, 1::2, 0::2] + t[:, :, 1::2, 1::2]) * 0.25
    return np.round(t).clip(0, 255).astype(np.uint8)


gaussian_filter = np.float32([
    [1, 4,  6,  4,  1],
    [4, 16, 24, 16, 4],
    [6, 24, 36, 24, 6],
    [4, 16, 24, 16, 4],
    [1, 4,  6,  4,  1]]) / 256.0


def pyr_down(minibatch):  # matches cv2.pyrDown()
    assert minibatch.ndim == 4
    return scipy.ndimage.convolve(minibatch, gaussian_filter[np.newaxis, np.newaxis, :, :], mode='mirror')[:, :, ::2, ::2]


def pyr_up(minibatch):  # matches cv2.pyrUp()
    # {{{
    assert minibatch.ndim == 4
    S = minibatch.shape
    res = np.zeros((S[0], S[1], S[2] * 2, S[3] * 2), minibatch.dtype)
    res[:, :, ::2, ::2] = minibatch
    return scipy.ndimage.convolve(res, gaussian_filter[np.newaxis, np.newaxis, :, :] * 4.0, mode='mirror')


def generate_laplacian_pyramid(minibatch, num_levels):
    pyramid = [np.float32(minibatch)]
    for i in range(1, num_levels):
        pyramid.append(pyr_down(pyramid[-1]))
        pyramid[-2] -= pyr_up(pyramid[-1])
    return pyramid


def reconstruct_laplacian_pyramid(pyramid):
    minibatch = pyramid[-1]
    for level in pyramid[-2::-1]:
        minibatch = pyr_up(minibatch) + level
    return minibatch

