from .. import Block, Image

from twirl.utils import find_transform
from skimage.transform import AffineTransform
import numpy as np
from itertools import product
from skimage.measure import LineModelND
from sklearn.metrics import r2_score

__all__ = ["TargetTrack"]

class TargetTrack(Block):
    
    def __init__(self, tolerance=2, n=10, threshold=2.5):
        super().__init__(sliding=True)

        self.tolerance = tolerance
        self.n = n
        self.threshold = threshold

        self.unknown_sets: list = []
        self.tforms: list = []
        self.ref_date = None
        self.lines: list = None
        self.models: list = None
    
    def _unknown_set(self, coords, sources, date):
        uniform_time = (date - self.ref_date).total_seconds()
        ids = np.array([s.i for s in sources])
        ids = ids[~sources.star_mask]

        unknown = np.concatenate((ids.reshape(-1, 1), coords), axis=1)
        unknown = np.insert(unknown, unknown.shape[1], uniform_time, axis=1)
        
        return unknown

    def run(self, image: Image, sliding_images):
        """
        Align adjacent images to identify stars and save all unknowns,
        which might be track target, noise, or star that can't be identified
        """

        if sliding_images is None:
            raise ValueError("TargetTrack block needs to be used with SlidingSequence.")

        if self.ref_date is None:
            self.ref_date = image.date

        prev_unknown_set = None
        curr_unknown_set = None
        
        curr_src = image.sources
        if not hasattr(curr_src, 'star_mask'):
            curr_src.star_mask = np.zeros(len(curr_src), dtype=bool)
        if not hasattr(curr_src, 'tform'):
            curr_src.tform = np.identity(3)

        # Align current image to previous image, mark stars and compute transform matrix 
        # that aligns current image to the first image
        prev_image, = sliding_images
        if prev_image is not None:

            prev_src = prev_image.sources
            if not hasattr(prev_src, 'star_mask'):
                prev_src.star_mask = np.zeros(len(prev_src), dtype=bool)
            if not hasattr(prev_src, 'tform'):
                prev_src.tform = np.identity(3)
                curr_src.tform = prev_src.tform

            prev_coords = prev_src.coords.copy()
            curr_coords = curr_src.coords.copy()
            tform = find_transform(curr_coords, prev_coords, tolerance=self.tolerance, n=self.n)

            if tform is not None:
                # align current image to previous image
                aligned = AffineTransform(tform)(curr_coords)

                # use aligned and previous coords difference to find stars
                diff = np.linalg.norm(aligned[:, np.newaxis, :] - prev_coords[np.newaxis, :, :], axis=2)
                masks = diff < self.threshold

                # update star masks
                prev_src.star_mask |= masks.any(axis=0)
                curr_src.star_mask |= masks.any(axis=1)

                # align previous image to first image
                aligned = AffineTransform(prev_src.tform)(prev_coords)
                aligned = aligned[~prev_src.star_mask]

                # update unknown set of previous image
                prev_unknown_set = self._unknown_set(aligned, prev_src, prev_image.date)

                # save transform that align to first image for current image
                curr_src.tform = tform @ prev_src.tform

                # align current image to first image
                aligned = AffineTransform(curr_src.tform)(curr_coords)
                aligned = aligned[~curr_src.star_mask]

                # save unknown set of current image
                curr_unknown_set = self._unknown_set(aligned, curr_src, image.date)

            self.unknown_sets[-1] = prev_unknown_set
        
        self.unknown_sets.append(curr_unknown_set)
        self.tforms.append(curr_src.tform)


    def terminate(self):
        self.lines, self.models = unknown_target_assocate(self.unknown_sets)

    def movie(self):
        pass

    def plot(self, shape, cmap=None, which='all', view=(16, -51)):

        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection

        plt.style.use('default')
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        if which == 'all':
            sub_sets = self.unknown_sets
            tforms = self.tforms
        elif isinstance(which, list):
            sub_sets = [self.unknown_sets[i] for i in which]
            tforms = [self.tforms[i] for i in which]
        elif isinstance(which, slice):
            sub_sets = self.unknown_sets[which]
            tforms = self.tforms[which]
        
        if cmap is None:
            cmap = plt.cm.tab10

        colors = cmap(np.linspace(0, 1, len(sub_sets)))
        for sets, tform, color in zip(sub_sets, tforms, colors):
            # plot aligned plane
            x = np.array([0, shape[0], shape[0], 0])
            y = np.array([0, 0, shape[1], shape[1]])
            z = np.full((4,), sets[0, 3])
            aligned = AffineTransform(tform)(np.stack([x, y], axis=1))
            vertices = [list(zip(aligned[:, 0], aligned[:, 1], z))]
            poly = Poly3DCollection(vertices, alpha=0.1, facecolor=color, edgecolor='k', linewidth=0.5)
            ax.add_collection3d(poly)

            # plot unknown points
            x = sets[:, 1]
            y = sets[:, 2]
            z = sets[:, 3]
            ax.scatter(x, y, z, s=2, marker='.', alpha=0.8, color=color)

        # plot predicts points
        times = [sets[0, 3] for sets in self.unknown_sets]
        times.insert(0, -1)
        times.append(times[-1] + 1)
        colors = cmap(np.linspace(0, 1, len(self.models)))
        for model, color in zip(self.models, colors):
            predicts = model.predict(times, axis=2)
            ax.plot(predicts[:, 0], predicts[:, 1], predicts[:, 2], color=color, 
                    linewidth=0.5, alpha=0.9, marker='.', markersize=3)

        ax.view_init(elev=view[0], azim=view[1])
        fig.tight_layout()


def unknown_target_assocate(
    unknown_sets,
    base_num=3,
    fit_threshold=0.95,
    predict_threshold=2.5,
    line_num_threshold=6
):
    """unknown target assocate
    first use first `base_num` unknown sets to estimate line model, and then use
    the line model to predict at next time, if the predict value and unknown point
    distance is less than `predict_threshold`, then the unknown point is assocated
    to the line model. if the line model has more than `line_num_threshold` points,
    then the line model is saved.

    Parameters
    ----------
    unknown_sets : list of ndarray
        unknown sets, shape is (n, 4), n is the number of unknown points, 4 is
        (id, x, y, time)
    base_num : int, optional
        the number of unknown sets to estimate line model, by default 3
    fit_threshold : float, optional
        the threshold of line model fit score, by default 0.95
    predict_threshold : float, optional
        the threshold of line model predict distance, by default 2.5
    """
    frame_num = len(unknown_sets)
    base_sets = unknown_sets[:base_num]
    check_sets = unknown_sets[base_num:]

    lines = []
    models = []
    for samples in product(*base_sets):

        # check base unknown sets
        samples = np.array(samples)

        # get datas
        ids = samples[:, 0]
        times = samples[:, 3]
        fit_datas = np.array(samples[:, 1:])

        # estimate line model
        model = LineModelND()
        model.estimate(fit_datas)
        predict = model.predict(times, axis=2)
        score = r2_score(fit_datas, predict)

        # check fit score
        if score < fit_threshold:
            continue

        # candidate associate line
        line = np.full((frame_num, ), np.nan)
        line[:base_num] = ids

        # check other unknown sets
        for i, sets in enumerate(check_sets):

            ids = sets[:, 0]
            times = sets[0, 3]
            datas = sets[:, 1:]

            predict = model.predict(times, axis=2)

            # if predict value is out of range, then skip
            minx, miny, _ = np.min(datas, axis=0)
            maxx, maxy, _ = np.max(datas, axis=0)
            if predict[0] < minx - predict_threshold or predict[0] > maxx + predict_threshold:
                continue
            if predict[1] < miny - predict_threshold or predict[1] > maxy + predict_threshold:
                continue

            # find nearest point
            diff = np.linalg.norm(predict - datas, axis=-1)
            idx = np.argmin(diff)

            # if nearest point is less than threshold, then assocate
            if diff[idx] < predict_threshold:
                line[base_num + i] = ids[idx]
                fit_datas = np.append(fit_datas, [datas[idx]], axis=0)
                model.estimate(fit_datas)

        # check line point number, whther save line model
        if np.count_nonzero(~np.isnan(line)) > line_num_threshold:
            lines.append(line)
            models.append(model)

    return lines, models