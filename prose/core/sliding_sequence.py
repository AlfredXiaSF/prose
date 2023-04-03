from .image import FITSImage
from .sequence import Sequence
from pathlib import Path

class SlidingSequence(Sequence):
    def __init__(self, blocks, sliding_size=2, name=None):
        """A sequence of :py:class:`Block` objects to sequentially process images with sliding window

        Parameters
        ----------
        blocks : list
            list of :py:class:`Block` objects
        sliding_size : list
            sliding window size
        name : str, optional
            name of the sequence, by default None
        """
        super().__init__(blocks, name=name)
        self.sliding_size = sliding_size

    def _run(self, loader=FITSImage):
        def sliding_window(lst, n):
            if n > len(lst):
                return [lst + [None] * (n - len(lst))]
            return [lst[i:i+n] for i in range(len(lst) - n + 1)]
        
        def load_image(image):
            if image is None:
                return None
            
            idx, img = image
            if isinstance(img, (str, Path)):
                image = loader(img)
                image.i = idx
            return image
            
        def run_image(image, sliding_images=None):
            if image is None:
                return
            
            for block in self.blocks:
                # if sliding images is None, then only run non sliding window blocks
                if sliding_images is None and not block.sliding:
                    block._run(image)
                # if sliding images is not None, then only run sliding window blocks
                elif sliding_images is not None and block.sliding:
                    block._run(image, sliding_images)

                # This allows to discard image in any Block
                if image.discard:
                    self._add_discard(type(block).__name__, image.i)
                    break
        
        def run_images(images, sliding_images=None):
            if isinstance(images, list):
                for image in images:
                    run_image(image, sliding_images)
            else:
                run_image(images, sliding_images)

        # to return something like [[None, None, 1], [None, 1, 2], [1, 2, 3], [2, 3, 4], [3, 4, 5]]
        # if provide [1, 2, 3, 4, 5], sliding size is 3
        img_lst = [None] * (self.sliding_size - 1) + [(i, img) for i, img in enumerate(self.images)]
        sliding_lst = sliding_window(img_lst, self.sliding_size)

        for images in self.progress(sliding_lst):

            # non sliding window blocks processing
            if self.last_image is None:
                sliding_images = [load_image(img) for img in images[:-1]]
                run_images(sliding_images)
            else:
                sliding_images = self.last_image

            curr_image = load_image(images[-1])
            run_images(curr_image)

            # sliding window block processing
            # sliding window block may depend on non sliding window block, so 
            # first run non sliding window block, then run sliding window block
            run_images(curr_image, sliding_images)

            self.last_image = sliding_images[1:] + [curr_image]

            del sliding_images[0]
            self.n_processed_images += 1