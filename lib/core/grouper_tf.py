import numpy as np
import tensorflow as tf
from tensorflow.keras import layers as tfl

# ==================================================================================================


class JointGrouper(tfl.Layer):  # pylint: disable=abstract-method
    def __init__(
        self, num_joints, input_size, hmap_threshold, refine=True, adjust=True
    ):
        super().__init__()

        self.num_joints = num_joints
        self.input_size = input_size
        self.hmap_threshold = hmap_threshold
        self.use_refine = refine
        self.use_adjust = adjust
        self.refine_threshold = 0.000000001
        self.nms_size = 5
        self.tag_threshold = 1.0

        jgo = [0, 1, 2, 3, 4, 5, 6, 11, 12, 7, 8, 9, 10, 13, 14, 15, 16]
        self.joint_group_order = jgo

        # import random
        # nrs = list(range(1, self.input_size[0] * self.input_size[1] + 1))
        # random.shuffle(nrs)
        # nrs = tf.reshape(nrs, [1, self.input_size[0], self.input_size[1], 1])
        # self.nms_random_array = tf.constant(nrs)

    # ==============================================================================================

    def nms(self, x):
        """Find fixed number of best keypoint proposals with their confidences"""

        # Add batch dimension for pooling
        x = tf.expand_dims(x, axis=0)

        # Filter positions with the highest value among their direct neighbors (local peaks).
        pmax = tf.nn.max_pool2d(x, ksize=self.nms_size, strides=1, padding="SAME")
        keep = tf.cast((pmax == x), dtype=x.dtype)
        peaks = x * keep

        # If there are directly neighboring peaks, select those with higher adjacent values, else we
        # could get two proposals in two neighboring pixels, which would cause errors in grouping.
        navg = tf.nn.avg_pool2d(x, ksize=self.nms_size, strides=1, padding="SAME")
        mask = tf.cast((peaks > 0), dtype=x.dtype)
        navg = navg * mask
        nmax = tf.nn.max_pool2d(navg, ksize=self.nms_size, strides=1, padding="SAME")
        keep = tf.cast((nmax == navg), dtype=x.dtype)
        peaks = peaks * keep

        # # If there are still neighboring peaks, because the heatmap spot is completely symmetric,
        # # choose one of them randomly
        # nrs = tf.cast(self.nms_random_array, x.dtype)
        # mask = tf.cast((peaks > 0), dtype=x.dtype)
        # rpks = nrs * mask
        # rmax = tf.nn.max_pool2d(rpks, ksize=self.nms_size, strides=1, padding="SAME")
        # keep = tf.cast((rmax == rpks), dtype=x.dtype)
        # peaks = peaks * keep

        # Remove batch dimension again
        x = peaks
        x = tf.squeeze(x, axis=0)

        # Find proposal positions
        x = tf.transpose(x, [2, 0, 1])
        tki = tf.where(x >= self.hmap_threshold)

        # Get heatmap scores for those proposals
        scores = tf.gather_nd(x, tki)
        scores = tf.expand_dims(scores, axis=-1)
        scores = tf.clip_by_value(scores, 0.0, 1.0)

        # Combine them to a single tensor
        tki = tf.cast(tki, scores.dtype)
        keypoints = tf.concat([tki, scores], axis=-1)

        # Convert keypoints to j-x-y-v order
        keypoints = tf.gather(keypoints, [0, 2, 1, 3], axis=-1)

        return keypoints

    # ==============================================================================================

    def add_tagscores(self, hmap_tag, keypoints):
        """Append values of tag heatmap to each keypoint"""

        # Scale back to smaller tag heatmap
        indices = keypoints[:, 1:3]
        indices = tf.floor(indices + 0.5)
        indices = tf.cast(indices, tf.int32)

        # Due to rounding the index could reach the size of the image, which is outside the heatmap
        hshape = tf.shape(hmap_tag)
        ix = tf.minimum(indices[:, 0], hshape[1] - 1)
        iy = tf.minimum(indices[:, 1], hshape[0] - 1)
        indices = tf.stack([iy, ix], axis=-1)

        # Add channel indices
        channels = keypoints[:, 0]
        channels = tf.expand_dims(tf.cast(channels, tf.int32), axis=-1)
        indices = tf.concat([indices, channels], axis=-1)

        # Extract tagscores
        tagscores = tf.gather_nd(hmap_tag, indices)

        # Combine them to a single tensor
        tagscores = tf.cast(tagscores, keypoints.dtype)
        tagscores = tf.expand_dims(tagscores, axis=-1)
        keypoints = tf.concat([keypoints, tagscores], axis=-1)

        return keypoints

    # ==============================================================================================

    def group_keypoints(self, keypoints, gscores, gweights):
        """Group keypoints to persons. Iterate through joints and match the new joints to the best
        matching person (nearest to average of tag values of already seen joints in this person)
        """

        infinite = tf.constant(np.inf, dtype=keypoints.dtype)
        channel = tf.cast(keypoints[:, 0], tf.int32)
        tagscores = keypoints[:, 4]

        # Placeholder for groups
        nprops = tf.cast(tf.shape(keypoints)[0], tf.int32)
        groups = tf.zeros([nprops], tf.int32)

        # Match group per joint
        for i in tf.constant(self.joint_group_order):
            tf.autograph.experimental.set_loop_options(
                shape_invariants=[
                    (gscores, tf.TensorShape([None])),
                    (gweights, tf.TensorShape([None])),
                ]
            )

            idx = tf.where(channel == i)
            idx = tf.squeeze(idx, axis=-1)

            jscores = tf.gather(tagscores, idx)
            jgroups = tf.zeros([nprops], tf.int32)
            num_pers = tf.reduce_sum(tf.cast(gscores < infinite, dtype=tf.int32))

            # Calculate distance of each joint proposal to each human average
            diff = tf.abs(gscores[None, ...] - jscores[..., None])

            fdiff = tf.reshape(diff, [-1])
            fshape = tf.shape(fdiff)

            fids = tf.where(tf.ones_like(fdiff) == 1)
            fids = tf.cast(tf.squeeze(fids, axis=-1), tf.int32)
            dwidth = tf.cast(tf.shape(diff)[1], tf.int32)

            # Match joint proposals to already existing persons or create new ones
            while tf.constant(True):
                if fshape[0] == 0:
                    # No proposals existing
                    break

                # Find the best matching joint-person pair. Unlike the other approaches the best
                # match for each person is selected instead of the best matches across all persons,
                # which they calculate with munkres (https://brc2.com/the-algorithm-workshop/).
                # Using sorting with removals instead of multiple argmin calls with removals was
                # much slower on average, even if infinite values where dropped beforehand.
                imin = tf.argmin(fdiff)
                imin = tf.cast(imin, tf.int32)
                vmin = fdiff[imin]

                if vmin == infinite:
                    # No proposals are left
                    break

                imin_y = tf.cast(
                    tf.floor(tf.cast(imin, tf.float32) / tf.cast(dwidth, tf.float32)),
                    tf.int32,
                )
                imin_x = imin - imin_y * dwidth
                imin = tf.stack([imin_y, imin_x], axis=-1)

                # Add found pair to group selection
                kidx = idx[imin[0]]
                kidx = tf.cast(kidx, tf.int32)
                if vmin < self.tag_threshold:
                    # Found a valid pair
                    gidx = imin[1] + 1
                else:
                    # Create a new person
                    gidx = num_pers + 1
                    num_pers = tf.add(num_pers, 1)
                jgroup = tf.expand_dims(gidx, axis=-1)
                jgroup = tf.pad(jgroup, [[kidx, nprops - kidx - 1]])
                jgroups = jgroups + jgroup

                # Mask out row and column of that minimum in the flattened array
                mask_col = tf.cast(fids % dwidth == imin[1], tf.bool)
                mask_row = tf.cast(
                    ((fids >= (imin[0] * dwidth)) & (fids < ((imin[0] + 1) * dwidth))),
                    tf.bool,
                )
                mask = tf.math.logical_or(mask_col, mask_row)
                fdiff = tf.where(mask, infinite, fdiff)

                # Set a defined shape
                fdiff = tf.reshape(fdiff, fshape)

            # More new joint proposals than persons
            matches = tf.gather(jgroups, idx)
            missing = tf.squeeze(tf.where(matches == 0), axis=-1)
            missing = tf.gather(idx, missing)
            missing = tf.cast(missing, tf.int32)
            for kidx in missing:
                gidx = num_pers + 1
                num_pers = tf.add(num_pers, 1)
                jgroup = tf.expand_dims(gidx, axis=-1)
                jgroup = tf.pad(jgroup, [[kidx, nprops - kidx - 1]])
                jgroups = jgroups + jgroup

            # Add matched joints to grouping placeholder
            groups = groups + jgroups

            # Pad gscores and gweights if new persons were added
            num_old = tf.shape(gscores)[0]
            if num_pers > num_old:
                gscores = tf.pad(
                    gscores, [[0, num_pers - num_old]], constant_values=infinite
                )
                gweights = tf.pad(gweights, [[0, num_pers - num_old]])

            # Update group scores
            mgroups = tf.gather(jgroups, idx) - 1
            mgroups = tf.expand_dims(mgroups, axis=-1)
            mscores = tf.scatter_nd(mgroups, jscores, [num_pers])
            mweights = tf.gather(keypoints[:, 3], idx)
            mweights = tf.scatter_nd(mgroups, mweights, [num_pers])
            if tf.reduce_sum(mweights) > 0:
                # Calculate weighted average
                gscores = gscores * gweights + mscores * mweights
                gscores = tf.where(tf.math.is_nan(gscores), mscores * mweights, gscores)
                gweights = gweights + mweights
                gscores = gscores / gweights
                gscores = tf.where(gweights == 0.0, infinite, gscores)
            else:
                # Keep initial scores
                pass

        groups = tf.expand_dims(groups, axis=-1)
        return groups

    # ==============================================================================================

    def refine(self, hmap_tag, hmap_avg, keypoints):
        """Add keypoints for missing joints even if they have low confidence scores in the case
        that the tagmap matches its corresponding position the the crrent person"""

        if tf.shape(keypoints)[0] == 0:
            return keypoints

        groups = keypoints[:, 5]
        tagscores = keypoints[:, 4]
        confidence = keypoints[:, 3]
        num_pers = tf.reduce_max(groups)

        extra_kps = tf.zeros([0, 6], keypoints.dtype)
        for i in tf.range(start=1, limit=num_pers + 1):
            tf.autograph.experimental.set_loop_options(
                shape_invariants=[
                    (extra_kps, tf.TensorShape([None, 6])),
                ]
            )

            # Find keypoints associated to the current group
            idx = tf.where(groups == i)

            # Calculate averaged tagscore (groupscore) of this person
            tscores = tf.gather(tagscores, idx)
            hscores = tf.gather(confidence, idx)
            avg_score = tf.reduce_sum(tscores * hscores) / tf.reduce_sum(hscores)

            # Find heatmap patches where the tagscore is close to the current person
            # Substracting a small value from the tag difference (similar to the rounding in the
            # other grouping algorithms), to reduce the impact of small tagscore variations,
            # improved AP and AR slightly
            diff = tf.abs(hmap_tag - avg_score)
            diff = hmap_avg - tf.maximum(0.0, diff - self.tag_threshold / 4)

            # Flatten and find maximum for each joint
            fdiff = tf.reshape(diff, [-1, self.num_joints])
            dwidth = tf.cast(tf.shape(diff)[1], tf.int32)
            imax = tf.argmax(fdiff)

            # Calculate 2D position of maximas
            imax = tf.cast(imax, tf.int32)
            imax_y = tf.cast(
                tf.floor(tf.cast(imax, tf.float32) / tf.cast(dwidth, tf.float32)),
                tf.int32,
            )
            imax_x = imax - imax_y * dwidth
            imax = tf.stack([imax_y, imax_x], axis=-1)

            # Drop scores for already existing joints
            indices = tf.gather(keypoints[:, 0], idx)
            indices = tf.cast(indices, tf.int32)
            mask = tf.scatter_nd(
                indices, tf.ones_like(tf.squeeze(indices, axis=1)), [self.num_joints]
            )
            mask = 1 - mask
            mask = tf.cast(mask, diff.dtype)
            mask = tf.reshape(mask, [1, 1, -1])
            hmap = hmap_avg * mask

            # Get heatmap scores for those positions
            channels = tf.constant(list(range(self.num_joints)), imax.dtype)
            channels = tf.reshape(channels, [-1, 1])
            indices = tf.concat([imax, channels], axis=-1)
            vmax = tf.gather_nd(hmap, indices)

            # Combine to a single tensor
            gs = tf.zeros_like(vmax) + i
            avg_score = tf.zeros_like(vmax) + avg_score
            imax_x = tf.cast(imax_x, keypoints.dtype)
            imax_y = tf.cast(imax_y, keypoints.dtype)
            channels = tf.squeeze(tf.cast(channels, keypoints.dtype), axis=1)
            vmax = tf.cast(vmax, keypoints.dtype)
            gs = tf.cast(gs, keypoints.dtype)
            new_kps = tf.stack([channels, imax_x, imax_y, vmax, avg_score, gs], axis=-1)

            # Filter and add to collection
            vmask = tf.cast(vmax >= self.refine_threshold, keypoints.dtype)
            indices = tf.squeeze(tf.where(vmask), axis=1)
            new_kps = tf.gather(new_kps, indices)
            extra_kps = tf.concat([extra_kps, new_kps], axis=0)

        # Combine new keypoints with the old ones. Use the last column to mark refined keypoints.
        zeros = tf.zeros([tf.shape(extra_kps)[0], 1], keypoints.dtype)
        extra_kps = tf.concat([extra_kps, zeros], axis=-1)
        keypoints = tf.concat([keypoints, extra_kps], axis=0)

        return keypoints

    # ==============================================================================================

    def signop(self, x):
        """Helper to calculate sign op because it is not supported by tflite"""

        ispositive = tf.cast(x > 0.0, dtype=x.dtype)
        isnegative = tf.cast(x < 0.0, dtype=x.dtype)

        sign = -1.0 * isnegative + 1.0 * ispositive
        return sign

    # ==============================================================================================

    def adjust(self, hmap_avg, keypoints):
        """Move keypoints a bit into the direction of their highest neighbour"""

        if tf.shape(keypoints)[0] == 0:
            return keypoints

        hshape = tf.cast(tf.shape(hmap_avg), tf.int32)
        kpx = tf.cast(keypoints[:, 1], tf.int32)
        kpy = tf.cast(keypoints[:, 2], tf.int32)

        # Which keypoints have all 4 neighbouring pixels?
        maskx = tf.cast(((kpx >= 1) & (kpx <= hshape[1] - 2)), keypoints.dtype)
        masky = tf.cast(((kpy >= 1) & (kpy <= hshape[0] - 2)), keypoints.dtype)

        # Make sure there are no keypoints on a boarder side
        ky = tf.maximum(1, tf.minimum(kpy, hshape[0] - 2))
        kx = tf.maximum(1, tf.minimum(kpx, hshape[1] - 2))

        # Build indices of neighbouring pixels
        kpc = tf.cast(keypoints[:, 0], tf.int32)
        k1 = tf.stack((ky, kx + 1, kpc), axis=1)
        k2 = tf.stack((ky, kx - 1, kpc), axis=1)
        k3 = tf.stack((ky + 1, kx, kpc), axis=1)
        k4 = tf.stack((ky - 1, kx, kpc), axis=1)

        # Calculate difference between left-right and top-bottom pixels to get a move direction
        diffx = tf.gather_nd(hmap_avg, k1) - tf.gather_nd(hmap_avg, k2)
        diffy = tf.gather_nd(hmap_avg, k3) - tf.gather_nd(hmap_avg, k4)

        # Move keypoints if they aren't edge pixels
        kpx = keypoints[:, 1] + maskx * self.signop(diffx) * 0.25
        kpy = keypoints[:, 2] + masky * self.signop(diffy) * 0.25

        # Add offset term to compensate rounding errors
        kpx = kpx + 0.5
        kpy = kpy + 0.5

        # Combine to single tensor again
        values = keypoints[:, 3:]
        kp = tf.stack([keypoints[:, 0], kpx, kpy], axis=-1)
        keypoints = tf.concat([kp, values], axis=-1)

        return keypoints

    # ==============================================================================================

    def reorder_keypoints(self, keypoints):
        """Reorder keypoints according to group matching"""

        if tf.shape(keypoints)[0] == 0:
            return tf.zeros([1, self.num_joints, 5], keypoints.dtype)

        keypoints = tf.gather(keypoints, [0, 5, 1, 2, 3, 4, 6], axis=-1)
        channel = keypoints[:, 0]
        groups = keypoints[:, 1]
        values = keypoints[:, 2:]

        num_pers = tf.reduce_max(groups)
        groups = groups - 1

        indices = tf.stack([groups, channel], axis=-1)
        indices = tf.cast(indices, tf.int32)

        # Insert found keypoints into empty placeholder tensor
        keypoints = tf.scatter_nd(indices, values, [num_pers, self.num_joints, 5])

        return keypoints

    # ==============================================================================================

    @tf.function(
        input_signature=[
            [
                tf.TensorSpec([1, None, None, None], tf.float32),
                tf.TensorSpec([1, None, None, None], tf.float32),
            ]
        ]
    )
    def call(self, x):  # pylint: disable=arguments-differ
        hmap_tag = x[0]
        hmap_avg = x[1]

        # Drop batch dimension
        hmap_tag = hmap_tag[0]
        hmap_avg = hmap_avg[0]

        # Get best keypoint proposals
        keypoints = self.nms(hmap_avg)

        # Get tagmap values
        keypoints = self.add_tagscores(hmap_tag, keypoints)

        # Match keypoints to groups
        gscores = tf.zeros([0], keypoints.dtype)
        gweights = tf.zeros_like(gscores)
        groups = self.group_keypoints(keypoints, gscores, gweights)

        # Append groups to keypoints
        groups = tf.cast(groups, keypoints.dtype)
        keypoints = tf.concat([keypoints, groups], axis=-1)

        # Add an new column to distinguish between normal and refined keypoints that are added
        # in the next step for later calculation of the person's complete detection score.
        ones = tf.ones([tf.shape(keypoints)[0], 1], keypoints.dtype)
        keypoints = tf.concat([keypoints, ones], axis=-1)

        # Add missing joints with low confidences
        if self.use_refine:
            keypoints = self.refine(hmap_tag, hmap_avg, keypoints)

        # Adjust positions depending on neighbour pixels
        if self.use_adjust:
            keypoints = self.adjust(hmap_avg, keypoints)

        # Reorder keypoints into the proposed groups
        keypoints = self.reorder_keypoints(keypoints)

        # Add batch dimension again for consistency
        keypoints = tf.expand_dims(keypoints, axis=0)

        return keypoints
