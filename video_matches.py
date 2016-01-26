#!/usr/bin/env python
from __future__ import print_function, division

from collections import deque
import os
import struct
import logging
import itertools
import crisp.tracking
logging.basicConfig(level=logging.DEBUG)

import numpy as np
import cv2
from crisp import VideoStream, AtanCameraModel

class Track(object):
    _next_id = 0

    def __init__(self):
        self.positions = []
        self.start = None
        self.last_retrack = None
        self.id = self.__class__._next_id
        self.__class__._next_id += 1

    def __len__(self):
        return len(self.positions)

    def __contains__(self, framenum):
        return self.start <= framenum < self.start + len(self)

    def __getitem__(self, framenum):
        offset = framenum - self.start
        if 0 <= offset < len(self.positions):
            return self.positions[framenum - self.start]
        else:
            raise IndexError("Track does not have frame number {:d}".format(framenum))

    def __repr__(self):
        return '<Track #{:d}>'.format(self.id)

    @property
    def last_framenum(self):
        return self.start + len(self.positions) - 1

    @property
    def first_framenum(self):
        return self.start

    @property
    def last_position(self):
        return self.positions[-1]

    def add_measurement(self, frame, position, color=None):
        if self.start is None:
            self.start = frame
        elif not frame == self.start + len(self):
            raise ValueError("Expected next measurement to be frame {:d}, got {:d}".format(self.start + len(self), frame))
        self.positions.append(position)

class FrameStore(object):
    def __init__(self, max_size=None):
        self._last_framenum = None
        self._frames = deque(maxlen=max_size)

    def __len__(self):
        return len(self._frames)

    def __getitem__(self, framenum):
        start = self._last_framenum - len(self._frames) + 1
        idx = framenum - start
        if 0 <= idx < len(self._frames):
            return self._frames[framenum - start]
        else:
            raise IndexError("Store does not contain frame {:d}".format(framenum))

    def add(self, framenum, frame):
        if self._last_framenum is not None and not framenum == self._last_framenum + 1:
            raise ValueError("Expected next framenum {:d}, got {:d}".format(self._last_framenum + 1, framenum))
        self._frames.append(frame)
        self._last_framenum = framenum

    def items(self):
        for i, frame in enumerate(self._frames):
            yield self._last_framenum - len(self._frames) + i + 1, frame

class VideoTracker(object):
    def __init__(self, stream, back_track=5, min_tracks=300, min_distance=10.):
        self.stream = stream
        self.min_tracks = min_tracks
        self.min_distance = min_distance
        self.back_track = back_track
        self.back_track_distance = 0.5 # pixels
        self.active_tracks = []
        self.dropped_tracks = []
        self.stored_tracks = []
        self.frames = FrameStore(back_track)

    def run(self, max_frame=None, start=0, visualize=False):
        final_framenum = None
        if visualize:
            cv2.namedWindow("tracks", cv2.WINDOW_AUTOSIZE)

        for framenum, frame in enumerate(self.stream):
            final_framenum = framenum
            if framenum < start:
                continue
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            self.frames.add(framenum, frame)

            # Track from last to current frame
            self.forward_track(framenum, frame)

            # Run retrack step
            self.retrack(framenum)

            # Fill out with more tracks if necessary
            if len(self.active_tracks) < self.min_tracks:
                self.more_tracks(framenum, frame)

            if visualize:
                draw = np.dstack((frame, frame, frame))
                for t in self.active_tracks:
                    cv2.circle(draw, tuple(t[framenum]), 10, (255, 0, 0))
                for t in self.stored_tracks:
                    try:
                        cv2.circle(draw, tuple(t[framenum]), 10, (0, 255, 255))
                    except IndexError:
                        pass
                label = 'Frame: {:d} - Active: {:d}, Stored: {:d}'.format(framenum,
                                                                          len(self.active_tracks),
                                                                          len(self.stored_tracks))
                font_size = 2
                thickness = 3
                (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_COMPLEX, font_size, thickness)
                rows, cols = frame.shape
                textpos = cols - tw - 10, th + 10
                cv2.putText(draw, label, textpos, cv2.FONT_HERSHEY_COMPLEX, font_size, (0, 0, 255), thickness=thickness)
                cv2.imshow("tracks", draw)
                if cv2.waitKey(1) == 27:
                    print('User break')
                    break

            if max_frame is not None and framenum > max_frame:
                break

        # Retrack all active tracks and save those still alive
        for t in self.active_tracks:
            t.last_retrack = None # Force a retrack
        self.retrack(final_framenum)
        self.stored_tracks.extend(self.active_tracks)
        self.active_tracks = []

    def more_tracks(self, framenum, frame):
        logging.info("Frame %d: Fetching more points", framenum)
        max_corners = self.min_tracks + len(self.active_tracks)
        quality_level = 0.07
        new_points = cv2.goodFeaturesToTrack(frame, max_corners, quality_level, self.min_distance)
        if self.active_tracks:
            current_points = np.vstack([t[framenum] for t in self.active_tracks])
        else:
            current_points = np.empty((0,2))
        assert current_points.shape == (len(self.active_tracks), 2)

        new_tracks = []
        for point in new_points:
            distances = np.linalg.norm(current_points - point, axis=1)
            assert distances.size == len(self.active_tracks)
            if np.all(distances > self.min_distance):
                track = Track()
                track.add_measurement(framenum, point.reshape(2))
                new_tracks.append(track)
                if len(new_tracks) + len(self.active_tracks) > self.min_tracks:
                    break
        self.active_tracks.extend(new_tracks)
        logging.info("Frame %d: Added %d new tracks", framenum, len(new_tracks))

    def forward_track(self, framenum, frame):
        if not self.active_tracks or len(self.frames) < 2:
            logging.debug("No active tracks, or not enough frames")
            return

        last_framenum = framenum - 1
        logging.debug("Frame %d: Tracking %d points from %d to %d", framenum, len(self.active_tracks), last_framenum, framenum)
        prev_frame = self.frames[last_framenum]
        prev_points = np.vstack([t[last_framenum].reshape(1,1,2) for t in self.active_tracks])
        next_points, status, err = cv2.calcOpticalFlowPyrLK(prev_frame, frame, prev_points, np.array([]))
        new_active = []
        self.dropped_tracks = []
        for track, point, valid in zip(self.active_tracks, next_points, status):
            if valid == 1:
                track.add_measurement(framenum, point.reshape(2))
                new_active.append(track)
            else:
                self.dropped_tracks.append(track)
        self.active_tracks = new_active
        logging.debug("Frame %d: After tracking, %d tracks remain", framenum, len(self.active_tracks))

    def retrack(self, framenum):
        def needs_retrack(t):
            long_enough = len(t) >= self.back_track
            if t.last_retrack is None:
                return long_enough
            else:
                since_retrack = framenum - t.last_retrack
                return since_retrack >= self.back_track and long_enough

        candidates = []
        candidates.extend((t for t in self.dropped_tracks if len(t) >= self.back_track))
        candidates.extend((t for t in self.active_tracks if needs_retrack(t)))
        print(len(candidates), len(self.active_tracks), len(self.dropped_tracks), [t.id for t in candidates])

        if not candidates:
            logging.info('Frame %d: Nothing to retrack', framenum)
            return

        def tracks_to_points(track_list, framenum):
            return np.vstack([t[framenum].reshape(1,1,2) for t in track_list])

        prev_frame = None
        alive_tracks = []
        success_tracks = []
        prev_pts = np.empty((0,1,2), dtype='float32')
        for cur_framenum, frame in reversed(list(self.frames.items())):
            if alive_tracks:
                pts, status, err = cv2.calcOpticalFlowPyrLK(prev_frame, frame, prev_pts, np.array([]))
                to_keep = []
                for i, (track, pt, status) in enumerate(zip(alive_tracks, pts, status)):
                    pt = pt.reshape(2)
                    distance = np.linalg.norm(pt - track[cur_framenum])
                    if status == 1 and distance <= self.back_track_distance:
                        prev_pts[i, 0] = pt
                        if track.first_framenum == cur_framenum:
                            success_tracks.append(track)
                        else: # Keep tracking
                            to_keep.append(i)
                    elif status == 1:
                        logging.debug("Dropping track %d with distance %.1f", track.id, distance)

                alive_tracks = [alive_tracks[i] for i in to_keep]
                prev_pts = prev_pts[to_keep]

            # Add new tracks
            starting_tracks = [t for t in candidates if t.last_framenum == cur_framenum]
            if starting_tracks:
                starting_pts = tracks_to_points(starting_tracks, cur_framenum)
                alive_tracks.extend(starting_tracks)
                prev_pts = np.vstack((prev_pts, starting_pts))
            prev_frame = frame

        success_tracks.extend(alive_tracks)
        new_stored_count = 0
        for t in self.dropped_tracks:
            if t in success_tracks:
                self.stored_tracks.append(t)
                new_stored_count += 1

        new_active = []
        for t in self.active_tracks:
            if t in candidates:
                if t in success_tracks:
                    new_active.append(t)
            else:
                t.last_retrack = framenum
                new_active.append(t)

        self.active_tracks = new_active
        logging.info('Frame %d: Back tracked %d of %d points, stored %d, %d still active',
            framenum, len(success_tracks), len(candidates), new_stored_count, len(self.active_tracks))

    def save_visualsfm(self, output_directory, fake_frames=False):
        min_frame = min(t.first_framenum for t in self.stored_tracks)
        max_frame = max(t.last_framenum for t in self.stored_tracks)
        frame_numbers = range(min_frame, max_frame + 1)

        descriptor_to_track = {} # frame -> (track id -> descriptor id)
        def chars_to_int(s):
            assert len(s) == 4
            res = 0
            for i, c in enumerate(s):
                res += ord(c) << (8 * i)
            return res

        name = chars_to_int('SIFT') #ord('S') + (ord('I') << 8) + (ord('F') << 16) + (ord('T') << 24)
        version = chars_to_int('V4.0') #ord('V') + ord('4') << 8 + ord('.') << 16 + ord('0') << 24
        eof = struct.pack('i', chars_to_int(chr(0xff) + 'EOF'))
        zeropad = int(np.log10(max_frame) + 0.5) + 1

        for framenum in frame_numbers:
            local_tracks = [t for t in self.stored_tracks if framenum in t]
            id_map = {}
            frame_fileroot = os.path.join(output_directory, 'frame_{n:0{zeropad}d}'.format(n=framenum, zeropad=zeropad))
            sift_filename = frame_fileroot + '.sift'
            frame_filename = frame_fileroot + '.jpg'
            if fake_frames:
                cv2.imwrite(frame_filename, np.zeros((128, 128), dtype='uint8'))
                logging.debug('Wrote fake frame %s', frame_filename)

            npoint = len(local_tracks)
            with open(sift_filename, 'wb') as f:
                header = struct.pack('5i', name, version, npoint, 5, 128)
                f.write(header)
                # Location Data
                for local_id, track in enumerate(local_tracks):
                    x, y = track[framenum].astype('float32')
                    location = struct.pack('5f', x, y, 0., 0., 0.) # color, scale, orientation not set
                    f.write(location)
                    id_map[track.id] = local_id
                # Descriptor Data
                fake_descriptor = struct.pack('128B', *([0]*128))
                for _ in range(npoint):
                    f.write(fake_descriptor)
                # EOF
                f.write(eof)
            logging.info('Wrote %s with %d descriptors', sift_filename, npoint)
            descriptor_to_track[framenum] = id_map

        # Match file
        match_filename = os.path.join(output_directory, 'matches.txt')
        frame_pairs = itertools.combinations(frame_numbers, 2)
        with open(match_filename, 'w') as f:
            for frame_1, frame_2 in frame_pairs:
                tracks_1 = set(descriptor_to_track[frame_1].keys())
                tracks_2 = set(descriptor_to_track[frame_2].keys())
                mutual = tracks_1.intersection(tracks_2)
                if mutual:
                    logging.debug('Writing %d matches for pair %d, %d', len(mutual), frame_1, frame_2)
                    f.write('frame_{f1:0{zeropad}d}.jpg frame_{f2:0{zeropad}d}.jpg {nmatch:d}\n'.format(
                        f1=frame_1, f2=frame_2, zeropad=zeropad, nmatch=len(mutual)
                    ))
                    f1_descs = ' '.join([str(descriptor_to_track[frame_1][track_id]) for track_id in mutual])
                    f2_descs = ' '.join([str(descriptor_to_track[frame_2][track_id]) for track_id in mutual])
                    f.write(f1_descs)
                    f.write('\n')
                    f.write(f2_descs)
                    f.write('\n')
        logging.info('Wrote %s', match_filename)



if __name__ == "__main__":
    VIDEO_PATH = '/home/hannes/Datasets/gopro-gyro-dataset/rccar.MP4'
    CAMERA_PATH = '/home/hannes/Code/crisp/hero3_atan.hdf'

    camera_model = AtanCameraModel.from_hdf(CAMERA_PATH)
    video = VideoStream.from_file(camera_model, VIDEO_PATH)
    tracker = VideoTracker(video)
    tracker.run(start=100, max_frame=120, visualize=False)
    tracker.save_visualsfm('/tmp/output', fake_frames=True)
    #tracker.run(visualize=True)
    print('Stored tracks: {:d}, still active: {:d}'.format(len(tracker.stored_tracks), len(tracker.active_tracks)))
    import shutil
    #for t in tracker.stored_tracks:
    #    print(t)
    #shutil.rmtree('tempout')
    #tracker.save('tempout')
