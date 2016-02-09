from __future__ import division, print_function

from collections import namedtuple
import re
from .dataset import Landmark

class View(object):
    __slots__ = ('id', 'time', 'position', 'orientation')

    def __init__(self, _id, time, position, orientation):
        self.id = _id
        self.time = time
        self.position = position
        self.orientation = orientation

class SfmResultError(Exception):
    pass

class SfmResult(object):
    next_view_id = 0
    next_landmark_id = 0

    @staticmethod
    def frame_from_filename(filename):
        m = re.findall(r'(\d+)\.[\w]+\b', filename)
        try:
            return int(m[-1])
        except (ValueError, IndexError):
            raise SfmResultError("Could not extract frame number from {}".format(filename))

    def __init__(self):
        self.views = []
        self.landmarks = []

    def add_view(self, time, position, orientation):
        _id = self.next_view_id
        self.next_view_id += 1

        view = View(_id, time, position, orientation)
        self.views.append(view)
        return _id

    def remap_views(self):
        remap = {}
        sorted_views = []
        for new_id, view in enumerate(sorted(self.views, key=lambda v: v.time)):
            remap[view.id] = new_id
            view.id = new_id
            sorted_views.append(view)
        self.views = sorted_views

        for lm in self.landmarks:
            remapped_observations = {
                remap[v_id] : measurement for v_id, measurement in lm.observations.items()
            }
            lm.observations = remapped_observations

    def add_landmark(self, position, visibility):
        for view_id in visibility:
            try:
                view = self.views[view_id]
            except IndexError:
                raise SfmResultError("No such view: {:d}".format(view_id))
        _id = self.next_landmark_id
        self.next_landmark_id += 1
        lm = Landmark(_id, position, visibility)
        self.landmarks.append(lm)
        return _id