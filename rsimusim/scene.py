from __future__ import division, print_function

from imusim.environment.base import Environment


class SceneEnvironment(Environment):
    def __init__(self, dataset, **kwargs):
        self.scene = dataset
        super(SceneEnvironment, self).__init__(**kwargs)

    def observe(self, t, position, orientation):
        landmarks = self.scene.visible_landmarks(t)
        return landmarks