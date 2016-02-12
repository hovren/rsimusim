from __future__ import print_function
import argparse
import time

from rsimusim.dataset import Dataset

parser = argparse.ArgumentParser()
parser.add_argument("dataset")
parser.add_argument("--axes", action='store_true')
parser.add_argument("--orientations", action='store_true')
args = parser.parse_args()

print('Loading from', args.dataset)
t0 = time.time()
ds = Dataset.from_file(args.dataset)
elapsed = time.time() - t0
print('Loaded {:d} landmarks in {:.1f} seconds'.format(len(ds.landmarks), elapsed))
ds.visualize(draw_axes=args.axes, draw_orientations=args.orientations)
