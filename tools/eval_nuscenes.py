import os, numpy as np, argparse, json, sys, numba, yaml, multiprocessing, shutil
from os import path as osp
import mmcv

from nuscenes.eval.tracking.evaluate import TrackingEval
from nuscenes.eval.common.config import config_factory
from nuscenes.eval.tracking.data_classes import TrackingMetrics, TrackingMetricDataList, TrackingConfig, TrackingBox, TrackingMetricData

# Settings.
parser = argparse.ArgumentParser(description='Evaluate nuScenes tracking results.',
                                    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('result_path', type=str, help='The submission as a JSON file.')
parser.add_argument('--output_dir', type=str, default='~/nuscenes-metrics',
                    help='Folder to store result metrics, graphs and example visualizations.')
parser.add_argument('--eval_set', type=str, default='val',
                    help='Which dataset split to evaluate on, train, val or test.')
parser.add_argument('--dataroot', type=str, default='/data/sets/nuscenes',
                    help='Default nuScenes data directory.')
parser.add_argument('--version', type=str, default='v1.0-trainval',
                    help='Which version of the nuScenes dataset to evaluate on, e.g. v1.0-trainval.')
parser.add_argument('--config_path', type=str, default='',
                    help='Path to the configuration file.'
                            'If no path given, the NIPS 2019 configuration will be used.')
parser.add_argument('--render_curves', type=int, default=1,
                    help='Whether to render statistic curves to disk.')
parser.add_argument('--verbose', type=int, default=1,
                    help='Whether to print to stdout.')
parser.add_argument('--render_classes', type=str, default='', nargs='+',
                    help='For which classes we render tracking results to disk.')
args = parser.parse_args()

result_path_ = os.path.expanduser(args.result_path)
output_dir_ = os.path.expanduser(args.output_dir)
eval_set_ = args.eval_set
dataroot_ = args.dataroot
version_ = args.version
config_path = args.config_path
render_curves_ = bool(args.render_curves)
verbose_ = bool(args.verbose)
render_classes_ = args.render_classes

if config_path == '':
    cfg_ = config_factory('tracking_nips_2019')
else:
    with open(config_path, 'r') as _f:
        cfg_ = TrackingConfig.deserialize(json.load(_f))

nusc_eval = TrackingEval(config=cfg_, result_path=result_path_, eval_set=eval_set_, output_dir=output_dir_,
                            nusc_version=version_, nusc_dataroot=dataroot_, verbose=verbose_,
                            render_classes=render_classes_)
metrics = nusc_eval.main(render_curves=render_curves_)

metrics = mmcv.load(osp.join(output_dir_, 'metrics_summary.json'))
print(metrics)