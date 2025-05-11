# # This code was written by Simon Alexanderson
# # and is released here: https://github.com/simonalexanderson/PyMO
# import glob
# import os
#
# import sys
# sys.path.append('.')
#
# from data_processing.pymo.parsers import BVHParser
# from pymo.data import Joint, MocapData
#
# from pymo.viz_tools import *
# from pymo.writers import *
# from pymo.preprocessing import *
#
#
# # from argparse import ArgumentParser
#
# import joblib as jl
#
# def feat2bvh(bvh_dir, feat, filename):
#
#     pipeline_dir = './data_processing/utils/'
#     # load data pipeline
#     pipeline = jl.load(pipeline_dir + 'conducting_60fps_data_pipe.sav')
#
#     # feat_files = []
#     # feat_files = sorted([file for file in glob.glob(feat_dir + '/*.npy')])
#     #
#     # for feat_file in feat_files:
#     features = np.load(feat, allow_pickle=False) #['clips']
#     print("Original features shape: ", features.shape)
#
#     # shorten sequence length for visualization
#     features = features[:]
#     # features = features[:3000]
#     print("Shortened features shape: ", features.shape)
#
#     # transform the data back to it's original shape
#     # note: in a real scenario this is usually done with predicted data
#     # note: some transformations (such as transforming to joint positions) are not inversible
#     bvh_data = pipeline.inverse_transform([features])
#
#     # # ensure correct body orientation
#     # bvh_data[0].values["body_world_Xrotation"] = 0
#     # bvh_data[0].values["body_world_Yrotation"] = 0
#     # bvh_data[0].values["body_world_Zrotation"] = 0
#
#     bvh_data[0].values["hips_JNT_Xposition"] = 0
#     bvh_data[0].values["hips_JNT_Yposition"] = 0
#     bvh_data[0].values["hips_JNT_Zposition"] = 0
#
#     bvh_data[0].values["hips_JNT_Xrotation"] = 0
#     bvh_data[0].values["hips_JNT_Yrotation"] = 0
#     bvh_data[0].values["hips_JNT_Zrotation"] = 0
#
#     # Test to write some of it to file for visualization in blender or motion builder
#     writer = BVHWriter()
#
#     name = filename.split('.')[0]
#     name = name.split('predicted_')[-1]
#     bvh_file = os.path.join(bvh_dir, f'pred_{name}.bvh')
#
#     with open(bvh_file,'w') as f:
#         writer.write(bvh_data[0], f)
#
# # if __name__ == '__main__':
# #
# #     # Setup parameter parser
# #     parser = ArgumentParser(add_help=False)
# #     parser.add_argument('--feat_dir', '-feat', default=,
# #                                    help="Path where motion features are stored")
# #     parser.add_argument('--bvh_dir', '-bvh', default=,
# #                                    help="Path where produced motion files (in BVH format) will be stored")
# #     parser.add_argument('--pipeline_dir', '-pipe', default="./data_processing/utils/",
# #                         help="Path where the motion data processing pipeline is be stored")
# #
# #     params = parser.parse_args()
# #
# #
# #     # load data pipeline
# #     pipeline = jl.load(params.pipeline_dir + 'conducting_60fps_data_pipe.sav')
# #
# #     # convert a file
# #     feat2bvh(params.feat_dir, params.bvh_dir)




# This code was written by Simon Alexanderson
# and is released here: https://github.com/simonalexanderson/PyMO
import glob
import os
import sys

module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)

from pymo.parsers import BVHParser
from pymo.data import Joint, MocapData
from pymo.preprocessing import *
from pymo.viz_tools import *
from pymo.writers import *

from argparse import ArgumentParser

import joblib as jl

def feat2bvh(feat_dir, bvh_dir):

    feat_files = []
    feat_files = sorted([file for file in glob.glob(feat_dir + '/*.npy')])

    for feat_file in feat_files:
        features = np.load(feat_file, allow_pickle=True) #['clips']
        print("Original features shape: ", features.shape)

        # shorten sequence length for visualization
        features = features[:]
        # features = features[:3000]
        print("Shortened features shape: ", features.shape)

        # transform the data back to it's original shape
        # note: in a real scenario this is usually done with predicted data
        # note: some transformations (such as transforming to joint positions) are not inversible
        bvh_data = pipeline.inverse_transform([features])

        # # ensure correct body orientation
        # bvh_data[0].values["body_world_Xrotation"] = 0
        # bvh_data[0].values["body_world_Yrotation"] = 0
        # bvh_data[0].values["body_world_Zrotation"] = 0

        bvh_data[0].values["hips_JNT_Xposition"] = 0
        bvh_data[0].values["hips_JNT_Yposition"] = 0
        bvh_data[0].values["hips_JNT_Zposition"] = 0

        bvh_data[0].values["hips_JNT_Xrotation"] = 0
        bvh_data[0].values["hips_JNT_Yrotation"] = 0
        bvh_data[0].values["hips_JNT_Zrotation"] = 0

        # Test to write some of it to file for visualization in blender or motion builder
        writer = BVHWriter()

        name = os.path.basename(feat_file).split('.')[0]
        name = name.split('predicted_')[-1]
        bvh_file = os.path.join(bvh_dir, f'pred_{name}.bvh')

        with open(bvh_file,'w') as f:
            writer.write(bvh_data[0], f)

if __name__ == '__main__':

    # Setup parameter parser
    parser = ArgumentParser(add_help=False)
    parser.add_argument('--feat_dir', '-feat', required=True,
                                   help="Path where motion features are stored")
    parser.add_argument('--bvh_dir', '-bvh', required=True,
                                   help="Path where produced motion files (in BVH format) will be stored")
    parser.add_argument('--pipeline_dir', '-pipe', default="./data_processing/utils/",
    # parser.add_argument('--pipeline_dir', '-pipe', default="./utils/",
                        help="Path where the motion data processing pipeline is be stored")

    params = parser.parse_args()


    # load data pipeline
    # pipeline = jl.load(params.pipeline_dir + 'final_60fps_data_pipe.sav')
    pipeline = jl.load(params.pipeline_dir + 'conducting_30fps_data_pipe.sav')

    # convert a file
    feat2bvh(params.feat_dir, params.bvh_dir)
