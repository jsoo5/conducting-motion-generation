# This code was written by Simon Alexanderson
# and is released here: https://github.com/simonalexanderson/PyMO

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from argparse import ArgumentParser

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

import joblib as jl
import lmdb
import pyarrow
import librosa


# 18 joints (only upper body)
#target_joints = ['b_spine0', 'b_spine1', 'b_spine2', 'b_spine3', 'b_l_shoulder', 'b_l_arm', 'b_l_arm_twist', 'b_l_forearm', 'b_l_wrist_twist', 'b_l_wrist', 'b_r_shoulder', 'b_r_arm', 'b_r_arm_twist', 'b_r_forearm', 'b_r_wrist_twist', 'b_r_wrist', 'b_neck0', 'b_head']

# 50 joints (upper body with fingers)
# target_joints = ['b_spine0', 'b_spine1', 'b_spine2', 'b_spine3',
#                  'b_l_shoulder', 'b_l_arm', 'b_l_arm_twist', 'b_l_forearm', 'b_l_wrist_twist', 'b_l_wrist',
#                  'b_l_pinky1', 'b_l_pinky2', 'b_l_pinky3',
#                  'b_l_ring1', 'b_l_ring2', 'b_l_ring3',
#                  'b_l_middle1', 'b_l_middle2', 'b_l_middle3',
#                  'b_l_index1', 'b_l_index2', 'b_l_index3',
#                  'b_l_thumb0', 'b_l_thumb1', 'b_l_thumb2', 'b_l_thumb3',
#                  'b_r_shoulder', 'b_r_arm', 'b_r_arm_twist', 'b_r_forearm', 'b_r_wrist_twist', 'b_r_wrist',
#                  'b_r_thumb0', 'b_r_thumb1', 'b_r_thumb2', 'b_r_thumb3',
#                  'b_r_pinky1', 'b_r_pinky2', 'b_r_pinky3',
#                  'b_r_middle1', 'b_r_middle2', 'b_r_middle3',
#                  'b_r_ring1', 'b_r_ring2', 'b_r_ring3',
#                  'b_r_index1', 'b_r_index2', 'b_r_index3',
#                  'b_neck0', 'b_head']

# 24 joints (upper and lower body excluding fingers)
# target_joints = ['body_world', 'b_root', 'b_l_upleg', 'b_l_leg', 'b_r_upleg', 'b_r_leg', 'b_spine0', 'b_spine1', 'b_spine2', 'b_spine3', 'b_l_shoulder', 'b_l_arm', 'b_l_arm_twist', 'b_l_forearm', 'b_l_wrist_twist', 'b_l_wrist', 'b_r_shoulder', 'b_r_arm', 'b_r_arm_twist', 'b_r_forearm', 'b_r_wrist_twist', 'b_r_wrist', 'b_neck0', 'b_head']

# 56 joints (upper and lower body including fingers)
# target_joints = ['body_world', 'b_root', 'b_l_upleg', 'b_l_leg', 'b_r_upleg', 'b_r_leg', 'b_spine0', 'b_spine1', 'b_spine2', 'b_spine3', 'b_l_shoulder', 'b_l_arm', 'b_l_arm_twist', 'b_l_forearm', 'b_l_wrist_twist', 'b_l_wrist', 'b_l_pinky1', 'b_l_pinky2', 'b_l_pinky3', 'b_l_ring1', 'b_l_ring2', 'b_l_ring3', 'b_l_middle1', 'b_l_middle2', 'b_l_middle3', 'b_l_index1', 'b_l_index2', 'b_l_index3', 'b_l_thumb0', 'b_l_thumb1', 'b_l_thumb2', 'b_l_thumb3', 'b_r_shoulder', 'b_r_arm', 'b_r_arm_twist', 'b_r_forearm', 'b_r_wrist_twist', 'b_r_wrist', 'b_r_thumb0', 'b_r_thumb1', 'b_r_thumb2', 'b_r_thumb3', 'b_r_pinky1', 'b_r_pinky2', 'b_r_pinky3', 'b_r_middle1', 'b_r_middle2', 'b_r_middle3', 'b_r_ring1', 'b_r_ring2', 'b_r_ring3', 'b_r_index1', 'b_r_index2', 'b_r_index3', 'b_neck0', 'b_head']

# ChoreoMaster_Dataset (49 joints)
# target_joints = ['Hips',                                                    # 1
#                  'LeftUpLeg', 'LeftLeg', 'RightUpLeg', 'RightLeg',          # 4
#                  'Spine', 'Spine1', 'Spine2',                               # 3
#                  'LeftShoulder', 'LeftArm', 'LeftForeArm', 'LeftHand',      # 4
#                  'LeftHandMiddle1', 'LeftHandMiddle2', 'LeftHandMiddle3',   # 3
#                  'LeftHandIndex1', 'LeftHandIndex2', 'LeftHandIndex3',      # 3
#                  'LeftHandRing1', 'LeftHandRing2', 'LeftHandRing3',         # 3
#                  'LeftHandPinky1', 'LeftHandPinky2', 'LeftHandPinky3',      # 3
#                  'LeftHandThumb1', 'LeftHandThumb2', 'LeftHandThumb3',      # 3
#                  'RightShoulder', 'RightArm', 'RightForeArm', 'RightHand',  # 4
#                  'RightHandIndex1', 'RightHandIndex2', 'RightHandIndex3',   # 3
#                  'RightHandMiddle1', 'RightHandMiddle2', 'RightHandMiddle3',# 3
#                  'RightHandRing1', 'RightHandRing2', 'RightHandRing3',      # 3
#                  'RightHandPinky1', 'RightHandPinky2', 'RightHandPinky3',   # 3
#                  'RightHandThumb1', 'RightHandThumb2', 'RightHandThumb3',   # 3
#                  'Neck', 'Neck1', 'Head']                                   # 3

# SMPL
# target_joints = ["pelvis",     "left_hip",    "right_hip",
#                  "spine1",    "left_knee",    "right_knee",
#                  "spine2",    "left_ankle",    "right_ankle",
#                  "spine3",    "left_foot",    "right_foot",
#                  "neck",    "left_collar",    "right_collar",
#                  "head",    "left_shoulder",    "right_shoulder",
#                  "jaw", "left_eye_smplhf", "right_eye_smplhf", "left_elbow",    "right_elbow",
#                  "left_wrist",    "right_wrist",
#                  "left_index1",    "left_index2",    "left_index3",
#                  "left_middle1",    "left_middle2",    "left_middle3",
#                  "left_pinky1",    "left_pinky2",    "left_pinky3",
#                  "left_ring1",    "left_ring2",    "left_ring3",
#                  "left_thumb1",    "left_thumb2",    "left_thumb3",
#                  "right_index1",    "right_index2",    "right_index3",
#                  "right_middle1",    "right_middle2",    "right_middle3",
#                  "right_pinky1",    "right_pinky2",    "right_pinky3",
#                  "right_ring1",    "right_ring2",    "right_ring3",
#                  "right_thumb1",    "right_thumb2",    "right_thumb3"]

# Custom Data (49 joints)
target_joints = ["spine_JNT",
                 "l_upleg_JNT",  "l_leg_JNT",  "l_foot_JNT",
                 "r_upleg_JNT",    "r_leg_JNT",   "r_foot_JNT",
                 "spine1_JNT",   "neck_JNT",    "head_JNT",
                 "spine2_JNT",  "l_shoulder_JNT",    "r_shoulder_JNT",
                 "l_arm_JNT",    "r_arm_JNT",
                 "l_forearm_JNT",    "r_forearm_JNT",
                 "l_hand_JNT",    "r_hand_JNT",
                 "l_handIndex1_JNT",    "l_handIndex2_JNT",    "l_handIndex3_JNT",
                 "l_handMiddle1_JNT",    "l_handMiddle2_JNT",    "l_handMiddle3_JNT",
                 "l_handPinky1_JNT",    "l_handPinky2_JNT",    "l_handPinky3_JNT",
                 "l_handRing1_JNT",    "l_handRing2_JNT",    "l_handRing3_JNT",
                 "l_handThumb1_JNT",    "l_handThumb2_JNT",    "l_handThumb3_JNT",
                 "r_handIndex1_JNT",    "r_handIndex2_JNT",    "r_handIndex3_JNT",
                 "r_handMiddle1_JNT",    "r_handMiddle2_JNT",    "r_handMiddle3_JNT",
                 "r_handPinky1_JNT",    "r_handPinky2_JNT",    "r_handPinky3_JNT",
                 "r_handRing1_JNT",    "r_handRing2_JNT",    "r_handRing3_JNT",
                 "r_handThumb1_JNT",    "r_handThumb2_JNT",    "r_handThumb3_JNT"]



def extract_joint_angles(data_dir, proc_dir, fps):
    p = BVHParser()

    files = sorted([f for f in glob.glob(data_dir + '/*.bvh')])
    data_all = list()

    for f in files:
        print(f)
        p_bvh = p.parse(f)
        data_all.append(p_bvh)

    data_pipe = Pipeline([
        ('dwnsampl', DownSampler(tgt_fps=fps,  keep_all=False)),
        # ('root', RootNormalizer()),
        ('jtsel', JointSelector(target_joints, include_root=False)),
        # ('mir', Mirror(axis='X', append=True)),
        ('exp', MocapParameterizer('expmap')),
        ('np', Numpyfier())
    ])

    out_data = data_pipe.fit_transform(data_all)

    # the datapipe will append the mirrored files to the end
    assert len(out_data) == len(files)

    jl.dump(data_pipe, os.path.join('./utils', f'final_{fps}fps_data_pipe.sav'))

    fi = 0
    for idx in range(0, len(files)):
        filename = os.path.basename(files[idx]).split('60fps')[0]
        ff = os.path.join(proc_dir, f'{filename}{str(idx).zfill(3)}_{fps}fps.npz')
        print(ff)
        np.savez(ff, clips=out_data[fi])
        fi = fi + 1

    # for idx in range(0, len(files)):
    #     filename = os.path.basename(files[idx]).split('30fps')[0]
    #     ff = os.path.join(proc_dir, 'npy', f'feat_{filename}.npy')
    #     print(ff)
    #     np.save(ff, out_data[fi])
    #     fi = fi + 1


# def extract_joint_angles(data_dir, proc_dir):
#     p = BVHParser()
#
#     # data_use = ['train', 'dev']
#
#     files = []
#     files = sorted([f for f in glob.glob(data_dir + '/bvh/*.bvh')])
#
#     file_name = []
#     data_all = list()
#
#     for f in files:
#         name = f[-19:]
#         file_name.append(name)
#         # data_all.append(p.parse(ff))
#         print(f)
#         p_bvh = p.parse(f)
#         data_all.append(p_bvh)
#         print('p_bvh.shape: ', p_bvh.shape)
#
#
#     data_pipe = Pipeline([
#        # ('dwnsampl', DownSampler(tgt_fps=30,  keep_all=False)),
#        ('root', RootNormalizer()),
#        ('jtsel', JointSelector(target_joints, include_root=False)),
#        #('mir', Mirror(axis='X', append=True)),
#        ('exp', MocapParameterizer('expmap')),
#        ('np', Numpyfier())
#     ])
#
#
#     out_data = data_pipe.fit_transform(data_all)
#
#     # the datapipe will append the mirrored files to the end
#     assert len(out_data) == len(files)
#
#     jl.dump(data_pipe, os.path.join('./data_processing/utils', 'data_pipe.sav'))
#
#
#     fi=0
#     for f in file_name:
#         ff = os.path.join(proc_dir, f)
#         print(ff)
#         np.savez(ff[:-4] + ".npz", clips=out_data[fi])
#         # np.savez(f[:-4] + ".npz", clips=out_data[fi])
#         #np.savez(ff[:-4] + "_mirrored.npz", clips=out_data[len(files)+fi])
#         fi=fi+1




if __name__ == '__main__':

    # Setup parameter parser
    parser = ArgumentParser(add_help=False)
    parser.add_argument('--data_dir', '-orig', required=True,
                        help="Path where original motion files (in BVH format) are stored")
    parser.add_argument('--proc_dir', '-proc', required=True,
                        help="Path where extracted motion features will be stored")
    parser.add_argument('--pipeline_dir', '-pipe', default="./utils/",
                        help="Path where the motion data processing pipeline will be stored")
    parser.add_argument('--fps', '-fps', default='60',
                        help='Frame Per Sec')

    params = parser.parse_args()

    # Go over all BVH files
    print("Going to pre-process the following motion files:")
    # files = sorted([f for f in glob.iglob(params.bvh_dir+'/*.bvh')])

    # file_name = []
    # for f in files:
    #     # name = os.path.splitext(f)[0]
    #     name = f[-19:-4]
    #     file_name.append(name)
    # print('*** Print file_name ***\n', file_name)

    extract_joint_angles(params.data_dir, params.proc_dir, 60)




# # This code was written by Simon Alexanderson
# # and is released here: https://github.com/simonalexanderson/PyMO
#
# import numpy as np
# import pandas as pd
# from sklearn.pipeline import Pipeline
#
# from argparse import ArgumentParser
#
# import glob
# import os
# import sys
#
# module_path = os.path.abspath(os.path.join('..'))
# if module_path not in sys.path:
#     sys.path.append(module_path)
#
# from pymo.parsers import BVHParser
# from pymo.data import Joint, MocapData
# from pymo.preprocessing import *
# from pymo.viz_tools import *
# from pymo.writers import *
#
# import joblib as jl
# import glob
#
# # 18 joints (only upper body)
# # target_joints = ['b_spine0', 'b_spine1', 'b_spine2', 'b_spine3', 'b_l_shoulder', 'b_l_arm', 'b_l_arm_twist', 'b_l_forearm', 'b_l_wrist_twist', 'b_l_wrist', 'b_r_shoulder', 'b_r_arm', 'b_r_arm_twist', 'b_r_forearm', 'b_r_wrist_twist', 'b_r_wrist', 'b_neck0', 'b_head']
#
# # 50 joints (upper body with fingers)
# target_joints = ['b_spine0', 'b_spine1', 'b_spine2', 'b_spine3', 'b_l_shoulder', 'b_l_arm', 'b_l_arm_twist',
#                  'b_l_forearm', 'b_l_wrist_twist', 'b_l_wrist', 'b_l_pinky1', 'b_l_pinky2', 'b_l_pinky3', 'b_l_ring1',
#                  'b_l_ring2', 'b_l_ring3', 'b_l_middle1', 'b_l_middle2', 'b_l_middle3', 'b_l_index1', 'b_l_index2',
#                  'b_l_index3', 'b_l_thumb0', 'b_l_thumb1', 'b_l_thumb2', 'b_l_thumb3', 'b_r_shoulder', 'b_r_arm',
#                  'b_r_arm_twist', 'b_r_forearm', 'b_r_wrist_twist', 'b_r_wrist', 'b_r_thumb0', 'b_r_thumb1',
#                  'b_r_thumb2', 'b_r_thumb3', 'b_r_pinky1', 'b_r_pinky2', 'b_r_pinky3', 'b_r_middle1', 'b_r_middle2',
#                  'b_r_middle3', 'b_r_ring1', 'b_r_ring2', 'b_r_ring3', 'b_r_index1', 'b_r_index2', 'b_r_index3',
#                  'b_neck0', 'b_head']
#
#
# # 24 joints (upper and lower body excluding fingers)
# # target_joints = ['body_world', 'b_root', 'b_l_upleg', 'b_l_leg', 'b_r_upleg', 'b_r_leg', 'b_spine0', 'b_spine1', 'b_spine2', 'b_spine3', 'b_l_shoulder', 'b_l_arm', 'b_l_arm_twist', 'b_l_forearm', 'b_l_wrist_twist', 'b_l_wrist', 'b_r_shoulder', 'b_r_arm', 'b_r_arm_twist', 'b_r_forearm', 'b_r_wrist_twist', 'b_r_wrist', 'b_neck0', 'b_head']
#
# # 56 joints (upper and lower body including fingers)
# # target_joints = ['body_world', 'b_root', 'b_l_upleg', 'b_l_leg', 'b_r_upleg', 'b_r_leg', 'b_spine0', 'b_spine1', 'b_spine2', 'b_spine3', 'b_l_shoulder', 'b_l_arm', 'b_l_arm_twist', 'b_l_forearm', 'b_l_wrist_twist', 'b_l_wrist', 'b_l_pinky1', 'b_l_pinky2', 'b_l_pinky3', 'b_l_ring1', 'b_l_ring2', 'b_l_ring3', 'b_l_middle1', 'b_l_middle2', 'b_l_middle3', 'b_l_index1', 'b_l_index2', 'b_l_index3', 'b_l_thumb0', 'b_l_thumb1', 'b_l_thumb2', 'b_l_thumb3', 'b_r_shoulder', 'b_r_arm', 'b_r_arm_twist', 'b_r_forearm', 'b_r_wrist_twist', 'b_r_wrist', 'b_r_thumb0', 'b_r_thumb1', 'b_r_thumb2', 'b_r_thumb3', 'b_r_pinky1', 'b_r_pinky2', 'b_r_pinky3', 'b_r_middle1', 'b_r_middle2', 'b_r_middle3', 'b_r_ring1', 'b_r_ring2', 'b_r_ring3', 'b_r_index1', 'b_r_index2', 'b_r_index3', 'b_neck0', 'b_head']
#
#
# def extract_joint_angles(bvh_dir, files, dest_dir, pipeline_dir, fps):
#     p = BVHParser()
#
#     data_all = list()
#     for f in files:
#         ff = os.path.join(bvh_dir, f)
#         print(ff)
#         data_all.append(p.parse(ff))
#
#     data_pipe = Pipeline([
#         ('dwnsampl', DownSampler(tgt_fps=30, keep_all=False)),
#         ('root', RootNormalizer()),
#         ('jtsel', JointSelector(target_joints, include_root=False)),
#         # ('mir', Mirror(axis='X', append=True)),
#         ('exp', MocapParameterizer('expmap')),
#         ('np', Numpyfier())
#     ])
#
#     out_data = data_pipe.fit_transform(data_all)
#
#     # the datapipe will append the mirrored files to the end
#     assert len(out_data) == len(files)
#
#     jl.dump(data_pipe, os.path.join(pipeline_dir + 'data_pipe.sav'))
#
#     fi = 0
#     for f in files:
#         ff = os.path.join(dest_dir, f)
#         print(ff)
#         np.savez(ff[:-4] + ".npz", clips=out_data[fi])
#         # np.savez(ff[:-4] + "_mirrored.npz", clips=out_data[len(files)+fi])
#         fi = fi + 1
#
#
# if __name__ == '__main__':
#     # Setup parameter parser
#     parser = ArgumentParser(add_help=False)
#     parser.add_argument('--bvh_dir', '-orig', required=True,
#                         help="Path where original motion files (in BVH format) are stored")
#     parser.add_argument('--dest_dir', '-dest', required=True,
#                         help="Path where extracted motion features will be stored")
#     parser.add_argument('--pipeline_dir', '-pipe', default="./utils/",
#                         help="Path where the motion data processing pipeline will be stored")
#
#     params = parser.parse_args()
#
#     files = []
#     # Go over all BVH files
#     print("Going to pre-process the following motion files:")
#     files = sorted([f for f in glob.iglob(params.bvh_dir + '/*.bvh')])
#
#     extract_joint_angles(params.bvh_dir, files, params.dest_dir, params.pipeline_dir, fps=30)