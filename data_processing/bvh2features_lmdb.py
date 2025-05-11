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
# target_joints = ['b_spine0', 'b_spine1', 'b_spine2', 'b_spine3', 'b_l_shoulder', 'b_l_arm', 'b_l_arm_twist', 'b_l_forearm', 'b_l_wrist_twist', 'b_l_wrist', 'b_l_pinky1', 'b_l_pinky2', 'b_l_pinky3', 'b_l_ring1', 'b_l_ring2', 'b_l_ring3', 'b_l_middle1', 'b_l_middle2', 'b_l_middle3', 'b_l_index1', 'b_l_index2', 'b_l_index3', 'b_l_thumb0', 'b_l_thumb1', 'b_l_thumb2', 'b_l_thumb3', 'b_r_shoulder', 'b_r_arm', 'b_r_arm_twist', 'b_r_forearm', 'b_r_wrist_twist', 'b_r_wrist', 'b_r_thumb0', 'b_r_thumb1', 'b_r_thumb2', 'b_r_thumb3', 'b_r_pinky1', 'b_r_pinky2', 'b_r_pinky3', 'b_r_middle1', 'b_r_middle2', 'b_r_middle3', 'b_r_ring1', 'b_r_ring2', 'b_r_ring3', 'b_r_index1', 'b_r_index2', 'b_r_index3', 'b_neck0', 'b_head']

# 24 joints (upper and lower body excluding fingers)
# target_joints = ['body_world', 'b_root', 'b_l_upleg', 'b_l_leg', 'b_r_upleg', 'b_r_leg', 'b_spine0', 'b_spine1', 'b_spine2', 'b_spine3', 'b_l_shoulder', 'b_l_arm', 'b_l_arm_twist', 'b_l_forearm', 'b_l_wrist_twist', 'b_l_wrist', 'b_r_shoulder', 'b_r_arm', 'b_r_arm_twist', 'b_r_forearm', 'b_r_wrist_twist', 'b_r_wrist', 'b_neck0', 'b_head']

#56 joints (upper and lower body including fingers)
target_joints = ['body_world', 'b_root', 'b_l_upleg', 'b_l_leg', 'b_r_upleg', 'b_r_leg', 'b_spine0', 'b_spine1', 'b_spine2', 'b_spine3', 'b_l_shoulder', 'b_l_arm', 'b_l_arm_twist', 'b_l_forearm', 'b_l_wrist_twist', 'b_l_wrist', 'b_l_pinky1', 'b_l_pinky2', 'b_l_pinky3', 'b_l_ring1', 'b_l_ring2', 'b_l_ring3', 'b_l_middle1', 'b_l_middle2', 'b_l_middle3', 'b_l_index1', 'b_l_index2', 'b_l_index3', 'b_l_thumb0', 'b_l_thumb1', 'b_l_thumb2', 'b_l_thumb3', 'b_r_shoulder', 'b_r_arm', 'b_r_arm_twist', 'b_r_forearm', 'b_r_wrist_twist', 'b_r_wrist', 'b_r_thumb0', 'b_r_thumb1', 'b_r_thumb2', 'b_r_thumb3', 'b_r_pinky1', 'b_r_pinky2', 'b_r_pinky3', 'b_r_middle1', 'b_r_middle2', 'b_r_middle3', 'b_r_ring1', 'b_r_ring2', 'b_r_ring3', 'b_r_index1', 'b_r_index2', 'b_r_index3', 'b_neck0', 'b_head']


def make_lmdb_gesture_dataset(data_dir, proc_dir):
    data_use = ['train', 'val']

    out_path = os.path.join(proc_dir, 'lmdb')
    if not os.path.exists(out_path):
        os.makedirs(out_path)

    map_size = 1024 * 20  # in MB
    map_size <<= 20  # in B
    db = [lmdb.open(os.path.join(out_path, 'lmdb_train'), map_size=map_size),
          lmdb.open(os.path.join(out_path, 'lmdb_test'), map_size=map_size)]

    # delete existing files
    for i in range(2):
        with db[i].begin(write=True) as txn:
            txn.drop(db[i].open_db())

    all_poses = []
    save_idx = 0

    for i in range(2):
        bvh_files = sorted(glob.glob(data_dir + str(data_use[i]) + '/bvh/*.bvh'))
        wav_files = sorted(glob.glob(data_dir + str(data_use[i]) + '/wav/*.wav'))
        # print(bvh_files)

        for bvh_file in bvh_files:
            name = os.path.split(bvh_file)[1][:-4]
            print(name)

        for wav_file in wav_files:
            # load audio
            audio_raw, audio_sr = librosa.load(wav_file, mono=True, sr=16000, res_type='kaiser_fast')

            # load skeletons
            dump_pipeline = (save_idx == 2)  # trn_2022_v1_002 has a good rest finger pose
            # poses, poses_mirror = process_bvh(bvh_file)
            poses = extract_joint_angles(bvh_files)

            # process
            clips = [{'vid': name, 'clips': []},  # train
                     {'vid': name, 'clips': []}]  # validation

            # split
            if save_idx % 100 == 0:
                dataset_idx = 1  # validation
            else:
                dataset_idx = 0  # train

            # save subtitles and skeletons
            poses = np.asarray(poses, dtype=np.float16)
            clips[dataset_idx]['clips'].append(
                {'poses': poses,
                 'audio_raw': audio_raw
                 })

            print(f'poses: {poses}')

            # write to db
            for i in range(2):
                with db[i].begin(write=True) as txn:
                    if len(clips[i]['clips']) > 0:
                        k = '{:010}'.format(save_idx).encode('ascii')
                        v = pyarrow.serialize(clips[i]).to_buffer()
                        txn.put(k, v)

            all_poses.append(poses)
            save_idx += 1

    # close db
    for i in range(2):
        db[i].sync()
        db[i].close()


    # calculate data mean
    print(f'all_poses.shape: {all_poses.shape}')
    all_poses = np.vstack(all_poses)
    pose_mean = np.mean(all_poses, axis=0, dtype=np.float64)
    pose_std = np.std(all_poses, axis=0, dtype=np.float64)

    print('data mean/std')
    print('data_mean:', str(["{:0.5f}".format(e) for e in pose_mean]).replace("'", ""))
    print('data_std:', str(["{:0.5f}".format(e) for e in pose_std]).replace("'", ""))



def extract_joint_angles(files):
    p = BVHParser()

    # files = []
    # files = sorted([f for f in glob.glob(data_dir + f'/{data_purpose}/*/*.bvh')])
    #
    file_name = []
    data_all = list()

    for f in files:
        name = f[-19:]
        file_name.append(name)
        # data_all.append(p.parse(ff))
        print(f)
        data_all.append(p.parse(f))


    data_pipe = Pipeline([
       # ('dwnsampl', DownSampler(tgt_fps=30,  keep_all=False)),
       ('root', RootNormalizer()),
       ('jtsel', JointSelector(target_joints, include_root=False)),
       #('mir', Mirror(axis='X', append=True)),
       ('exp', MocapParameterizer('expmap')),
       ('np', Numpyfier())
    ])


    out_data = data_pipe.fit_transform(data_all)
    
    # the datapipe will append the mirrored files to the end
    assert len(out_data) == len(files)
    
    jl.dump(data_pipe, os.path.join('./data_processing/utils', 'new_data_pipe.sav'))

    # Code from 'twh_dataset_to_lmdb.py'
    # euler -> rotation matrix
    out_data = out_data.reshape((out_data.shape[0], out_data.shape[1], -1, 6))  # 3 pos (XYZ), 3 rot (ZXY)
    out_matrix = np.zeros((out_data.shape[0], out_data.shape[1], out_data.shape[2], 12))  # 3 pos, 1 rot matrix (9 elements)
    for i in range(out_data.shape[0]):  # mirror
        for j in range(out_data.shape[1]):  # frames
            for k in range(out_data.shape[2]):  # joints
                out_matrix[i, j, k, :3] = out_data[i, j, k, :3]  # positions
                r = R.from_euler('ZXY', out_data[i, j, k, 3:], degrees=True)
                out_matrix[i, j, k, 3:] = r.as_matrix().flatten()  # rotations
    out_matrix = out_matrix.reshape((out_data.shape[0], out_data.shape[1], -1))

    print(f'out_matrix: {out_matrix[0]}')

    # fi=0
    # for f in file_name:
    #     ff = os.path.join(dest_dir, f)
    #     print(ff)
    #     np.savez(ff[:-4] + ".npz", clips=out_data[fi])
    #     # np.savez(f[:-4] + ".npz", clips=out_data[fi])
    #     #np.savez(ff[:-4] + "_mirrored.npz", clips=out_data[len(files)+fi])
    #     fi=fi+1

    return out_matrix[0]




if __name__ == '__main__':

    # Setup parameter parser
    parser = ArgumentParser(add_help=False)
    parser.add_argument('--data_dir', '-orig', required=True,
                                   help="Path where original motion files (in BVH format) are stored")
    parser.add_argument('--proc_dir', '-proc', required=True,
                                   help="Path where extracted motion features will be stored")
    # parser.add_argument('--data_purpose', '-use', default="trn",
    #                     help="What is the purpose of data trn/val/tst")

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

    make_lmdb_gesture_dataset(params.data_dir, params.proc_dir)




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