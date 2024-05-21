from .humanml.utils.word_vectorizer import WordVectorizer
from .utils import mld_collate, a2m_collate
from .datamodule.humanml_data_module import HumanML3DDataModule
from .datamodule.kit_data_module import KITDataModule
from .datamodule.humanact12_data_module import Humanact12DataModule
import numpy as np
import os

dataset_module_map = {
    "humanml3d": HumanML3DDataModule,
    "kit": KITDataModule,
    "humanact12": Humanact12DataModule,
    # "uestc": UestcDataModule,
}

motion_subdir = {"humanml3d": "new_joint_vecs", "kit": "new_joint_vecs"}

def get_WordVectorizer(cfg, phase, dataset_name):
    if phase not in ["text_only"]:
        if dataset_name.lower() in ["humanml3d", "kit"]:
            return WordVectorizer('deps/glove', "our_vab")
        else:
            raise ValueError("Only support WordVectorizer for HumanML3D")
    else:
        return None

def get_collate_fn(name, phase="train"):
    if name.lower() in ["humanml3d", "kit"]:
        return mld_collate
    elif name.lower() in ["humanact12", 'uestc']:
        return a2m_collate

def get_mean_std(phase, cfg, dataset_name):
    # todo: use different mean and val for phases
    name = "t2m" if dataset_name == "humanml3d" else dataset_name
    assert name in ["t2m", "kit"]
    if phase in ["val"]:
        if name == 't2m':
            data_root = os.path.join('deps', name, "Comp_v6_KLD01", "meta")
        elif name == 'kit':
            data_root = os.path.join('deps', name, "Comp_v6_KLD005", "meta")
        else:
            raise ValueError("Only support t2m and kit")
        mean = np.load(os.path.join(data_root, "mean.npy"))
        std = np.load(os.path.join(data_root, "std.npy"))
    else:
        data_root = cfg.data.root
        mean = np.load(os.path.join(data_root, "Mean.npy"))
        std = np.load(os.path.join(data_root, "Std.npy"))

    return mean, std

def get_datasets(cfg, phase='train'):
    # get dataset names form cfg
    dataset_name = cfg.data.name

    if dataset_name.lower() in ["humanml3d", "kit"]:
        data_root = cfg.data.root

        # get mean and std corresponding to dataset
        mean, std = get_mean_std(phase, cfg, dataset_name)
        mean_eval, std_eval = get_mean_std("val", cfg, dataset_name)

        # get WordVectorizer
        wordVectorizer = get_WordVectorizer(cfg, phase, dataset_name)

        # get collect_fn
        collate_fn = get_collate_fn(dataset_name, phase)

        # get dataset module
        dataset = dataset_module_map[dataset_name.lower()](
            cfg=cfg,
            batch_size=cfg.train.batchsize,
            num_workers=cfg.train.num_workers,
            collate_fn=collate_fn,
            mean=mean,
            std=std,
            mean_eval=mean_eval,
            std_eval=std_eval,
            w_vectorizer=wordVectorizer,
            text_dir=os.path.join(data_root, "texts"),
            motion_dir=os.path.join(data_root, motion_subdir[dataset_name]),
            max_motion_length=cfg.data.max_motion_length,
            min_motion_length=cfg.data.min_motion_length,
            max_text_len=cfg.data.max_text_len,
            unit_length=cfg.data.unit_length,
        )

    elif dataset_name.lower() in ["humanact12", 'uestc']:
        # get collect_fn
        collate_fn = get_collate_fn(dataset_name, phase)

        # get dataset module
        dataset = dataset_module_map[dataset_name.lower()](
            datapath=cfg.data.root,
            cfg=cfg,
            batch_size=cfg.train.batchsize,
            num_workers=cfg.train.num_workers,
            collate_fn=collate_fn,
            num_frames=cfg.data.num_frames,
            sampling=cfg.data.sampler.sampling,
            sampling_step=cfg.data.sampler.sampling_step,
            pose_rep=cfg.data.pose_rep,
            max_len=cfg.data.sampler.max_len,
            min_len=cfg.data.sampler.min_len,
            num_seq_max=cfg.data.sampler.max_seq,
            glob=cfg.data.glob,
            translation=cfg.data.translation)


    return dataset