import torch
import os
import numpy as np
from tqdm import tqdm
import pickle
from utils.data.utils import lengths_to_mask
from sklearn.cluster import KMeans
from sklearnex import patch_sklearn
patch_sklearn(['KMeans','DBSCAN'])


def Kmeans(n_clusters, text_embeds, motion_embeds):
    seed = 42
    k_means = KMeans(n_clusters=n_clusters, random_state=seed, n_init="auto").fit(text_embeds[:, 0])
    text_global = k_means.cluster_centers_

    cluster_list = []
    for cluster_label in np.unique(k_means.labels_):
        cluster_indices = np.where(k_means.labels_ == cluster_label)[0]
        cluster_list.append(np.mean(motion_embeds[cluster_indices], axis=0, keepdims=True))

    motion_global = np.concatenate(cluster_list, axis=0)

    return text_global, motion_global

@torch.no_grad()
def cluster_reset(cfg, device, motion_encoder, text_encoder, denoiser, denoiser_ema, text_name):
    motion_embeds = []
    text_embeds = []
    with open(os.path.join(cfg.data.root, 'train.txt'), 'r') as f:
        for idx, name in enumerate(tqdm(f.readlines())):
            name = name.replace('\n', '')
            motion_dir = os.path.join(cfg.data.root, 'new_joint_vecs', name + '.npy')
            text_dir = os.path.join(os.path.join(cfg.data.root, 'texts', name + '.txt'))

            if not os.path.exists(motion_dir):
                continue
            motion = np.load(motion_dir)

            if (len(motion)) < cfg.data.min_motion_length or (len(motion) >= 200):
                continue

            with open(text_dir, 'r') as ft:
                texts = ft.readlines()

                texts_list = []
                for text in texts:
                    text = text.split('#')[0]
                    texts_list.append(text)

            text_embed = torch.mean(text_encoder(texts_list), dim=0, keepdim=True)
            length = motion.shape[0]
            mask = lengths_to_mask([length], max_len=cfg.data.max_motion_length, device=device)
            if length >= cfg.data.max_motion_length:
                motion = motion[:cfg.data.max_motion_length]
            else:
                motion = np.concatenate((motion, np.zeros((cfg.data.max_motion_length - length, motion.shape[1]))), axis=0)

            motion_embed, _ = motion_encoder.motion_encode(torch.from_numpy(motion).to(device)[None, :, :].float(), ~mask)
            
            motion_embeds.append(motion_embed.detach().cpu().numpy())
            text_embeds.append(text_embed.detach().cpu().numpy())

            # if idx > 800:
            #     break
    
    motion_embeds = np.concatenate(motion_embeds, axis=0)
    text_embeds = np.concatenate(text_embeds, axis=0)
    
    n_cluster = cfg.diffusion.n_cluster
    text_global, motion_global = Kmeans(int(n_cluster), text_embeds, motion_embeds)
    cluster_dict = {"text": text_global, "motion": motion_global}
    
    with open(os.path.join(cfg.motion_ae.pretrain_dir, f'clusters_{text_name}_{cfg.diffusion.n_cluster}.plk'), 'wb') as f:
        pickle.dump(cluster_dict, f)

    denoiser_ema.load_cluster_dict(cluster_dict)
    denoiser.load_cluster_dict(cluster_dict)

@torch.no_grad()
def cluster_load(cfg, text_name, denoiser, denoiser_ema):
    with open(os.path.join(cfg.motion_ae.pretrain_dir, f'clusters_{text_name}_{cfg.diffusion.n_cluster}.plk'), 'rb') as f:
        print(f'load clusters_{text_name}_{cfg.diffusion.n_cluster}.plk')
        cluster_dict = pickle.load(f)

    if denoiser_ema is not None:
        denoiser_ema.load_cluster_dict(cluster_dict)
    denoiser.load_cluster_dict(cluster_dict)