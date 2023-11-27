import os, torch, json, argparse
from easydict import EasyDict as edict
import yaml
from models.pipeline_sim_c2f import Pipeline
from lib.utils import setup_seed
from configs.models import architectures
from models.model import Model
import torch
from pointscope import PointScopeClient as PSC


setup_seed(0)

def join(loader, node):
    seq = loader.construct_sequence(node)
    return '_'.join([str(i) for i in seq])

yaml.add_constructor('!join', join)


if __name__ == '__main__':
    # load configs
    config_file = "configs/test/3dmatch.yaml"
    with open(config_file,'r') as f:
        config = yaml.load(f, Loader=yaml.Loader)

    config = edict(config)
    
    # model initialization
    config.kpfcn_config.architecture = architectures[config.dataset]
    config.model = Pipeline(config)
    
    model = Model.load_from_checkpoint(
        "weights.ckpt", config=config
    )
    model.eval()
    src_pcd, tgt_pcd, est_tsfm = model.demo()

    PSC().vedo(subplot=2).add_pcd(src_pcd).add_pcd(tgt_pcd).draw_at(1).add_pcd(src_pcd, est_tsfm).add_pcd(tgt_pcd).show()