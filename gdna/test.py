import warnings

warnings.filterwarnings("ignore")

import pytorch_lightning as pl
import hydra
import torch
import os
import sys
import numpy as np
from lib.gdna_model import BaseModel
from tqdm import tqdm
import imageio
from lib.utils.render import render_mesh_dict, Renderer
import glob
from lib.dataset.datamodule import DataProcessor
from lib.model.helpers import Dict2Class
import pandas
from numpy import savez_compressed
from pathlib import Path
import omegaconf

FILE = Path(__file__).resolve()
ROOTD = FILE.parents[0]  # YOLOv5 root directory
if str(ROOTD) not in sys.path:
    sys.path.append(str(ROOTD))  # add ROOT to PATH
ROOTD = Path(os.path.relpath(ROOTD, Path.cwd()))  # relative

DATAMODULE, CFG_MODEL, EXPNAME, ROOT = None, None, None, None


@hydra.main(config_path="gdna/config", config_name="config")
def get_cfg(opt):
    global DATAMODULE, CFG_MODEL, EXPNAME, ROOT
    DATAMODULE = opt.datamodule
    CFG_MODEL = opt.model
    EXPNAME = opt.expname
    ROOT = opt.r_path
    print(ROOT)


class GDNA:
    def __init__(self, seed: int = 42, expname=None):
        if expname == 'thuman':
            self.expname = expname
            sys.argv[1:] = ['expname=thuman', 'model.norm_network.multires=6', '+experiments=fine', 'datamodule=thuman',
                            f'+r_path={ROOTD}/..']
            # self.datamodule = omegaconf.OmegaConf.load(f'{ROOT}/gdna/config/datamodule/thuman.yaml').datamodule
            # self.cfg_model.norm_network.multires = 6
        else:
            self.expname = 'renderpeople'
        # print(self.datamodule, self.expname)
        get_cfg()
        self.seed, self.datamodule, self.cfg_model, self.expname = \
            seed, DATAMODULE, CFG_MODEL, EXPNAME

        self.model, self.eval_mode = None, None
        self.output_folder = f'{ROOT}/gdna/outputs/{self.expname}/results'
        if not os.path.exists(self.output_folder):
            os.makedirs(self.output_folder)

        try:
            if not torch.cuda.is_available():
                raise
        except (Exception,):
            print('GPU is not detected!')

    def pre_load(self):
        pl.seed_everything(self.seed, workers=True)
        torch.set_num_threads(10)

        scan_info = pandas.read_csv(hydra.utils.to_absolute_path(self.datamodule.data_list))
        meta_info = Dict2Class({'n_samples': len(scan_info)})
        # print(opt.datamodule.data_list)

        data_processor = DataProcessor(self.datamodule)
        checkpoint_path = os.path.join(f'{ROOT}/gdna/outputs/{self.expname}/checkpoints', 'last.ckpt')

        self.model = BaseModel.load_from_checkpoint(
            checkpoint_path=checkpoint_path,
            strict=False,
            opt=self.cfg_model,
            meta_info=meta_info,
            data_processor=data_processor,
        ).cuda()

        renderer = Renderer(256, anti_alias=True)

        max_samples = 15

        smpl_param_zero = torch.zeros((1, 86)).cuda().float()
        smpl_param_zero[:, 0] = 1

        motion_folder = hydra.utils.to_absolute_path(f'{ROOT}/gdna/data/aist_demo/seqs')
        motion_files = sorted(glob.glob(os.path.join(motion_folder, '*.npz')))
        smpl_param_anim = []
        for f in motion_files:
            f = np.load(f)
            smpl_params = np.zeros(86)
            smpl_params[0], smpl_params[4:76] = 1, f['pose']
            smpl_param_anim.append(torch.tensor(smpl_params))
        smpl_param_anim = torch.stack(smpl_param_anim).float().cuda()

        return meta_info, max_samples, smpl_param_zero, smpl_param_anim, renderer, data_processor

    def get_mesh(self, batch_list, renderer, data_processor):
        images_all = []
        with torch.no_grad():
            import time

            for i, batch in enumerate(tqdm(batch_list)):
                cond = self.model.prepare_cond(batch)
                batch_smpl = data_processor.process_smpl({'smpl_params': batch['smpl_params']}, self.model.smpl_server)

                # print(batch_smpl['smpl_jnts'])

                mesh_cano = self.model.extract_mesh(batch_smpl['smpl_verts_cano'], batch_smpl['smpl_tfs'], cond,
                                                    res_up=4)
                mesh_def = self.model.deform_mesh(mesh_cano, batch_smpl['smpl_tfs'])

                verts = mesh_def['verts'].cpu().numpy()
                faces = mesh_def['faces'].cpu().numpy()
                # print("VERTS: ", verts[:3].tolist())
                # print("FACES: ", faces[:3].tolist())

                npz_folder = 'tmp'
                if not os.path.exists(hydra.utils.to_absolute_path(f'{ROOT}/{npz_folder}')):
                    os.makedirs((hydra.utils.to_absolute_path(f'{ROOT}/{npz_folder}')))

                savez_compressed(hydra.utils.to_absolute_path(f'{ROOT}/{npz_folder}/verts{i}.npz'), verts)
                savez_compressed(hydra.utils.to_absolute_path(f'{ROOT}/{npz_folder}/faces{i}.npz'), faces)

                img_def = render_mesh_dict(mesh_def, mode='xy', render_new=renderer)
                # img_def = np.concatenate([img_def[:256,:,:], img_def[256:,:,:]],axis=1)
                img_def = np.concatenate([img_def[256:, :, :]], axis=1)

                imageio.mimsave(
                    os.path.join(self.output_folder,
                                 '%s_seed%d_%d.png' % (self.eval_mode, self.seed, int(time.time()))),
                    [img_def], codec='libx264')

                '''images_all.append(img_def)
                if i % 10 == 0:
                    imageio.mimsave(os.path.join(output_folder, '%s_seed%d.mp4' % (opt.eval_mode, opt.seed)), images_all,
                                    codec='libx264')'''

    def action_z_shape(self, z_shape=None, z_detail=None):
        self.eval_mode = 'z_shape'
        meta_info, max_samples, smpl_param_zero, smpl_param_anim, renderer, data_processor = self.pre_load()

        batch_list = []

        idx_b = np.random.randint(0, meta_info.n_samples)

        while len(batch_list) < max_samples:
            idx_a = idx_b
            idx_b = np.random.randint(0, meta_info.n_samples)

            z_shape_a = self.model.z_shapes.weight.data[idx_a]
            z_shape_b = self.model.z_shapes.weight.data[idx_b]
            z_detail = self.model.z_details.weight.data.mean(0)

            for i in range(10):
                z_shape = torch.lerp(z_shape_a, z_shape_b, i / 10)

                batch = {'z_shape': z_shape[None],
                         'z_detail': z_detail[None],
                         'smpl_params': smpl_param_zero}

                batch_list.append(batch)

        self.get_mesh(batch_list, renderer, data_processor)

    def action_z_detail(self, z_shape=None, z_detail=None):
        self.eval_mode = 'z_detail'
        meta_info, max_samples, smpl_param_zero, smpl_param_anim, renderer, data_processor = self.pre_load()

        batch_list = []

        idx_b = np.random.randint(0, meta_info.n_samples)

        while len(batch_list) < max_samples:
            idx_a = idx_b
            idx_b = np.random.randint(0, meta_info.n_samples)

            z_detail_a = self.model.z_details.weight.data[idx_a]
            z_detail_b = self.model.z_details.weight.data[idx_b]
            z_shape = self.model.z_shapes.weight.data.mean(0)

            for i in range(10):
                z_detail = torch.lerp(z_detail_a, z_detail_b, i / 10)

                batch = {'z_shape': z_shape[None],
                         'z_detail': z_detail[None],
                         'smpl_params': smpl_param_zero}

                batch_list.append(batch)

        self.get_mesh(batch_list, renderer, data_processor)

    def action_betas(self, z_shape=None, z_detail=None):
        self.eval_mode = 'betas'
        _, _, smpl_param_zero, _, renderer, data_processor = self.pre_load()

        batch_list = []

        z_shape = self.model.z_shapes.weight.data.mean(0)
        z_detail = self.model.z_details.weight.data.mean(0)

        betas = torch.cat([torch.linspace(0, -2, 10),
                           torch.linspace(-2, 0, 10),
                           torch.linspace(0, 2, 10),
                           torch.linspace(2, 0, 10)])

        for i in range(len(betas)):
            smpl_param = smpl_param_zero.clone()
            smpl_param[:, -10] = betas[i]

            batch = {'z_shape': z_shape[None],
                     'z_detail': z_detail[None],
                     'smpl_params': smpl_param}

            batch_list.append(batch)

        self.get_mesh(batch_list, renderer, data_processor)

    def action_thetas(self, z_shape=None, z_detail=None):
        self.eval_mode = 'thetas'
        _, _, _, smpl_param_anim, renderer, data_processor = self.pre_load()

        batch_list = []

        z_shape = self.model.z_shapes.weight.data.mean(0)
        z_detail = self.model.z_details.weight.data.mean(0)

        for i in range(len(smpl_param_anim)):
            batch = {'z_shape': z_shape[None],
                     'z_detail': z_detail[None],
                     'smpl_params': smpl_param_anim[[i]]
                     }

            batch_list.append(batch)

        self.get_mesh(batch_list, renderer, data_processor)

    def action_sample(self):
        self.eval_mode = 'sample'
        meta_info, max_samples, smpl_param_zero, smpl_param_anim, renderer, data_processor = self.pre_load()

        batch_list = []

        z_shapes, z_details = self.model.sample_codes(max_samples)

        for i in range(len(z_shapes)):
            id_smpl = np.random.randint(len(smpl_param_anim))
            batch = {'z_shape': z_shapes[i][None],
                     'z_detail': z_details[i][None],
                     'smpl_params': smpl_param_anim[id_smpl][None],
                     }
            # print(batch)
            batch_list.append(batch)

        self.get_mesh(batch_list, renderer, data_processor)
