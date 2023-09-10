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

# import utils.glob_vars as gl

# Start global var
FILE = Path(__file__).resolve()
ROOTD = FILE.parents[1]
if str(ROOTD) not in sys.path:
    sys.path.append(str(ROOTD))  # add ROOT to PATH
ROOTD = Path(os.path.relpath(ROOTD, Path.cwd()))  # relative

DATAMODULE, CFG_MODEL, EXPNAME, ROOT = None, None, None, None

BATCH_GEN = []
ACT_GEN = []
MESH_GEN = []


# End global var


@hydra.main(config_path="gdna/config", config_name="config")
def get_cfg(opt):
    global DATAMODULE, CFG_MODEL, EXPNAME, ROOT
    DATAMODULE = opt.datamodule
    CFG_MODEL = opt.model
    EXPNAME = opt.expname
    ROOT = opt.r_path


class GDNA:
    def __init__(self, max_samples: int = 1, seed: int = 42, expname=None):
        self.model, self.eval_mode, self.renderer, self.data_processor, self.meta_info, \
            self.smpl_param_zero, self.smpl_param_anim = None, None, None, None, None, None, None
        if expname == 'thuman':
            self.expname = expname
            sys.argv[1:] = ['expname=thuman', 'model.norm_network.multires=6', '+experiments=fine', 'datamodule=thuman',
                            f'+r_path={ROOTD}']
            # self.datamodule = omegaconf.OmegaConf.load(f'{ROOT}/gdna/config/datamodule/thuman.yaml').datamodule
            # self.cfg_model.norm_network.multires = 6
        else:
            self.expname = expname
            sys.argv[1:] = ['expname=renderpeople', '+experiments=fine', f'+r_path={ROOTD}']
        # print(self.datamodule, self.expname)
        get_cfg()
        self.max_samples, self.seed, self.datamodule, self.cfg_model, self.expname = \
            max_samples, seed, DATAMODULE, CFG_MODEL, EXPNAME

        self.output_folder = f'{ROOT}/gdna/outputs/{self.expname}/results'
        if not os.path.exists(self.output_folder):
            os.makedirs(self.output_folder)

        try:
            if not torch.cuda.is_available():
                raise
        except (Exception,):
            print('GPU is not detected!')

        self.pre_load()

    def pre_load(self):
        pl.seed_everything(self.seed, workers=True)
        torch.set_num_threads(10)

        scan_info = pandas.read_csv(hydra.utils.to_absolute_path(self.datamodule.data_list))
        self.meta_info = Dict2Class({'n_samples': len(scan_info)})
        # print(opt.datamodule.data_list)

        self.data_processor = DataProcessor(self.datamodule)
        checkpoint_path = os.path.join(f'{ROOT}/gdna/outputs/{self.expname}/checkpoints', 'last.ckpt')

        self.model = BaseModel.load_from_checkpoint(
            checkpoint_path=checkpoint_path,
            strict=False,
            opt=self.cfg_model,
            meta_info=self.meta_info,
            data_processor=self.data_processor,
        ).cuda()

        self.renderer = Renderer(256, anti_alias=True)

        self.smpl_param_zero = torch.zeros((1, 86)).cuda().float()
        self.smpl_param_zero[:, 0] = 1

        motion_folder = hydra.utils.to_absolute_path(f'{ROOT}/gdna/data/aist_demo/seqs')
        motion_files = sorted(glob.glob(os.path.join(motion_folder, '*.npz')))
        self.smpl_param_anim = []
        for f in motion_files:
            f = np.load(f)
            smpl_params = np.zeros(86)
            smpl_params[0], smpl_params[4:76] = 1, f['pose']
            self.smpl_param_anim.append(torch.tensor(smpl_params))
        self.smpl_param_anim = torch.stack(self.smpl_param_anim).float().cuda()

    def get_mesh(self, batch_list):
        bones = []
        MESH_GEN.clear()
        with torch.no_grad():
            import time

            for i, batch in enumerate(tqdm(batch_list, desc=self.eval_mode)):
                # print(list(batch.keys())[0])
                if self.eval_mode == 'sample':
                    batch = batch[list(batch.keys())[0]]

                cond = self.model.prepare_cond(batch)
                batch_smpl = self.data_processor.process_smpl({'smpl_params': batch['smpl_params']},
                                                              self.model.smpl_server)

                mesh_cano = self.model.extract_mesh(batch_smpl['smpl_verts_cano'], batch_smpl['smpl_tfs'], cond,
                                                    res_up=4)
                mesh_def = self.model.deform_mesh(mesh_cano, batch_smpl['smpl_tfs'])
                # print(mesh_def)
                verts = mesh_def['verts'].cpu().numpy()
                faces = mesh_def['faces'].cpu().numpy()

                d = {'verts': verts, 'faces': faces}
                MESH_GEN.append(d)

                img_def = render_mesh_dict(mesh_def, mode='xy', render_new=self.renderer)
                # img_def = np.concatenate([img_def[:256,:,:], img_def[256:,:,:]],axis=1)
                img_def = np.concatenate([img_def[256:, :, :]], axis=1)

                imageio.mimsave(
                    os.path.join(self.output_folder,
                                 '%s_seed%d_%d.png' % (self.eval_mode, self.seed, int(time.time()))),
                    [img_def], codec='libx264')

    def action_z_shape(self, batch=None):
        global ACT_GEN
        self.eval_mode = 'z_shape'
        batch_list = []
        if batch is None:
            idx_b = np.random.randint(0, self.meta_info.n_samples)
            while len(batch_list) < self.max_samples:
                idx_a = idx_b
                idx_b = np.random.randint(0, self.meta_info.n_samples)

                z_shape_a = self.model.z_shapes.weight.data[idx_a]
                z_shape_b = self.model.z_shapes.weight.data[idx_b]
                z_detail = self.model.z_details.weight.data.mean(0)

                for i in range(10):
                    z_shape = torch.lerp(z_shape_a, z_shape_b, i / 10)

                    batch = {'z_shape': z_shape[None],
                             'z_detail': z_detail[None],
                             'smpl_params': self.smpl_param_zero}

                    batch_list.append(batch)
        else:
            z_shape = None
            while len(batch_list) < self.max_samples:
                if len(batch_list) == 0:
                    z_shape_a = batch['z_shape'].squeeze()
                else:
                    z_shape_a = z_shape

                idx_b = np.random.randint(0, self.meta_info.n_samples)
                z_shape_b = self.model.z_shapes.weight.data[idx_b]

                for i in range(10):
                    z_shape = torch.lerp(z_shape_a, z_shape_b, i / 10)

                    batch = {'z_shape': z_shape[None],
                             'z_detail': batch['z_detail'],
                             'smpl_params': batch['smpl_params']}

                    batch_list.append(batch)

        ACT_GEN = batch_list
        self.get_mesh(batch_list)
        return batch_list

    def action_z_detail(self, batch=None):
        global ACT_GEN
        self.eval_mode = 'z_detail'
        batch_list = []

        if batch is None:
            idx_b = np.random.randint(0, self.meta_info.n_samples)
            while len(batch_list) < self.max_samples:
                idx_a = idx_b
                idx_b = np.random.randint(0, self.meta_info.n_samples)

                z_detail_a = self.model.z_details.weight.data[idx_a]
                z_detail_b = self.model.z_details.weight.data[idx_b]
                z_shape = self.model.z_shapes.weight.data.mean(0)

                for i in range(10):
                    z_detail = torch.lerp(z_detail_a, z_detail_b, i / 10)

                    batch = {'z_shape': z_shape[None],
                             'z_detail': z_detail[None],
                             'smpl_params': self.smpl_param_zero}

                    batch_list.append(batch)
        else:
            z_detail = None
            while len(batch_list) < self.max_samples:
                if len(batch_list) == 0:
                    z_detail_a = batch['z_detail'].squeeze()
                else:
                    z_detail_a = z_detail
                idx_b = np.random.randint(0, self.meta_info.n_samples)
                z_detail_b = self.model.z_details.weight.data[idx_b]
                for i in range(10):
                    z_detail = torch.lerp(z_detail_a, z_detail_b, i / 10)

                    batch = {'z_shape': batch['z_shape'],
                             'z_detail': z_detail[None],
                             'smpl_params': batch['smpl_params']}

                    batch_list.append(batch)

        ACT_GEN = batch_list
        self.get_mesh(batch_list)
        return batch_list

    def action_betas(self, batch=None):
        global ACT_GEN
        self.eval_mode = 'betas'

        batch_list = []
        betas = torch.cat([torch.linspace(0, -2, 10),
                           torch.linspace(-2, 0, 10),
                           torch.linspace(0, 2, 10),
                           torch.linspace(2, 0, 10)])
        if batch is None:
            z_shape = self.model.z_shapes.weight.data.mean(0)
            z_detail = self.model.z_details.weight.data.mean(0)

            for i in range(len(betas)):
                smpl_param = self.smpl_param_zero.clone()
                smpl_param[:, -10] = betas[i]

                batch = {'z_shape': z_shape[None],
                         'z_detail': z_detail[None],
                         'smpl_params': smpl_param}

                batch_list.append(batch)
        else:
            for i in range(len(betas)):
                smpl_param = batch['smpl_params'].clone()
                smpl_param[:, -10] = betas[i]

                batch = {'z_shape': batch['z_shape'],
                         'z_detail': batch['z_detail'],
                         'smpl_params': smpl_param}

                batch_list.append(batch)

        ACT_GEN = batch_list
        self.get_mesh(batch_list)
        return batch_list

    def action_thetas(self, batch=None, n_pose=None):
        global ACT_GEN
        self.eval_mode = 'thetas'
        batch_list = []
        if batch is None:
            z_shape = self.model.z_shapes.weight.data.mean(0)
            z_detail = self.model.z_details.weight.data.mean(0)
            for i in range(len(self.smpl_param_anim)):
                batch = {'z_shape': z_shape[None],
                         'z_detail': z_detail[None],
                         'smpl_params': self.smpl_param_anim[[i]]
                         }

                batch_list.append(batch)
        else:
            x = int(n_pose / 2)
            max = len(self.smpl_param_anim.tolist())
            # print(max)
            j = torch.where((self.smpl_param_anim == batch['smpl_params'].unsqueeze(1))
                            .prod(dim=2))[1].cpu().numpy()[0].item()
            if j + x > max:
                lower = ((j + x) - max) + x
                upper = (j + x) - max
            elif j - x < 0:
                lower = abs(j - x)
                upper = x + abs(j - x)
            else:
                lower = x
                upper = x
            for i in range(len(self.smpl_param_anim.tolist()[j - lower:j + upper])):
                batch = {'z_shape': batch['z_shape'],
                         'z_detail': batch['z_detail'],
                         'smpl_params': self.smpl_param_anim[[i + (j - lower)]]
                         }

                batch_list.append(batch)

        ACT_GEN = batch_list
        # print("N° Poses: ", len(batch_list))
        self.get_mesh(batch_list)
        return batch_list

    def action_sample(self):
        self.eval_mode = 'sample'

        batch_list = []

        z_shapes, z_details = self.model.sample_codes(self.max_samples)
        x = len(BATCH_GEN)
        for i in range(len(z_shapes)):
            id_smpl = np.random.randint(len(self.smpl_param_anim))
            batch = {f'batch_{i + x}': {'z_shape': z_shapes[i][None],
                                        'z_detail': z_details[i][None],
                                        'smpl_params': self.smpl_param_anim[id_smpl][None],
                                        }
                     }
            # print(batch)
            batch_list.append(batch)
            BATCH_GEN.append(batch)
        #print(BATCH_GEN)
        self.get_mesh(batch_list)
        return batch_list
