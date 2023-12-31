bl_info = {
    "name": "gDNA for Blender",
    "author": "Arment Pelivani, Enrico Tarsi, Riccardo Mancini",
    "version": (1, 0, 0),
    "blender": (3, 6, 0),
    "location": "Viewport > Right panel",
    "description": "gDNA for Blender",
    "category": "gDNA"}

import bpy
from bpy.props import BoolProperty, EnumProperty, FloatProperty, PointerProperty, IntProperty, StringProperty
from bpy.types import PropertyGroup
import os
import sys
import torch
import glob

import numpy as np
from pathlib import Path
from functools import partial
from threading import Thread
from math import radians
import shutil

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

# remove old temporary dir
dirpath = Path(f'{ROOT}/tmp')
if os.path.exists(dirpath) and os.path.isdir(dirpath):
    shutil.rmtree(dirpath)

sys.path.insert(1, f'{ROOT}/gdna')

from test import GDNA
import test as gl

objR, objT, twrv, last_obj = None, None, None, None
G = {'m1': [], 'm2': []}
Last = {}
zero = {"Shape": True, "Scale": True, "Pose": True}
index_Gen = {}
zero_s = {"Slider": "0", "Val": 0}
memory_slider = {}
memory_shading = {}
num = 0


class ThreadWithReturnValue(Thread):
    def __init__(self, group=None, target=None, name=None,
                 args=(), kwargs={}, Verbose=None):
        Thread.__init__(self, group, target, name, args, kwargs)
        self._return = None

    def run(self):
        if self._target is not None:
            self._return = self._target(*self._args,
                                        **self._kwargs)

    def join(self, *args):
        Thread.join(self, *args)
        return self._return


def save_tmp(avatar_id, eval_mode, expname, btch_sample=None, i=None):
    path = f'{ROOT}/tmp'
    if not os.path.exists(path):
        os.makedirs(path)
    if eval_mode == 'sample':
        np.savez(f'{path}/ID_{avatar_id}_{eval_mode}_{expname}.npz', gl.MESH_GEN[i])
        torch.save(btch_sample, f'{path}/ID_{avatar_id}_{eval_mode}_{expname}.pt')

        mod = 'm1' if expname == 'renderpeople' else 'm2'
        G[mod].append(f'Avatar_{avatar_id}')
    else:
        np.savez(f'{path}/ID_{avatar_id}_{eval_mode}_{expname}.npz', *gl.MESH_GEN)
        torch.save(gl.ACT_GEN, f'{path}/ID_{avatar_id}_{eval_mode}_{expname}.pt')


def load_tmp(avatar_id, eval_mode, expname):
    path = f'{ROOT}/tmp'
    loaded_data = np.load(f'{path}/ID_{avatar_id}_{eval_mode}_{expname}.npz', allow_pickle=True)
    gl.MESH_GEN = [loaded_data[f"arr_{i}"][()] for i in range(len(loaded_data.files))]

    gl.ACT_GEN = torch.load(f'{path}/ID_{avatar_id}_{eval_mode}_{expname}.pt')


def check_tmp(eval_mode):
    path = f'{ROOT}/tmp'
    avatar_id = bpy.context.view_layer.objects.active.name.split('_')[-1]
    if len(glob.glob(f'{path}/ID_{avatar_id}_{eval_mode}_*')) > 0:
        return True
    else:
        return False


def get_n_avatar():
    path = f'{ROOT}/tmp'
    return len(glob.glob(f'{path}/ID_*_sample_*.npz'))


def array2mesh(verts, faces, replace=False):
    global last_obj, num
    if not replace:
        mesh = bpy.data.meshes.new(f'Avatar_{str(num)}')
        obj = bpy.data.objects.new(f'Avatar_{str(num)}', mesh)
        mesh.from_pydata(verts.tolist(), [], faces.tolist())
        mesh.update(calc_edges=True)  # Update mesh with new data
        bpy.context.collection.objects.link(obj)  # Link to scene
        centr_mesh()

        bpy.context.view_layer.objects.active = obj
        Last[bpy.context.active_object] = None
        index_Gen[bpy.context.active_object] = zero.copy()
        memory_slider[bpy.context.active_object] = zero_s.copy()
        memory_shading[bpy.context.active_object] = False
        last_obj = bpy.context.active_object
        num += 1

    else:
        obj = bpy.context.view_layer.objects.active
        n = obj.name.split('_')[-1]
        mesh = bpy.data.meshes.new(f'Avatar_{n}')
        mesh.from_pydata(verts.tolist(), [], faces.tolist())
        mesh.update(calc_edges=True)
        obj.data = mesh
        obj.data.name = f'Avatar_{n}'

        bpy.context.view_layer.objects.active = obj


def centr_mesh(c=2):
    objs = [ob for ob in bpy.context.scene.objects if ob.type == "MESH" and "Avatar" in ob.name]
    l = len(objs)
    for i, obj in enumerate(objs):
        m = (((l / 2) - 1) * c) + c / 2
        x = i * c - m
        obj.location = (x, 0, 1.2)
        obj.rotation_euler = (radians(90), 0, 0)


def show_progress(area, process, obj, avatar_id=None):
    if not process.is_alive():
        area.header_text_set(None)
        if 'sample' in process._target.__name__:
            gl.ACT_GEN = process.join()
            torch.cuda.empty_cache()
            for i, batch in enumerate(gl.ACT_GEN):
                name_btch = list(batch.keys())[0]
                smpl = batch[name_btch]

                save_tmp(list(batch.keys())[0].split('_')[-1], obj.eval_mode, obj.expname, smpl, i)

        else:
            process.join()
            torch.cuda.empty_cache()

            save_tmp(avatar_id, obj.eval_mode, obj.expname)

        return

    else:
        msg = "\r{0}{1}".format('Loading', "." * show_progress.n_dot)
        area.header_text_set(msg)
        if (show_progress.n_dot + 1) % 4 == 0:
            show_progress.n_dot = 0
        else:
            show_progress.n_dot += 1
        return 0.8


def generate_mesh(process):
    if generate_mesh.n_mesh_gen == len(gl.MESH_GEN) and not process.is_alive():
        print('Avatar(s) generated!')
        return
    else:
        while generate_mesh.n_mesh_gen < len(gl.MESH_GEN):
            array2mesh(gl.MESH_GEN[generate_mesh.n_mesh_gen]['verts'],
                       gl.MESH_GEN[generate_mesh.n_mesh_gen]['faces'])
            generate_mesh.n_mesh_gen += 1

        return 0.2


def update_z_shape(self, context):
    num_slider = bpy.context.window_manager.gdna_tool.gdna_z_shape
    GDNA_Gen_Shape.bl_slider_val[bpy.context.active_object] = num_slider
    
    if num_slider != 0 and bpy.context.active_object == last_obj and Last[bpy.context.active_object] == 'Shape':
        if num_slider > len(gl.MESH_GEN):
            num_slider = len(gl.MESH_GEN)
            bpy.context.window_manager.gdna_tool.gdna_z_shape = num_slider

        array2mesh(gl.MESH_GEN[num_slider - 1]['verts'], gl.MESH_GEN[num_slider - 1]['faces'], True)
        avatar_id = bpy.context.view_layer.objects.active.name.split('_')[-1]
        a = [i for i, d in enumerate(gl.BATCH_GEN) if f'batch_{avatar_id}' in d.keys()]
        gl.BATCH_GEN[a[0]][f'batch_{avatar_id}'] = gl.ACT_GEN[num_slider - 1]


def update_scale(self, context):
    num_slider = bpy.context.window_manager.gdna_tool.gdna_scale
    GDNA_Gen_Scale.bl_slider_val[bpy.context.active_object] = num_slider

    if num_slider != 0 and bpy.context.active_object == last_obj and Last[bpy.context.active_object] == 'Scale':
        if num_slider > len(gl.MESH_GEN):
            num_slider = len(gl.MESH_GEN)
            bpy.context.window_manager.gdna_tool.gdna_scale = num_slider

        array2mesh(gl.MESH_GEN[num_slider - 1]['verts'], gl.MESH_GEN[num_slider - 1]['faces'], True)
        avatar_id = bpy.context.view_layer.objects.active.name.split('_')[-1]
        a = [i for i, d in enumerate(gl.BATCH_GEN) if f'batch_{avatar_id}' in d.keys()]
        gl.BATCH_GEN[a[0]][f'batch_{avatar_id}'] = gl.ACT_GEN[num_slider - 1]


def update_pose(self, context):
    num_slider = 0
    n_pose = memory_slider[bpy.context.active_object]["Slider"]
    x = int(n_pose)
    x = int(x / 2)
    if n_pose == "20":
        num_slider = bpy.context.window_manager.gdna_tool.gdna_pose1
        memory_slider[bpy.context.active_object]["Val"] = num_slider
    elif n_pose == "40":
        num_slider = bpy.context.window_manager.gdna_tool.gdna_pose2
        memory_slider[bpy.context.active_object]["Val"] = num_slider
    elif n_pose == "60":
        num_slider = bpy.context.window_manager.gdna_tool.gdna_pose3
        memory_slider[bpy.context.active_object]["Val"] = num_slider

    if num_slider != 0 and bpy.context.active_object == last_obj and Last[bpy.context.active_object] == 'Pose':
        num_slider += x
        if num_slider >= len(gl.MESH_GEN) != x*2:
            num_slider = len(gl.MESH_GEN)
            if n_pose == '20':
                bpy.context.window_manager.gdna_tool.gdna_pose1 = num_slider - x - 1
            elif n_pose == '40':
                bpy.context.window_manager.gdna_tool.gdna_pose2 = num_slider - x - 1
            elif n_pose == '60':
                bpy.context.window_manager.gdna_tool.gdna_pose3 = num_slider - x - 1

            if num_slider - x - 1 <= 0:
                num_slider -= 1

        if (n_pose == '20' and bpy.context.window_manager.gdna_tool.gdna_pose1 > 0) or \
                (n_pose == '40' and bpy.context.window_manager.gdna_tool.gdna_pose2 > 0) or \
                (n_pose == '60' and bpy.context.window_manager.gdna_tool.gdna_pose3 > 0):
            num_slider -= 1

        array2mesh(gl.MESH_GEN[num_slider]['verts'], gl.MESH_GEN[num_slider]['faces'], True)
        avatar_id = bpy.context.view_layer.objects.active.name.split('_')[-1]
        a = [i for i, d in enumerate(gl.BATCH_GEN) if f'batch_{avatar_id}' in d.keys()]
        gl.BATCH_GEN[a[0]][f'batch_{avatar_id}'] = gl.ACT_GEN[num_slider]


def update_shading(self, context):
    if bpy.context.window_manager.gdna_tool.gdna_shading == True:
        memory_shading[bpy.context.active_object] = True
        for face in bpy.context.active_object.data.polygons:
            face.use_smooth = True
    else:
        memory_shading[bpy.context.active_object] = False
        for face in bpy.context.active_object.data.polygons:
            face.use_smooth = False


def action_retrieve(avatar_id, eval_mode, obj, num_slider, pose=False, x=0):
    load_tmp(avatar_id, eval_mode, obj.expname)
    if pose:
        if num_slider > 0:
            num_slider += x
        else:
            num_slider += x + 1

    array2mesh(gl.MESH_GEN[num_slider - 1]['verts'], gl.MESH_GEN[num_slider - 1]['faces'], True)
    a = [i for i, d in enumerate(gl.BATCH_GEN) if f'batch_{avatar_id}' in d.keys()]
    gl.BATCH_GEN[a[0]][f'batch_{avatar_id}'] = gl.ACT_GEN[num_slider - 1]


def abil_retrieve(mod):
    parse_eval_mode = {"Shape": "z_shape", "Scale": "betas", "Pose": "thetas"}
    if (not (bpy.context.active_object in dix[mod])
            or (bpy.context.active_object.select_get() == False)
            or not (check_tmp(parse_eval_mode[mod]))
            or (Last[bpy.context.active_object] == mod)
            or twrv.is_alive()):
        return False
    else:
        return True


def abil_generate(mod):
    try:
        if (bpy.context.active_object is None
                or bpy.context.active_object.select_get() == False
                or index_Gen[bpy.context.active_object][mod] == False
                or twrv.is_alive()):
            return False
        else:
            return True
    except:
        return False


def abil_generate_pose(mod):
    try:
        if ((bpy.context.active_object is None)
                or (bpy.context.active_object.select_get() == False)
                or (index_Gen[bpy.context.active_object][mod] == False
                    and int(memory_slider[bpy.context.active_object]["Slider"]) >= int(
                            bpy.context.window_manager.gdna_tool.gdna_n_pose))
                or twrv.is_alive()):
            return False
        else:
            return True
    except:
        return False


def abil_slider(mod):
    try:
        if (bpy.context.active_object is None
                or bpy.context.active_object.select_get() == False
                or (Last[bpy.context.active_object] != mod)
                or (twrv.is_alive() and bpy.context.active_object != last_obj)):
            return False
        else:
            return True
    except:
        return False


# Property groups for UI
class GDNA_Properties(PropertyGroup):
    gdna_model: EnumProperty(
        items=[
            ('model_1', 'Renderpeople', 'Set Renderpeople Weights', '', 0),
            ('model_2', 'THuman2.0', 'Set THuman2.0 Weights', '', 1)
        ],
        default='model_1'
    )

    gdna_n_pose: EnumProperty(
        items=[
            ('20', '20', 'Generate 20 differents poses', '', 0),
            ('40', '40', 'Generate 40 differents poses', '', 1),
            ('60', '60', 'Generate 60 differents poses', '', 2)
        ],
        default='20'
    )
    gdna_remesh_mode: EnumProperty(
        items=[
            ('BLOCKS', 'BLOCKS', 'BLOCKS', '', 0),
            ('SMOOTH', 'SMOOTH', 'SMOOTH', '', 1),
            ('SHARP', 'SHARP', 'SHARP', '', 2),
            ('VOXEL', 'VOXEL', 'VOXEL', '', 3)
        ],
        default='BLOCKS'
    )
    gdna_z_shape: IntProperty(name="Select the Avatar Shape", default=0, min=0, max=10, update=update_z_shape)
    gdna_scale: IntProperty(name="Select the Avatar Size", default=0, min=0, max=40, update=update_scale)
    gdna_pose: IntProperty(name="Select the Avatar Pose", default=0, min=-20, max=20)
    gdna_pose1: IntProperty(name="Select the Avatar Pose", default=0, min=-10, max=10, update=update_pose)
    gdna_pose2: IntProperty(name="Select the Avatar Pose", default=0, min=-20, max=20, update=update_pose)
    gdna_pose3: IntProperty(name="Select the Avatar Pose", default=0, min=-30, max=30, update=update_pose)

    gdna_n_models: IntProperty(name="Number of random Avatars to generate", default=1, min=0, max=100)
    gdna_decimate_ratio: FloatProperty(name="Set Decimate Ratio for Decimate modifier", default=0.5, min=0, max=1)
    gdna_octree_depth: IntProperty(name="Set Octree Depth for Remesh modifier", default=4, min=1, max=10)
    gdna_seed: IntProperty(name="Set Seed for the Weight initialization", default=1, min=0, max=100)
    gdna_shading: BoolProperty(name="shading", description="Enable/Disable Smooth Shading on the selected Avatar",
                               default=False, update=update_shading)

    path: StringProperty(
        name="",
        description="Choose a directory:",
        default="",
        maxlen=1024,
        subtype='DIR_PATH')


# operator
class GDNA_Start(bpy.types.Operator):
    bl_idname = "object.start"
    bl_label = "Start"
    bl_description = ("Add gDNA model/s with selected weights to scene")
    bl_options = {'REGISTER', 'UNDO'}
    bl_Last_OS = None  # attributo con l'ultimo oggetto selezionato

    @classmethod
    def poll(cls, context):
        try:
            if bpy.context.active_object != GDNA_Start.bl_Last_OS:
                GDNA_Start.bl_Last_OS = bpy.context.active_object
                bpy.context.window_manager.gdna_tool.gdna_shading = memory_shading[bpy.context.active_object]

            if twrv is not None and twrv.is_alive():
                return False
            else:
                return True
        except:
            return False

    def execute(self, context):
        global objR, objT, twrv
        d_model = {'model_1': 'renderpeople', 'model_2': 'thuman'}
        n_samples = bpy.context.window_manager.gdna_tool.gdna_n_models
        seed = bpy.context.window_manager.gdna_tool.gdna_seed
        model = bpy.context.window_manager.gdna_tool.gdna_model
        if model == 'model_1':
            objR = GDNA(seed=seed, max_samples=n_samples, expname=d_model[model])
            obj = objR
        else:
            objT = GDNA(seed=seed, max_samples=n_samples, expname=d_model[model])
            obj = objT

        twrv = ThreadWithReturnValue(target=obj.action_sample)
        twrv.start()

        show_progress.n_dot = 0
        bpy.app.timers.register(partial(show_progress, bpy.context.area, twrv, obj))

        generate_mesh.n_mesh_gen = 0
        bpy.app.timers.register(partial(generate_mesh, twrv))

        return {'FINISHED'}


class GDNA_Gen_Shape(bpy.types.Operator):
    bl_idname = "object.gen_shape"
    bl_label = "Start Shape"
    bl_description = ("Start generation of 10 differents Shapes of the selected Avatar")
    bl_options = {'REGISTER', 'UNDO'}
    bl_objects = []  # lista degli oggetti su cui è stato effettuato un generate Shape
    bl_slider_val = {}  # valore dello slider di ogni oggetto presente
    bl_Last_OS = None  # variabile con l'ultimo oggetto selezionato

    @classmethod
    def poll(cls, context):
        global last_obj
        # mette a 0 lo slider se non è stato ancora modificato
        if not (bpy.context.active_object in GDNA_Gen_Shape.bl_slider_val):
            bpy.context.window_manager.gdna_tool.gdna_z_shape = 0

        if bpy.context.active_object != GDNA_Gen_Shape.bl_Last_OS:
            GDNA_Gen_Shape.bl_Last_OS = bpy.context.active_object
            bpy.context.window_manager.gdna_tool.gdna_z_shape = GDNA_Gen_Shape.bl_slider_val[bpy.context.active_object]

            if twrv is not None and not twrv.is_alive():
                # load avatar weights on click
                if bpy.context.active_object in index_Gen:
                    parse_eval_mode = {"Shape": "z_shape", "Scale": "betas", "Pose": "thetas"}
                    eval_mode = [key for key, val in index_Gen[bpy.context.active_object].items() if val == False]
                    avatar_id = bpy.context.view_layer.objects.active.name.split('_')[-1]
                    m = [k for k, v in G.items() if f'Avatar_{avatar_id}' in v]
                    if len(m) != 0:
                        obj = objR if m[0] == 'm1' else objT
                        if len(eval_mode) != 0:
                            load_tmp(avatar_id, parse_eval_mode[eval_mode[0]], obj.expname)
                        else:
                            load_tmp(avatar_id, 'sample', obj.expname)

                        last_obj = bpy.context.active_object

        return abil_generate("Shape")

    def execute(self, context):
        global objR, objT, twrv, last_obj
        bpy.context.window_manager.gdna_tool.gdna_z_shape = 0

        if not (bpy.context.active_object in self.bl_objects):
            self.bl_objects.append(bpy.context.active_object)
        Last[bpy.context.active_object] = "Shape"

        ob = bpy.context.active_object
        for o in list(index_Gen[ob].keys()):
            if o == "Shape":
                index_Gen[ob][o] = False
            else:
                index_Gen[ob][o] = True

        last_obj = bpy.context.active_object

        avatar_id = bpy.context.view_layer.objects.active.name.split('_')[-1]
        m = [k for k, v in G.items() if f'Avatar_{avatar_id}' in v]
        obj = objR if m[0] == 'm1' else objT

        a = [i for i, d in enumerate(gl.BATCH_GEN) if f'batch_{avatar_id}' in d.keys()]
        twrv = ThreadWithReturnValue(target=obj.action_z_shape, args=(gl.BATCH_GEN[a[0]][f'batch_{avatar_id}'],))
        twrv.start()

        show_progress.n_dot = 0
        bpy.app.timers.register(partial(show_progress, bpy.context.area, twrv, obj, avatar_id))

        return {'FINISHED'}


class GDNA_Gen_Scale(bpy.types.Operator):
    bl_idname = "object.gen_scale"
    bl_label = "Start Scale"
    bl_description = ("Start generation of 40 differents Sizes of the selected Avatar")
    bl_options = {'REGISTER', 'UNDO'}
    bl_objects = []
    bl_slider_val = {}
    bl_Last_OS = None

    @classmethod
    def poll(cls, context):
        if (not (bpy.context.active_object in GDNA_Gen_Scale.bl_slider_val)):
            bpy.context.window_manager.gdna_tool.gdna_scale = 0
        if (bpy.context.active_object != GDNA_Gen_Scale.bl_Last_OS):
            GDNA_Gen_Scale.bl_Last_OS = bpy.context.active_object
            bpy.context.window_manager.gdna_tool.gdna_scale = GDNA_Gen_Scale.bl_slider_val[bpy.context.active_object]
        return abil_generate("Scale")

    def execute(self, context):
        global objR, objT, twrv, last_obj
        bpy.context.window_manager.gdna_tool.gdna_scale = 0
        if (not (bpy.context.active_object in self.bl_objects)):
            self.bl_objects.append(bpy.context.active_object)
        Last[bpy.context.active_object] = "Scale"

        ob = bpy.context.active_object
        for o in list(index_Gen[ob].keys()):
            if o == "Scale":
                index_Gen[ob][o] = False
            else:
                index_Gen[ob][o] = True

        last_obj = bpy.context.active_object

        avatar_id = bpy.context.view_layer.objects.active.name.split('_')[-1]
        m = [k for k, v in G.items() if f'Avatar_{avatar_id}' in v]
        obj = objR if m[0] == 'm1' else objT

        a = [i for i, d in enumerate(gl.BATCH_GEN) if f'batch_{avatar_id}' in d.keys()]
        twrv = ThreadWithReturnValue(target=obj.action_betas, args=(gl.BATCH_GEN[a[0]][f'batch_{avatar_id}'],))
        twrv.start()

        show_progress.n_dot = 0
        bpy.app.timers.register(partial(show_progress, bpy.context.area, twrv, obj, avatar_id))

        return {'FINISHED'}


class GDNA_Gen_Pose(bpy.types.Operator):
    bl_idname = "object.gen_pose"
    bl_label = "Start Pose"
    bl_description = ("Start generation of N differents Poses of the selected Avatar")
    bl_options = {'REGISTER', 'UNDO'}
    bl_objects = []
    bl_slider_val = {}
    bl_Last_OS = None

    @classmethod
    def poll(cls, context):
        try:
            if bpy.context.active_object != GDNA_Gen_Pose.bl_Last_OS:
                GDNA_Gen_Pose.bl_Last_OS = bpy.context.active_object
                if memory_slider[bpy.context.active_object]["Slider"] == "20":
                    bpy.context.window_manager.gdna_tool.gdna_pose1 = memory_slider[bpy.context.active_object]["Val"]
                    bpy.context.window_manager.gdna_tool.gdna_n_pose = "20"
                elif memory_slider[bpy.context.active_object]["Slider"] == "40":
                    bpy.context.window_manager.gdna_tool.gdna_pose2 = memory_slider[bpy.context.active_object]["Val"]
                    bpy.context.window_manager.gdna_tool.gdna_n_pose = "40"
                elif memory_slider[bpy.context.active_object]["Slider"] == "60":
                    bpy.context.window_manager.gdna_tool.gdna_pose3 = memory_slider[bpy.context.active_object]["Val"]
                    bpy.context.window_manager.gdna_tool.gdna_n_pose = "60"
                elif (memory_slider[bpy.context.active_object][
                          "Slider"] == "0"):  # quando crea un nuovo oggetto il selettore di n_pose è su 20
                    bpy.context.window_manager.gdna_tool.gdna_n_pose = "20"
            return abil_generate_pose("Pose")
        except:
            return False

    def execute(self, context):
        global objR, objT, twrv, last_obj
        if bpy.context.window_manager.gdna_tool.gdna_n_pose == "20":
            bpy.context.window_manager.gdna_tool.gdna_pose1 = 0
        elif bpy.context.window_manager.gdna_tool.gdna_n_pose == "40":
            bpy.context.window_manager.gdna_tool.gdna_pose2 = 0
        elif bpy.context.window_manager.gdna_tool.gdna_n_pose == "60":
            bpy.context.window_manager.gdna_tool.gdna_pose3 = 0

        if not (bpy.context.active_object in self.bl_objects):
            self.bl_objects.append(bpy.context.active_object)
        Last[bpy.context.active_object] = "Pose"

        ob = bpy.context.active_object
        for o in list(index_Gen[ob].keys()):
            if o == "Pose":
                index_Gen[ob][o] = False
            else:
                index_Gen[ob][o] = True

        memory_slider[bpy.context.active_object]["Slider"] = bpy.context.window_manager.gdna_tool.gdna_n_pose
        last_obj = bpy.context.active_object

        avatar_id = bpy.context.view_layer.objects.active.name.split('_')[-1]
        m = [k for k, v in G.items() if f'Avatar_{avatar_id}' in v]
        obj = objR if m[0] == 'm1' else objT

        a = [i for i, d in enumerate(gl.BATCH_GEN) if f'batch_{avatar_id}' in d.keys()]
        twrv = ThreadWithReturnValue(target=obj.action_thetas,
                                     args=(gl.BATCH_GEN[a[0]][f'batch_{avatar_id}'],
                                           int(memory_slider[bpy.context.active_object]["Slider"])))
        twrv.start()

        show_progress.n_dot = 0
        bpy.app.timers.register(partial(show_progress, bpy.context.area, twrv, obj, avatar_id))

        return {'FINISHED'}


class GDNA_Ret_Shape(bpy.types.Operator):
    bl_idname = "object.ret_shape"
    bl_label = "Retrive Shape"
    bl_description = ("Retrive the latest Shape generation for the selected Avatar")
    bl_options = {'REGISTER', 'UNDO'}
    bl_objects = []

    @classmethod
    def poll(cls, context):
        return abil_retrieve("Shape")

    def execute(self, context):
        global objR, objT, last_obj
        Last[bpy.context.active_object] = "Shape"

        ob = bpy.context.active_object
        for o in list(index_Gen[ob].keys()):
            if o == "Shape":
                index_Gen[ob][o] = False
            else:
                index_Gen[ob][o] = True

        last_obj = bpy.context.active_object
        avatar_id = bpy.context.view_layer.objects.active.name.split('_')[-1]
        m = [k for k, v in G.items() if f'Avatar_{avatar_id}' in v]
        obj = objR if m[0] == 'm1' else objT

        num_slider = bpy.context.window_manager.gdna_tool.gdna_z_shape
        ret_thread = ThreadWithReturnValue(target=action_retrieve, args=(avatar_id, 'z_shape', obj, num_slider,))
        ret_thread.start()
        ret_thread.join()
        return {'FINISHED'}


class GDNA_Ret_Scale(bpy.types.Operator):
    bl_idname = "object.ret_scale"
    bl_label = "Retrive Scale"
    bl_description = ("Retrive the latest Size generation for the selected Avatar")
    bl_options = {'REGISTER', 'UNDO'}
    bl_objects = []

    @classmethod
    def poll(cls, context):
        return abil_retrieve("Scale")

    def execute(self, context):
        global objR, objT, last_obj
        Last[bpy.context.active_object] = "Scale"

        ob = bpy.context.active_object
        for o in list(index_Gen[ob].keys()):
            if o == "Scale":
                index_Gen[ob][o] = False
            else:
                index_Gen[ob][o] = True

        last_obj = bpy.context.active_object
        avatar_id = bpy.context.view_layer.objects.active.name.split('_')[-1]
        m = [k for k, v in G.items() if f'Avatar_{avatar_id}' in v]
        obj = objR if m[0] == 'm1' else objT

        num_slider = bpy.context.window_manager.gdna_tool.gdna_scale
        ret_thread = ThreadWithReturnValue(target=action_retrieve, args=(avatar_id, 'betas', obj, num_slider,))
        ret_thread.start()
        ret_thread.join()

        return {'FINISHED'}


class GDNA_Ret_Pose(bpy.types.Operator):
    bl_idname = "object.ret_pose"
    bl_label = "Retrive Pose"
    bl_description = ("Retrive the latest Pose generation for the selected Avatar")
    bl_options = {'REGISTER', 'UNDO'}
    bl_objects = []

    @classmethod
    def poll(cls, context):
        return abil_retrieve("Pose")

    def execute(self, context):
        global objR, objT, last_obj
        Last[bpy.context.active_object] = "Pose"

        ob = bpy.context.active_object
        for o in list(index_Gen[ob].keys()):
            if o == "Pose":
                index_Gen[ob][o] = False
            else:
                index_Gen[ob][o] = True

        last_obj = bpy.context.active_object
        avatar_id = bpy.context.view_layer.objects.active.name.split('_')[-1]
        m = [k for k, v in G.items() if f'Avatar_{avatar_id}' in v]
        obj = objR if m[0] == 'm1' else objT

        num_slider = memory_slider[bpy.context.active_object]["Val"]
        n_pose = int(memory_slider[bpy.context.active_object]["Slider"])
        n_pose = int(n_pose / 2)
        ret_thread = ThreadWithReturnValue(target=action_retrieve,
                                           args=(avatar_id, 'thetas', obj, num_slider, True, n_pose))
        ret_thread.start()
        ret_thread.join()
        return {'FINISHED'}


class GDNA_Reset(bpy.types.Operator):
    bl_idname = "object.reset"
    bl_label = "Total Retrive"
    bl_description = ("Return to initial Avatar generation")
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        try:
            if (bpy.context.active_object is None
                    or bpy.context.active_object.select_get() == False
                    or twrv is not None and twrv.is_alive()):
                return False
            else:
                return True
        except:
            return False

    def execute(self, context):
        global objR, objT, last_obj
        bpy.context.window_manager.gdna_tool.gdna_z_shape = 0
        bpy.context.window_manager.gdna_tool.gdna_scale = 0
        ###
        bpy.context.window_manager.gdna_tool.gdna_pose1 = 0
        bpy.context.window_manager.gdna_tool.gdna_pose2 = 0
        bpy.context.window_manager.gdna_tool.gdna_pose3 = 0
        bpy.context.window_manager.gdna_tool.gdna_n_pose = "20"

        index_Gen[bpy.context.active_object]["Shape"] = True
        index_Gen[bpy.context.active_object]["Scale"] = True
        index_Gen[bpy.context.active_object]["Pose"] = True
        Last[bpy.context.active_object] = None
        memory_slider[bpy.context.active_object] = zero_s.copy()
        last_obj = bpy.context.active_object

        # eliminare file di retrive per disabilitare i retrieve
        path = f'{ROOT}/tmp'
        avatar_id = bpy.context.view_layer.objects.active.name.split('_')[-1]
        file_paths = glob.glob(f'{path}/ID_{avatar_id}_*')
        for file_path in file_paths:
            if 'sample' not in file_path:
                os.remove(file_path)

        m = [k for k, v in G.items() if f'Avatar_{avatar_id}' in v]
        obj = objR if m[0] == 'm1' else objT
        # action_retrieve(avatar_id, 'sample', obj, 1)
        load_tmp(avatar_id, 'sample', obj.expname)

        array2mesh(gl.MESH_GEN[0]['verts'], gl.MESH_GEN[0]['faces'], True)
        a = [i for i, d in enumerate(gl.BATCH_GEN) if f'batch_{avatar_id}' in d.keys()]
        gl.BATCH_GEN[a[0]][f'batch_{avatar_id}'] = gl.ACT_GEN

        return {'FINISHED'}


class Organize(bpy.types.Operator):
    bl_idname = "object.organize"
    bl_label = "Organize"
    bl_description = "Organize all Avatars in the scene"

    @classmethod
    def poll(cls, context):
        if ((bpy.context.active_object == None) or (bpy.context.active_object.select_get() == False)):
            return False
        else:
            return True

    def execute(self, context):
        centr_mesh()
        return {'FINISHED'}


class Decimate(bpy.types.Operator):
    bl_idname = "object.decimate"
    bl_label = "Decimate"
    bl_description = "Apply the Decimate modifier to reduce the vertex/face on the selected Avatar with selected Ratio"

    @classmethod
    def poll(cls, context):
        if ((bpy.context.active_object == None) or (bpy.context.active_object.select_get() == False)):
            return False
        else:
            return True

    def execute(self, context):
        # Enable Decimate Modifier
        modifier = bpy.context.active_object.modifiers.new('DecimateMod', 'DECIMATE')
        # ratio definisce il rapporto di riduzione dei poligoni della mesh ed assume un valore compreso tra 0-1;
        # 0.5 significa che riduco della metà i poligoni della mesh
        modifier.ratio = bpy.context.window_manager.gdna_tool.gdna_decimate_ratio
        return {'FINISHED'}


class Remesh(bpy.types.Operator):
    bl_idname = "object.remesh"
    bl_label = "Octree"
    bl_description = "Apply the Remesh modifier set in 'remesh mode' to the selected Avatar with selected Octree Depth"

    @classmethod
    def poll(cls, context):

        if ((bpy.context.active_object == None) or (bpy.context.active_object.select_get() == False)):
            return False
        else:
            return True

    def execute(self, context):
        remesh_modifier = bpy.context.active_object.modifiers.new("Remesh", 'REMESH')

        # Imposta il tipo di remesh
        remesh_modifier.mode = bpy.context.window_manager.gdna_tool.gdna_remesh_mode

        # Imposta la profondità Octree Depth
        remesh_modifier.octree_depth = bpy.context.window_manager.gdna_tool.gdna_octree_depth

        return {'FINISHED'}


class Save(bpy.types.Operator):
    bl_idname = "object.save"
    bl_label = "Save"
    bl_description = "Save selected Avatar"

    def execute(self, context):
        myfile = bpy.context.active_object
        name = myfile.name

        if myfile.name in bpy.data.objects:
            bpy.ops.object.select_all(action='DESELECT')
            bpy.data.objects[myfile.name].select_set(True)

            percorso_file_esportazione = bpy.context.window_manager.gdna_tool.path
            bpy.ops.export_scene.obj(filepath=f'{percorso_file_esportazione}{name}.obj', check_existing=True,
                                     use_materials=False, use_selection=True)
            print(f"Object '{myfile.name}' successfully exported")
        else:
            print(f"Object with name '{myfile.name}' not exist in the scene")

        return {'FINISHED'}


dix = {"Shape": GDNA_Gen_Shape.bl_objects,
       "Scale": GDNA_Gen_Scale.bl_objects,
       "Pose": GDNA_Gen_Pose.bl_objects
       }


# UI
class GDNA_PT_Model(bpy.types.Panel):
    bl_label = "gDNA Model"
    bl_category = "gDNA"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"

    def draw(self, context):
        layout = self.layout
        layout.label(text="Model:")
        layout.prop(context.window_manager.gdna_tool, "gdna_model", expand=True)
        
        col = layout.column(align=True)
        split = col.split(factor=0.50, align=False)
        split.label(text="Number:")
        split.label(text="Seed:")
        
        split = col.split(factor=0.5, align=False)
        split.prop(context.window_manager.gdna_tool, "gdna_n_models", text="")
        split.prop(context.window_manager.gdna_tool, "gdna_seed", text="")
        col.separator()
        col.operator("object.start", text="Create", icon="CUBE")


class GDNA_PT_Edit(bpy.types.Panel):
    bl_label = "Edit"
    bl_category = "gDNA"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    label = ''

    def draw(self, context):
        layout = self.layout
        col = layout.column(align=True)
        row = col.row(align=True)
        try:
            index_Gen[bpy.context.active_object]
            if ((bpy.context.active_object == None) or (bpy.context.active_object.select_get() == False)):
                layout.enabled = False
            else:
                layout.enabled = True
            if ((bpy.context.active_object == None) or (bpy.context.active_object.select_get() == False)):
                self.label = 'No Avatar Selected'
            else:
                self.label = bpy.context.active_object.name
        except:
            self.label = 'No Avatar Selected'
            layout.enabled = False

        row.label(text=self.label)

        ##Shape##
        col = layout.box().column(align=True)
        split = col.split(factor=0.21, align=True)
        # Slider
        split.enabled = abil_slider("Shape")
        split.label(text="Shape:")
        split.prop(context.window_manager.gdna_tool, "gdna_z_shape", text="", slider=False)
        # abilitazione Generate
        col.separator()
        row = col.row(align=True)
        row.operator("object.gen_shape", text="Generate", icon="PLUS")
        # abilitazione Retrieve
        row.operator("object.ret_shape", text="Retrieve", icon="RECOVER_LAST")
        col.separator()

        ##Scale##
        col = layout.box().column(align=True)
        split = col.split(factor=0.21, align=True)
        # Slider
        split.enabled = abil_slider("Scale")
        split.label(text="Size:")
        split.prop(context.window_manager.gdna_tool, "gdna_scale", text='', slider=False)
        # Abilitazione Generate
        col.separator()
        row = col.row(align=True)
        row.operator("object.gen_scale", text="Generate", icon="PLUS")
        # Retrieve
        row.operator("object.ret_scale", text="Retrieve", icon="RECOVER_LAST")
        col.separator()

        ##Pose##
        col = layout.box().column(align=True)
        split = col.split(factor=0.21, align=True)
        # Slider
        split.enabled = abil_slider("Pose")
        split.label(text="Pose:")
        try:
            if (bpy.context.active_object == None or memory_slider[bpy.context.active_object]["Slider"] == "0"):
                split.prop(context.window_manager.gdna_tool, "gdna_pose", text="", slider=False)
            elif (memory_slider[bpy.context.active_object]["Slider"] == "20"):
                split.prop(context.window_manager.gdna_tool, "gdna_pose1", text="(20)", slider=False)
            elif (memory_slider[bpy.context.active_object]["Slider"] == "40"):
                split.prop(context.window_manager.gdna_tool, "gdna_pose2", text="(40)", slider=False)
            elif (memory_slider[bpy.context.active_object]["Slider"] == "60"):
                split.prop(context.window_manager.gdna_tool, "gdna_pose3", text="(60)", slider=False)
        except:
            split.prop(context.window_manager.gdna_tool, "gdna_pose", text="", slider=False)
        # N. Pose
        col.separator()
        row = col.row(align=True)
        row.prop(context.window_manager.gdna_tool, "gdna_n_pose", expand=True)

        # Generate
        row = col.row(align=True)
        col.separator()
        row.operator("object.gen_pose", text="Generate", icon="PLUS")
        
        # Retrieve
        row.operator("object.ret_pose", text="Retrieve", icon="RECOVER_LAST")

        ##Reset
        layout.operator("object.reset", text="Reset", icon="RECOVER_LAST")


class GDNA_PT_Utility(bpy.types.Panel):
    bl_label = "Utility"
    bl_category = "gDNA"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"

    def draw(self, context):
        layout = self.layout
        try:
            index_Gen[bpy.context.active_object]
            if ((bpy.context.active_object == None) or (bpy.context.active_object.select_get() == False)):
                layout.enabled = False
            else:
                layout.enabled = True
        except:
            layout.enabled = False

        layout.operator("object.organize", icon='TRACKER')
        # layout.separator()
        col = layout.column(align=True)
        split = col.split(factor=0.45, align=False)
        split.label(text='Remesh mode:')
        split.prop(context.window_manager.gdna_tool, "gdna_remesh_mode", text='')
        

        split = col.split(factor=0.50, align=True)

        split.operator("object.remesh", text="Remesh", icon='MOD_DECIM')
        split.label(text="Depth:")
        split.prop(context.window_manager.gdna_tool, "gdna_octree_depth", text="", slider=True)

        col = layout.column(align=True)
        split = col.split(factor=0.50, align=True)
        split.operator("object.decimate", text="Decimate", icon='MOD_DECIM')
        split.label(text="Ratio:")
        split.prop(context.window_manager.gdna_tool, "gdna_decimate_ratio", text="", slider=True)

        layout.prop(context.window_manager.gdna_tool, "gdna_shading", text="Smooth Shading")

        layout.label(text="Export Avatar:")
        col = layout.column(align=True)
        col.prop(context.window_manager.gdna_tool, "path", text="")
        layout.operator("object.save", icon='IMPORT')


classes = [
    GDNA_Properties,
    GDNA_Start,
    GDNA_Gen_Shape,
    GDNA_Gen_Scale,
    GDNA_Gen_Pose,
    GDNA_Ret_Shape,
    GDNA_Ret_Scale,
    GDNA_Ret_Pose,
    GDNA_Reset,
    Organize,
    Decimate,
    Remesh,
    Save,
    GDNA_PT_Model,
    GDNA_PT_Edit,
    GDNA_PT_Utility
]


def register():
    from bpy.utils import register_class
    for cls in classes:
        register_class(cls)
    # Store properties under WindowManager (not Scene) so that they are not saved in .blend files and always show
    # default values after loading
    bpy.types.WindowManager.gdna_tool = PointerProperty(type=GDNA_Properties)


def unregister():
    from bpy.utils import unregister_class
    for cls in classes:
        unregister_class(cls)

    del bpy.types.WindowManager.gdna_tool


if __name__ == "__main__":
    register()
