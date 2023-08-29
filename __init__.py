import glob

bl_info = {
    "name": "GDNA for Blender",
    "author": "Arment Pelivani, Enrico Tarsi, Riccardo Mancini",
    "version": (2, 0, 0),
    "blender": (3, 0, 1),
    "location": "Viewport > Right panel",
    "description": "gDNA for Blender",
    "category": "GDNA"}

import bpy
from bpy.props import BoolProperty, EnumProperty, FloatProperty, PointerProperty, IntProperty
from bpy.types import PropertyGroup
import os
import sys
import torch

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

# Blender path where addons is installed
# print(bpy.utils.user_resource('SCRIPTS', path='addons'))
objR, objT, twrv, last_obj = None, None, None, None
G = {'m1': [], 'm2': []}
Last = {}
zero = {"Shape": True, "Details": True, "Scale": True, "Pose": True}
index_Gen = {}
zero_s = {"Slider": "0", "Val": 0}
memory_slider = {}


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


def array2mesh(verts, faces, replace=False):
    global last_obj
    n_avatars = list(filter(lambda x: 'Avatar' in x.name, bpy.data.objects))
    num = len(n_avatars) if len(n_avatars) != 0 else 0
    if not replace:
        mesh = bpy.data.meshes.new(f'Avatar_{num}')
        obj = bpy.data.objects.new(f'Avatar_{num}', mesh)
        mesh.from_pydata(verts.tolist(), [], faces.tolist())
        mesh.update(calc_edges=True)  # Update mesh with new data
        bpy.context.collection.objects.link(obj)  # Link to scene
        centr_mesh()

        bpy.context.view_layer.objects.active = obj
        Last[bpy.context.active_object] = None
        index_Gen[bpy.context.active_object] = zero.copy()
        memory_slider[bpy.context.active_object] = zero_s.copy()
        last_obj = bpy.context.active_object
        # print(index_Gen)

    else:
        obj = bpy.context.view_layer.objects.active
        num = obj.name.split('_')[-1]
        mesh = bpy.data.meshes.new(f'Avatar_{num}')
        mesh.from_pydata(verts.tolist(), [], faces.tolist())
        mesh.update(calc_edges=True)
        obj.data = mesh
        obj.data.name = f'Avatar_{num}'

        bpy.context.view_layer.objects.active = obj


def array2bones(bones, num):
    armature = bpy.data.armatures.new(f'Armature{num}')
    rig = bpy.data.objects.new(f'Armature{num}', armature)
    bpy.context.scene.collection.objects.link(rig)

    bpy.context.view_layer.objects.active = rig
    bpy.ops.object.editmode_toggle()

    for i, bone in enumerate(bones[:-1]):
        # create new bone
        current_bone = armature.edit_bones.new(f'Bone{i}')
        # print('Head: ', bone)
        next_bone_vector = bones[i + 1]
        # print('Tail: ', next_bone_vector)
        current_bone.head = bone
        current_bone.tail = next_bone_vector
        if i == 0:
            parent_bone = current_bone
        elif i == (len(bones) - 1):
            current_bone.parent = parent_bone
            current_bone.use_connect = True
        else:
            # connect
            current_bone.parent = parent_bone
            current_bone.use_connect = True

            # save bone, its tail position (next bone will be moved to it) and quaternion rotation
            parent_bone = current_bone

    bpy.ops.object.editmode_toggle()


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
        print('Created!')
        if 'sample' in process._target.__name__:
            gl.ACT_GEN = process.join()
            # print(gl.MESH_GEN)
            # print(gl.ACT_GEN)
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
        print('Mesh generated!')
        return
    else:
        while generate_mesh.n_mesh_gen < len(gl.MESH_GEN):
            array2mesh(gl.MESH_GEN[generate_mesh.n_mesh_gen]['verts'],
                       gl.MESH_GEN[generate_mesh.n_mesh_gen]['faces'])
            generate_mesh.n_mesh_gen += 1

        return 1.5


def update_z_shape(self, context):
    #print("shape")
    num_slider = bpy.context.window_manager.gdna_tool.gdna_z_shape
    GDNA_Gen_Shape.bl_slider_val[bpy.context.active_object] = num_slider

    print(num_slider)
    if num_slider != 0 and bpy.context.active_object == last_obj:
        if num_slider > len(gl.MESH_GEN):
            num_slider = len(gl.MESH_GEN)
            bpy.context.window_manager.gdna_tool.gdna_z_shape = num_slider

        GDNA_Gen_Shape.bl_slider_val[bpy.context.active_object] = num_slider
        array2mesh(gl.MESH_GEN[num_slider - 1]['verts'], gl.MESH_GEN[num_slider - 1]['faces'], True)
        avatar_id = bpy.context.view_layer.objects.active.name.split('_')[-1]
        a = [i for i, d in enumerate(gl.BATCH_GEN) if f'batch_{avatar_id}' in d.keys()]
        gl.BATCH_GEN[a[0]][f'batch_{avatar_id}'] = gl.ACT_GEN[num_slider - 1]
    else:
        bpy.context.window_manager.gdna_tool.gdna_z_shape = GDNA_Gen_Shape.bl_slider_val[bpy.context.active_object]

def update_z_details(self, context):
    #print("shape")
    num_slider = bpy.context.window_manager.gdna_tool.gdna_z_details
    if num_slider > len(gl.MESH_GEN):
        num_slider = len(gl.MESH_GEN)
        bpy.context.window_manager.gdna_tool.gdna_z_details = num_slider

    GDNA_Gen_Details.bl_slider_val[bpy.context.active_object] = num_slider
    print(num_slider)
    if num_slider != 0 and bpy.context.active_object == last_obj:
        array2mesh(gl.MESH_GEN[num_slider - 1]['verts'], gl.MESH_GEN[num_slider - 1]['faces'], True)
        avatar_id = bpy.context.view_layer.objects.active.name.split('_')[-1]
        a = [i for i, d in enumerate(gl.BATCH_GEN) if f'batch_{avatar_id}' in d.keys()]
        gl.BATCH_GEN[a[0]][f'batch_{avatar_id}'] = gl.ACT_GEN[num_slider - 1]


def update_scale(self, context):
    #print("scale")
    num_slider = bpy.context.window_manager.gdna_tool.gdna_scale
    if num_slider > len(gl.MESH_GEN):
        num_slider = len(gl.MESH_GEN)
        bpy.context.window_manager.gdna_tool.gdna_scale = num_slider

    GDNA_Gen_Scale.bl_slider_val[bpy.context.active_object] = num_slider
    print(num_slider)
    if num_slider != 0 and bpy.context.active_object == last_obj:
        array2mesh(gl.MESH_GEN[num_slider - 1]['verts'], gl.MESH_GEN[num_slider - 1]['faces'], True)
        avatar_id = bpy.context.view_layer.objects.active.name.split('_')[-1]
        a = [i for i, d in enumerate(gl.BATCH_GEN) if f'batch_{avatar_id}' in d.keys()]
        gl.BATCH_GEN[a[0]][f'batch_{avatar_id}'] = gl.ACT_GEN[num_slider - 1]
    else:
        bpy.context.window_manager.gdna_tool.gdna_scale = GDNA_Gen_Scale.bl_slider_val[bpy.context.active_object]


def update_pose(self, context):
    num_slider = 0
    n_pose = memory_slider[bpy.context.active_object]["Slider"]
    x = int(n_pose)
    x = int(x / 2)
    if bpy.context.active_object == last_obj:
        if n_pose == "20":
            num_slider = bpy.context.window_manager.gdna_tool.gdna_pose1 + x
            if num_slider > len(gl.MESH_GEN):
                num_slider = len(gl.MESH_GEN)
                bpy.context.window_manager.gdna_tool.gdna_pose1 = num_slider - x
            act_n = bpy.context.window_manager.gdna_tool.gdna_pose1
            if bpy.context.window_manager.gdna_tool.gdna_pose1 > 0:
                num_slider -= 1
            memory_slider[bpy.context.active_object]["Val"] = num_slider



        elif n_pose == "40":
            num_slider = bpy.context.window_manager.gdna_tool.gdna_pose2
            memory_slider[bpy.context.active_object]["Val"] = num_slider
        elif n_pose == "60":
            num_slider = bpy.context.window_manager.gdna_tool.gdna_pose3
            memory_slider[bpy.context.active_object]["Val"] = num_slider


    print(num_slider)




    '''if num_slider != 0 and bpy.context.active_object == last_obj:
        array2mesh(gl.MESH_GEN[num_slider - 1]['verts'], gl.MESH_GEN[num_slider - 1]['faces'], True)
        avatar_id = bpy.context.view_layer.objects.active.name.split('_')[-1]
        a = [i for i, d in enumerate(gl.BATCH_GEN) if f'batch_{avatar_id}' in d.keys()]
        gl.BATCH_GEN[a[0]][f'batch_{avatar_id}'] = gl.ACT_GEN[num_slider - 1]'''

'''def update_pose1(self, context):
    if (memory_slider[bpy.context.active_object]["Slider"] == "20"):  # credo che questo if non sia necessario
        memory_slider[bpy.context.active_object]["Val"] = bpy.context.window_manager.gdna_tool.gdna_pose1

    print(self.get("gdna_pose1"))


def update_pose2(self, context):
    if (memory_slider[bpy.context.active_object]["Slider"] == "40"):
        memory_slider[bpy.context.active_object]["Val"] = bpy.context.window_manager.gdna_tool.gdna_pose2
        print(memory_slider)
    # memory_slider[bpy.context.active_object] = d_slider["160"]
    print(self.get("gdna_pose2"))


def update_pose3(self, context):
    if (memory_slider[bpy.context.active_object]["Slider"] == "60"):
        memory_slider[bpy.context.active_object]["Val"] = bpy.context.window_manager.gdna_tool.gdna_pose3
    # memory_slider[bpy.context.active_object] = d_slider["320"]
    print(self.get("gdna_pose3"))

'''
def action_retrieve(avatar_id, eval_mode, obj, num_slider):
    load_tmp(avatar_id, eval_mode, obj.expname)
    array2mesh(gl.MESH_GEN[num_slider - 1]['verts'], gl.MESH_GEN[num_slider - 1]['faces'], True)

    a = [i for i, d in enumerate(gl.BATCH_GEN) if f'batch_{avatar_id}' in d.keys()]
    gl.BATCH_GEN[a[0]][f'batch_{avatar_id}'] = gl.ACT_GEN[num_slider - 1]


def abil_retrieve(mod):
    parse_eval_mode = {"Shape": "z_shape", "Details": "z_detail", "Scale": "betas", "Pose": "thetas"}
    if (not (bpy.context.active_object in dix[mod])
            or (bpy.context.active_object.select_get() == False)
            or not (check_tmp(parse_eval_mode[mod]))
            or (Last[bpy.context.active_object] == mod)
            or twrv.is_alive()):
        return False
    else:
        return True


def abil_generate(mod):
    if (bpy.context.active_object is None
            or bpy.context.active_object.select_get() == False
            or index_Gen[bpy.context.active_object][mod] == False
            or twrv.is_alive()):
        return False
    else:
        return True


def abil_generate_pose(mod):
    if ((bpy.context.active_object is None)
            or (bpy.context.active_object.select_get() == False)
            or (index_Gen[bpy.context.active_object][mod] == False
                and int(memory_slider[bpy.context.active_object]["Slider"]) >= int(
                        bpy.context.window_manager.gdna_tool.gdna_n_pose))
            or twrv.is_alive()):
        return False
    else:
        return True


def abil_slider(mod):
    if (bpy.context.active_object is None
            or bpy.context.active_object.select_get() == False
            or (Last[bpy.context.active_object] != mod)
            or len(gl.MESH_GEN) == 0):
        return False
    else:
        return True


# Property groups for UI
class GDNA_Properties(PropertyGroup):
    gdna_model: EnumProperty(
        items=[
            ('model_1', 'Model 1', 'Renderpeople weights', '', 0),
            ('model_2', 'Model 2', 'Thuman weights', '', 1)
        ],
        default='model_1'
    )

    gdna_n_pose: EnumProperty(
        items=[
            ('20', '20', '20', '', 0),
            ('40', '40', '40', '', 1),
            ('60', '60', '60', '', 2)
        ],
        default='20'
    )
    gdna_z_shape: IntProperty(name="Slider Z Shape", default=0, min=0, max=10, update=update_z_shape)
    gdna_z_details: IntProperty(name="Slider Z Details", default=0, min=0, max=10, update=update_z_details)
    gdna_scale: IntProperty(name="Slider Scale", default=0, min=0, max=40, update=update_scale)
    gdna_pose: IntProperty(name="Slider Pose", default=0, min=-20, max=20)
    gdna_pose1: IntProperty(name="Slider Pose", default=0, min=-10, max=10, update=update_pose)
    gdna_pose2: IntProperty(name="Slider Pose", default=0, min=-19, max=20, update=update_pose)
    gdna_pose3: IntProperty(name="Slider Pose", default=0, min=-29, max=30, update=update_pose)

    gdna_n_models: IntProperty(name="number models generation", default=1, min=0, max=100)
    gdna_seed: IntProperty(name="seed", default=1, min=0, max=100)


# operator
class GDNA_Start(bpy.types.Operator):
    bl_idname = "object.start"
    bl_label = "Start"
    bl_description = ("")
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        if twrv is not None and twrv.is_alive():
            return False
        else:
            return True

    def execute(self, context):
        global objR, objT, twrv
        d_model = {'model_1': 'renderpeople', 'model_2': 'thuman'}
        n_samples = bpy.context.window_manager.gdna_tool.gdna_n_models
        seed = bpy.context.window_manager.gdna_tool.gdna_seed
        model = bpy.context.window_manager.gdna_tool.gdna_model
        # print(n_samples, seed, model)
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
    bl_description = ("")
    bl_options = {'REGISTER', 'UNDO'}
    bl_objects = []  # lista degli oggetti su cui è stato effettuato un generate Shape
    bl_slider_val = {}  # valore dello slider di ogni oggetto presente
    bl_Last_OS = None  # variabile con l'ultimo oggetto selezionato

    @classmethod
    def poll(cls, context):
        # mette a 0 lo slider se non è stato ancora modificato
        if not (bpy.context.active_object in GDNA_Gen_Shape.bl_slider_val):
            bpy.context.window_manager.gdna_tool.gdna_z_shape = 0
        #####################################################################################################
        if bpy.context.active_object != GDNA_Gen_Shape.bl_Last_OS:
            GDNA_Gen_Shape.bl_Last_OS = bpy.context.active_object
            bpy.context.window_manager.gdna_tool.gdna_z_shape = GDNA_Gen_Shape.bl_slider_val[bpy.context.active_object]
            # QUIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIII
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

        # print(gl.BATCH_GEN)
        a = [i for i, d in enumerate(gl.BATCH_GEN) if f'batch_{avatar_id}' in d.keys()]
        twrv = ThreadWithReturnValue(target=obj.action_z_shape, args=(gl.BATCH_GEN[a[0]][f'batch_{avatar_id}'],))
        twrv.start()

        show_progress.n_dot = 0
        bpy.app.timers.register(partial(show_progress, bpy.context.area, twrv, obj, avatar_id))

        return {'FINISHED'}


class GDNA_Gen_Details(bpy.types.Operator):
    bl_idname = "object.gen_details"
    bl_label = "Start Details"
    bl_description = ("")
    bl_options = {'REGISTER', 'UNDO'}
    bl_objects = []
    bl_slider_val = {}
    bl_Last_OS = None

    @classmethod
    def poll(cls, context):
        if not (bpy.context.active_object in GDNA_Gen_Details.bl_slider_val):
            bpy.context.window_manager.gdna_tool.gdna_z_details = 0
        if bpy.context.active_object != GDNA_Gen_Details.bl_Last_OS:
            GDNA_Gen_Details.bl_Last_OS = bpy.context.active_object
            bpy.context.window_manager.gdna_tool.gdna_z_details = GDNA_Gen_Details.bl_slider_val[
                bpy.context.active_object]
        return abil_generate("Details")

    def execute(self, context):
        global objR, objT, twrv, last_obj
        bpy.context.window_manager.gdna_tool.gdna_z_details = 0

        if (not (bpy.context.active_object in self.bl_objects)):
            self.bl_objects.append(bpy.context.active_object)
        Last[bpy.context.active_object] = "Details"

        ob = bpy.context.active_object
        for o in list(index_Gen[ob].keys()):
            if o == "Details":
                index_Gen[ob][o] = False
            else:
                index_Gen[ob][o] = True

        last_obj = bpy.context.active_object

        avatar_id = bpy.context.view_layer.objects.active.name.split('_')[-1]
        m = [k for k, v in G.items() if f'Avatar_{avatar_id}' in v]
        obj = objR if m[0] == 'm1' else objT

        # print(gl.BATCH_GEN)
        a = [i for i, d in enumerate(gl.BATCH_GEN) if f'batch_{avatar_id}' in d.keys()]
        twrv = ThreadWithReturnValue(target=obj.action_z_detail, args=(gl.BATCH_GEN[a[0]][f'batch_{avatar_id}'],))
        twrv.start()

        show_progress.n_dot = 0
        bpy.app.timers.register(partial(show_progress, bpy.context.area, twrv, obj, avatar_id))

        return {'FINISHED'}


class GDNA_Gen_Scale(bpy.types.Operator):
    bl_idname = "object.gen_scale"
    bl_label = "Start Scale"
    bl_description = ("")
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

        # print(gl.BATCH_GEN)
        a = [i for i, d in enumerate(gl.BATCH_GEN) if f'batch_{avatar_id}' in d.keys()]
        twrv = ThreadWithReturnValue(target=obj.action_betas, args=(gl.BATCH_GEN[a[0]][f'batch_{avatar_id}'],))
        twrv.start()

        show_progress.n_dot = 0
        bpy.app.timers.register(partial(show_progress, bpy.context.area, twrv, obj, avatar_id))

        return {'FINISHED'}


class GDNA_Gen_Pose(bpy.types.Operator):
    bl_idname = "object.gen_pose"
    bl_label = "Start Pose"
    bl_description = ("")
    bl_options = {'REGISTER', 'UNDO'}
    bl_objects = []
    bl_slider_val = {}
    bl_Last_OS = None

    @classmethod
    def poll(cls, context):
        if bpy.context.active_object != GDNA_Gen_Pose.bl_Last_OS:
            GDNA_Gen_Pose.bl_Last_OS = bpy.context.active_object
            print(memory_slider)
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
                      "Slider"] == "0"):  # quando crea un nuovo oggetto il selettore di n_pose è su 40
                bpy.context.window_manager.gdna_tool.gdna_n_pose = "20"
        return abil_generate_pose("Pose")

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

        # print(gl.BATCH_GEN)
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
    bl_description = ("")
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

        return {'FINISHED'}


class GDNA_Ret_Details(bpy.types.Operator):
    bl_idname = "object.ret_details"
    bl_label = "Retrive Details"
    bl_description = ("")
    bl_options = {'REGISTER', 'UNDO'}
    bl_objects = []

    @classmethod
    def poll(cls, context):
        return abil_retrieve("Details")

    def execute(self, context):
        global objR, objT, last_obj
        Last[bpy.context.active_object] = "Details"

        ob = bpy.context.active_object
        for o in list(index_Gen[ob].keys()):
            if o == "Details":
                index_Gen[ob][o] = False
            else:
                index_Gen[ob][o] = True

        last_obj = bpy.context.active_object
        avatar_id = bpy.context.view_layer.objects.active.name.split('_')[-1]
        m = [k for k, v in G.items() if f'Avatar_{avatar_id}' in v]
        obj = objR if m[0] == 'm1' else objT

        num_slider = bpy.context.window_manager.gdna_tool.gdna_z_details
        ret_thread = ThreadWithReturnValue(target=action_retrieve, args=(avatar_id, 'z_detail', obj, num_slider,))
        ret_thread.start()

        return {'FINISHED'}


class GDNA_Ret_Scale(bpy.types.Operator):
    bl_idname = "object.ret_scale"
    bl_label = "Retrive Scale"
    bl_description = ("")
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

        return {'FINISHED'}


class GDNA_Ret_Pose(bpy.types.Operator):
    bl_idname = "object.ret_pose"
    bl_label = "Retrive Pose"
    bl_description = ("")
    bl_options = {'REGISTER', 'UNDO'}
    bl_objects = []

    @classmethod
    def poll(cls, context):
        return abil_retrieve("Pose")

    def execute(self, context):
        Last[bpy.context.active_object] = "Pose"

        ob = bpy.context.active_object
        for o in list(index_Gen[ob].keys()):
            if o == "Pose":
                index_Gen[ob][o] = False
            else:
                index_Gen[ob][o] = True
        return {'FINISHED'}


class GDNA_Reset(bpy.types.Operator):
    bl_idname = "object.reset"
    bl_label = "Total Retrive"
    bl_description = ("")
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        bpy.context.window_manager.gdna_tool.gdna_z_shape = 0
        bpy.context.window_manager.gdna_tool.gdna_z_details = 0
        bpy.context.window_manager.gdna_tool.gdna_scale = 0
        ###
        bpy.context.window_manager.gdna_tool.gdna_pose1 = 0
        bpy.context.window_manager.gdna_tool.gdna_pose2 = 0
        bpy.context.window_manager.gdna_tool.gdna_pose3 = 0
        bpy.context.window_manager.gdna_tool.gdna_n_pose = "20"

        index_Gen[bpy.context.active_object]["Shape"] = True
        index_Gen[bpy.context.active_object]["Details"] = True
        index_Gen[bpy.context.active_object]["Scale"] = True
        index_Gen[bpy.context.active_object]["Pose"] = True
        Last[bpy.context.active_object] = None
        # eliminare file di retrive per disabilitare i retrieve
        memory_slider[bpy.context.active_object] = zero_s.copy()
        return {'FINISHED'}


dix = {"Shape": GDNA_Gen_Shape.bl_objects,
       "Details": GDNA_Gen_Details.bl_objects,
       "Scale": GDNA_Gen_Scale.bl_objects,
       "Pose": GDNA_Gen_Pose.bl_objects
       }


# UI
class GDNA_PT_Model(bpy.types.Panel):
    bl_label = "GDNA Model"
    bl_category = "GDNA"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "GDNA"

    def draw(self, context):
        layout = self.layout
        layout.label(text="Model:")
        layout.prop(context.window_manager.gdna_tool, "gdna_model", expand=True)
        col = layout.column(align=True)
        row = col.row(align=True)
        split = col.split(factor=0.50, align=False)
        split.label(text="Number:")
        split.label(text="Seed:")
        # row = col.row(align=True)
        split = col.split(factor=0.5, align=False)
        split.prop(context.window_manager.gdna_tool, "gdna_n_models", text="")
        split.prop(context.window_manager.gdna_tool, "gdna_seed", text="")
        col.separator()
        col.operator("object.start", text="Create", icon="CUBE")


class GDNA_PT_Edit(bpy.types.Panel):
    bl_label = "Edit"
    bl_idname = "GDNA"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "GDNA"

    def draw(self, context):
        layout = self.layout
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

        ##Details##
        col = layout.box().column(align=True)
        split = col.split(factor=0.21, align=False)
        # Slider
        split.enabled = abil_slider("Details")
        split.label(text="Details:")
        split.prop(context.window_manager.gdna_tool, "gdna_z_details", text="", slider=False)
        # Generate
        col.separator()
        row = col.row(align=True)
        row.operator("object.gen_details", text="Generate", icon="PLUS")
        # Retrieve
        row.operator("object.ret_details", text="Retrieve", icon="RECOVER_LAST")
        col.separator()

        ##Scale##
        col = layout.box().column(align=True)
        split = col.split(factor=0.21, align=True)
        # Slider
        split.enabled = abil_slider("Scale")
        split.label(text="Scale:")
        split.prop(context.window_manager.gdna_tool, "gdna_scale", text="", slider=False)
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

        if (bpy.context.active_object == None or memory_slider[bpy.context.active_object]["Slider"] == "0"):
            split.prop(context.window_manager.gdna_tool, "gdna_pose", text="", slider=False)
        elif (memory_slider[bpy.context.active_object]["Slider"] == "20"):
            split.prop(context.window_manager.gdna_tool, "gdna_pose1", text="(20)", slider=False)
        elif (memory_slider[bpy.context.active_object]["Slider"] == "40"):
            split.prop(context.window_manager.gdna_tool, "gdna_pose2", text="(40)", slider=False)
        elif (memory_slider[bpy.context.active_object]["Slider"] == "60"):
            split.prop(context.window_manager.gdna_tool, "gdna_pose3", text="(60)", slider=False)

        # N. Pose
        col.separator()
        row = col.row(align=True)
        row.prop(context.window_manager.gdna_tool, "gdna_n_pose", expand=True)

        # Generate
        # col.separator()
        row = col.row(align=True)
        col.separator()

        row.operator("object.gen_pose", text="Generate", icon="PLUS")
        # Retrieve
        row.operator("object.ret_pose", text="Retrieve", icon="RECOVER_LAST")

        ##Reset
        layout.operator("object.reset", text="Reset", icon="RECOVER_LAST")


classes = [
    GDNA_Properties,
    GDNA_Start,
    GDNA_Gen_Shape,
    GDNA_Gen_Details,
    GDNA_Gen_Scale,
    GDNA_Gen_Pose,
    GDNA_Ret_Shape,
    GDNA_Ret_Details,
    GDNA_Ret_Scale,
    GDNA_Ret_Pose,
    GDNA_Reset,
    GDNA_PT_Model,
    GDNA_PT_Edit
]


def register():
    from bpy.utils import register_class
    for cls in classes:
        bpy.utils.register_class(cls)
    # Store properties under WindowManager (not Scene) so that they are not saved in .blend files and always show
    # default values after loading
    bpy.types.WindowManager.gdna_tool = PointerProperty(type=GDNA_Properties)


def unregister():
    from bpy.utils import unregister_class
    for cls in classes:
        bpy.utils.unregister_class(cls)

    del bpy.types.WindowManager.gdna_tool


if __name__ == "__main__":
    register()
    # obj = GDNA(max_samples=5, seed=45)
    # obj.action_sample()
    # obj.max_samples = 20
    # gl.ACT_GEN = obj.action_z_shape(gl.BATCH_GEN[0][list(gl.BATCH_GEN[0].keys())[0]])
    # gl.ACT_GEN = obj.action_z_detail(gl.BATCH_GEN[0][list(gl.BATCH_GEN[0].keys())[0]])
    # gl.ACT_GEN = obj.action_betas(gl.BATCH_GEN[0][list(gl.BATCH_GEN[0].keys())[0]])
    # gl.ACT_GEN = obj.action_thetas(gl.BATCH_GEN[0][list(gl.BATCH_GEN[0].keys())[0]])

    # For retrieving past computed mesh and features generated
    # save_tmp(0, obj.eval_mode, obj.seed, obj.expname)
    # load_tmp(0, obj.eval_mode, obj.seed, obj.expname)
