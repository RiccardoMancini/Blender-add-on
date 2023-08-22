bl_info = {
    "name": "GDNA for Blender",
    "author": "Arment Pelivani, Enrico Tarsi, Riccardo Mancini",
    "version": (2, 0, 0),
    "blender": (3, 0, 1),
    "location": "Viewport > Right panel",
    "description": "GDNA for Blender",
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

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

sys.path.insert(1, f'{ROOT}/gdna')

from test import GDNA
import test as gl

# Blender path where addons is installed
# print(bpy.utils.user_resource('SCRIPTS', path='addons'))
objR, objT = None, None
G = {'m1': [], 'm2': []}


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


def array2mesh(verts, faces, replace=False):
    n_avatars = list(filter(lambda x: 'Avatar' in x.name, bpy.data.objects))
    num = len(n_avatars) if len(n_avatars) != 0 else 0
    if not replace:
        mesh = bpy.data.meshes.new(f'Avatar_{num}')
        obj = bpy.data.objects.new(f'Avatar_{num}', mesh)
        mesh.from_pydata(verts.tolist(), [], faces.tolist())
        mesh.update(calc_edges=True)  # Update mesh with new data
        bpy.context.collection.objects.link(obj)  # Link to scene
    else:
        obj = bpy.context.active_object
        num = obj.name.split('_')[-1]
        mesh = bpy.data.meshes.new(f'Avatar_{num}')
        mesh.from_pydata(verts.tolist(), [], faces.tolist())
        mesh.update(calc_edges=True)
        obj.data = mesh
        obj.data.name = f'Avatar_{num}'

    bpy.context.view_layer.objects.active = obj
    centr_mesh()


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
    objs = [ob for ob in bpy.context.scene.objects if ob.type == "MESH"]
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
            gl.ACT_GEN = process.join()
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

        return 1.0


def update_z_shape(self, context):
    print(self.get("gdna_z_shape"))


def update_z_details(self, context):
    print(self.get("gdna_z_details"))


def update_n_models(self, context):
    print(self.get("gdna_n_models"))


# Property groups for UI
class GDNA_Properties(PropertyGroup):
    gdna_model: EnumProperty(
        items=[
            ('model_1', 'Model 1', 'Renderpeople weights', '', 0),
            ('model_2', 'Model 2', 'Thuman weights', '', 1)
        ],
        default='model_1'
    )

    gdna_z_shape: IntProperty(
        name="Slider Z Shape",
        default=0,
        min=0,
        max=10,
        update=update_z_shape
    )

    gdna_z_details: IntProperty(
        name="Slider Z Details",
        default=0,
        min=0,
        max=10,
        update=update_z_details
    )

    gdna_scale: IntProperty(
        name="Slider Z Details",
        default=0,
        min=0,
        max=10,
        update=update_z_details
    )

    gdna_pose: IntProperty(
        name="Slider Z Details",
        default=0,
        min=0,
        max=10,
        update=update_z_details
    )

    gdna_shape_start: BoolProperty(
        name="Z Shape Start",
        description="Enable generative different Z shapes of GDNA model",
    )

    gdna_details_start: BoolProperty(
        name="Z Details Start",
        description="Enable generative different Z details of GDNA model",
    )

    gdna_scale_start: BoolProperty(
        name="Scale Start",
        description="Enable generative different Scale of GDNA model",
    )

    gdna_pose_start: BoolProperty(
        name="Pose Start",
        description="Enable generative different Pose of GDNA model",
    )

    gdna_n_models: IntProperty(
        name="number models generation",
        default=1,
        min=0,
        max=100
    )

    gdna_seed: IntProperty(
        name="seed",
        default=1,
        min=0,
        max=100
    )


# operator
class GDNA_Start(bpy.types.Operator):
    bl_idname = "object.start"
    bl_label = "Start"
    bl_description = ("")
    bl_options = {'REGISTER', 'UNDO'}

    # bpy.context.window_manager.gdna_tool.gdna_details_start

    def execute(self, context):
        global objR, objT
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


#
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
        split.label(text="Number of Avatars:")
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
        col = layout.column(align=True)
        row = col.row(align=True)
        split = row.split(factor=0.90, align=False)
        split.label(text="Shape:")
        split.prop(context.window_manager.gdna_tool, "gdna_shape_start", text="")

        col = layout.column(align=True)

        if (bpy.context.window_manager.gdna_tool.gdna_shape_start and bpy.context.active_object.select_get() == True):
            col.enabled = True
            col.prop(context.window_manager.gdna_tool, "gdna_z_shape", text="", slider=True)
        else:
            col.enabled = False
            col.prop(context.window_manager.gdna_tool, "gdna_z_shape", text="", slider=True)
        col.separator()

        col = layout.column(align=True)
        row = col.row(align=True)
        split = row.split(factor=0.90, align=False)
        split.label(text="Details:")
        split.prop(context.window_manager.gdna_tool, "gdna_details_start", text="")

        col = layout.column(align=True)
        if (bpy.context.window_manager.gdna_tool.gdna_details_start and bpy.context.active_object.select_get() == True):
            col.enabled = True
            col.prop(context.window_manager.gdna_tool, "gdna_z_details", text="", slider=True)
        else:
            col.enabled = False
            col.prop(context.window_manager.gdna_tool, "gdna_z_details", text="", slider=True)

        col = layout.column(align=True)
        row = col.row(align=True)
        split = row.split(factor=0.90, align=False)
        split.label(text="Scale:")
        split.prop(context.window_manager.gdna_tool, "gdna_scale_start", text="")

        col = layout.column(align=True)
        if (bpy.context.window_manager.gdna_tool.gdna_scale_start and bpy.context.active_object.select_get() == True):
            col.enabled = True
            col.prop(context.window_manager.gdna_tool, "gdna_scale", text="", slider=True)
        else:
            col.enabled = False
            col.prop(context.window_manager.gdna_tool, "gdna_scale", text="", slider=True)

        col = layout.column(align=True)
        row = col.row(align=True)
        split = row.split(factor=0.90, align=False)
        split.label(text="Pose:")
        split.prop(context.window_manager.gdna_tool, "gdna_pose_start", text="")

        col = layout.column(align=True)
        if (bpy.context.window_manager.gdna_tool.gdna_pose_start and bpy.context.active_object.select_get() == True):
            col.enabled = True
            col.prop(context.window_manager.gdna_tool, "gdna_pose", text="", slider=True)
        else:
            col.enabled = False
            col.prop(context.window_manager.gdna_tool, "gdna_pose", text="", slider=True)
        # split.operator("object.start", text = GDNA_Start.bl_label, icon = "HIDE_OFF")
        '''
        if (bpy.context.window_manager.gdna_tool.gdna_shape_start):
            split.prop(context.window_manager.gdna_tool, "gdna_z_shape", slider = True)
        split.prop(context.window_manager.gdna_tool, "gdna_shape_start", text = "Start")
        split.operator("object.start")
        '''


classes = [
    GDNA_Properties,
    GDNA_Start,
    GDNA_PT_Model,
    GDNA_PT_Edit,
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
