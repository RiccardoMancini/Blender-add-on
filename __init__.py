bl_info = {
    "name": "My First Add-on",
    "blender": (3, 6, 0),
    "category": "Object",
}

#import bpy
import os
import sys
import torch

import numpy as np
from pathlib import Path
from functools import partial
from threading import Thread

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

sys.argv[1:] = ['expname=renderpeople', '+experiments=fine', f'+r_path={ROOT}']
sys.path.insert(1, f'{ROOT}/gdna')

from test import GDNA
import test as gl


# Blender path where addons is installed
# print(bpy.utils.user_resource('SCRIPTS', path='addons'))

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


def save_tmp(avatar_id, eval_mode, seed, expname):
    np.savez(f'{eval_mode}_{seed}_{expname}.npz', *gl.MESH_GEN)
    torch.save(gl.ACT_GEN, f'{eval_mode}_{seed}_{expname}.pt')


def load_tmp(avatar_id, eval_mode, seed, expname):
    # TODO: 1) check se esiste 2) sistemare le directory
    loaded_data = np.load(f'{eval_mode}_{seed}_{expname}.npz', allow_pickle=True)
    gl.MESH_GEN = [loaded_data[f"arr_{i}"][()] for i in range(len(loaded_data.files))]

    gl.ACT_GEN = torch.load(f'{eval_mode}_{seed}_{expname}.pt')


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


def show_progress(area, process):
    if not process.is_alive():
        area.header_text_set(None)
        print('Created!')
        if 'sample' in process._target.__name__:
            process.join()
            torch.cuda.empty_cache()
        else:
            gl.ACT_GEN = process.join()
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


'''class TestClass1(bpy.types.Operator):
    """My Object First Script"""  # Use this as a tooltip for menu items and buttons.
    bl_idname = "object.test_ao1"  # Unique identifier for buttons and menu items to reference.
    bl_label = "Test 1 add-on"  # Display name in the interface.
    bl_options = {'REGISTER', 'UNDO'}  # Enable undo for the operator.

    def execute(self, context):
        obj = GDNA(seed=20, max_samples=5)
        twrv = ThreadWithReturnValue(target=obj.action_sample)
        twrv.start()

        show_progress.n_dot = 0
        bpy.app.timers.register(partial(show_progress, bpy.context.area, twrv))

        generate_mesh.n_mesh_gen = 0
        bpy.app.timers.register(partial(generate_mesh, twrv))

        # _, bones = obj.action_sample()
        # for j, b in enumerate(bones):
        #    array2bones(b['joints'][0], j)

        return {'FINISHED'}


class TestClass2(bpy.types.Operator):
    """My Object Second Script"""  # Use this as a tooltip for menu items and buttons.
    bl_idname = "object.test_ao2"  # Unique identifier for buttons and menu items to reference.
    bl_label = "Test 2 add-on"  # Display name in the interface.
    bl_options = {'REGISTER', 'UNDO'}  # Enable undo for the operator.

    def execute(self, context):
        array2mesh(gl.MESH_GEN[0]['verts'], gl.MESH_GEN[0]['faces'], replace=True)

        return {'FINISHED'}


classes = (
    TestClass1,
    TestClass2
)'''


def menu_func(self, context):
    for cls in classes:
        self.layout.operator(cls.bl_idname)


def register():
    from bpy.utils import register_class
    for cls in classes:
        register_class(cls)
    bpy.types.VIEW3D_MT_object.append(menu_func)  # Adds the new operator to an existing menu.


def unregister():
    from bpy.utils import unregister_class
    for cls in reversed(classes):
        unregister_class(cls)


if __name__ == "__main__":
    # register()
    obj = GDNA(max_samples=5, seed=45)
    obj.action_sample()
    # obj.max_samples = 20
    # gl.ACT_GEN = obj.action_z_shape(gl.BATCH_GEN[0][list(gl.BATCH_GEN[0].keys())[0]])
    # gl.ACT_GEN = obj.action_z_detail(gl.BATCH_GEN[0][list(gl.BATCH_GEN[0].keys())[0]])
    # gl.ACT_GEN = obj.action_betas(gl.BATCH_GEN[0][list(gl.BATCH_GEN[0].keys())[0]])
    # gl.ACT_GEN = obj.action_thetas(gl.BATCH_GEN[0][list(gl.BATCH_GEN[0].keys())[0]])

    # For retrieving past computed mesh and features generated
    # save_tmp(0, obj.eval_mode, obj.seed, obj.expname)
    # load_tmp(0, obj.eval_mode, obj.seed, obj.expname)
