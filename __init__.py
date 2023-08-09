import time

bl_info = {
    "name": "My First Add-on",
    "blender": (3, 6, 0),
    "category": "Object",
}

import bpy
import os
import sys
import torch
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
# print(sys.argv[1:])

from test import GDNA

# Start global var
PATH = r'/home/richi/Scrivania/Blender/tmp/'


# End global var

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


def array2mesh(verts, faces, num):
    mesh = bpy.data.meshes.new(f'AvatarMesh{num}')
    obj = bpy.data.objects.new(f'Avatar{num}', mesh)

    # Must call tolist() to pass to from_pydata()!
    mesh.from_pydata(verts.tolist(), [], faces.tolist())
    mesh.update(calc_edges=True)  # Update mesh with new data
    bpy.context.collection.objects.link(obj)  # Link to scene


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

        mesh = process.join()
        torch.cuda.empty_cache()
        for i, m in enumerate(mesh):
            array2mesh(m['verts'], m['faces'], i)
        return
    else:
        msg = "\r{0}{1}".format('Loading', "." * show_progress.n_dot)
        area.header_text_set(msg)
        if (show_progress.n_dot + 1) % 4 == 0:
            show_progress.n_dot = 0
        else:
            show_progress.n_dot += 1
        return 0.8


class TestClass(bpy.types.Operator):
    """My Object First Script"""  # Use this as a tooltip for menu items and buttons.
    bl_idname = "object.test_ao"  # Unique identifier for buttons and menu items to reference.
    bl_label = "Test add-on"  # Display name in the interface.
    bl_options = {'REGISTER', 'UNDO'}  # Enable undo for the operator.

    def execute(self, context):
        # Blender path where addons is installed
        # print(bpy.utils.user_resource('SCRIPTS', path='addons'))

        obj = GDNA(seed=50)
        twrv = ThreadWithReturnValue(target=obj.action_sample)
        twrv.start()

        show_progress.n_dot = 0
        bpy.app.timers.register(partial(show_progress, bpy.context.area, twrv))

        # for j, b in enumerate(bones):
        #    array2bones(b['joints'][0], j)'''

        return {'FINISHED'}


def menu_func(self, context):
    self.layout.operator(TestClass.bl_idname)


def register():
    bpy.utils.register_class(TestClass)
    bpy.types.VIEW3D_MT_object.append(menu_func)  # Adds the new operator to an existing menu.


def unregister():
    bpy.utils.unregister_class(TestClass)


if __name__ == "__main__":
    register()

    # obj = GDNA(seed=20)
    # obj.action_sample()
    # obj.action_z_shape()
    # obj.action_z_detail()
    # obj.action_betas()
    # obj.action_thetas()

    '''_, bones = obj.action_sample()
    for j, b in enumerate(bones):
        array2bones(b['joints'][0], j)'''
