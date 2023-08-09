bl_info = {
    "name": "My First Add-on",
    "blender": (3, 6, 0),
    "category": "Object",
}

import bpy
import numpy as np
import os
import sys
from pathlib import Path

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

def array2mesh():
    # Ottieni tutti i file nella directory
    all_files = os.listdir(PATH)

    # Raggruppa i file con lo stesso numero (utilizzando il prefisso come chiave)
    file_groups = {}
    for file in all_files:
        prefix, ext = os.path.splitext(file)
        if ext == '.npz':
            num = prefix[-1]
            file_type = prefix[:-1]
            if file_type in ['verts', 'faces']:
                if num not in file_groups:
                    file_groups[num] = {}
                file_groups[num][file_type] = os.path.join(PATH, file)

    # Itera attraverso i gruppi di file e stampa le coppie
    for num, files in file_groups.items():
        if 'verts' in files and 'faces' in files:
            verts_file = files['verts']
            faces_file = files['faces']
            # print(f"Verts file: {verts_file}, Faces file: {faces_file}")

            v_data = np.load(verts_file)
            f_data = np.load(faces_file)

            verts = v_data['arr_0']
            faces = f_data['arr_0']

            '''verts = np.array([[-0.2210070639848709, -0.33245623111724854, 0.16561070084571838], [-0.22099994122982025, -0.3324548304080963, 0.1654796153306961], [-0.22059719264507294, -0.33251500129699707, 0.16561555862426758]], dtype=np.float32)

            faces =  np.array([[0, 1, 2], [141882, 142757, 141885], [142755, 141884, 142758]],
                             dtype=int)'''

            mesh = bpy.data.meshes.new(f'AvatarMesh{num}')
            obj = bpy.data.objects.new(f'Avatar{num}', mesh)

            # Must call tolist() to pass to from_pydata()!
            mesh.from_pydata(verts.tolist(), [], faces.tolist())
            mesh.update(calc_edges=True)  # Update mesh with new data
            bpy.context.collection.objects.link(obj)  # Link to scene


class TestClass(bpy.types.Operator):
    """My Object Moving Script"""      # Use this as a tooltip for menu items and buttons.
    bl_idname = "object.test_ao"        # Unique identifier for buttons and menu items to reference.
    bl_label = "Test add-on"         # Display name in the interface.
    bl_options = {'REGISTER', 'UNDO'}  # Enable undo for the operator.

    def execute(self, context):
        # Blender path where addons is installed
        # print(bpy.utils.user_resource('SCRIPTS', path='addons'))
        obj = GDNA(seed=50)
        obj.action_sample()

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

    # obj = GDNA(seed=20, expname='thuman')
    # obj.action_sample()
    # obj.action_z_shape()
    # obj.action_z_detail()
    # obj.action_betas()
    # obj.action_thetas()
