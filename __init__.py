#import bpy
import numpy as np
import glob
import os
import sys
from pathlib import Path
import hydra

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

sys.path.insert(1, f'{ROOT}/gdna')

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


if __name__ == "__main__":
    # Blender path where addons is installed
    # bpy.utils.user_resource('SCRIPTS', path='addons')

    #array2mesh()
    obj = GDNA()
    obj.action_sample()