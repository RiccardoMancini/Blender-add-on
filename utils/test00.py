import os

if __name__ == "__main__":

    path = r'/home/richi/Scrivania/Blender/faces&verts/'

    # Ottieni tutti i file nella directory
    all_files = os.listdir(path)
    print(all_files)
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
                file_groups[num][file_type] = os.path.join(path, file)

    # Itera attraverso i gruppi di file e stampa le coppie
    for num, files in file_groups.items():
        if 'verts' in files and 'faces' in files:
            verts_file = files['verts']
            faces_file = files['faces']
            print(f"Verts file: {verts_file}, Faces file: {faces_file}")