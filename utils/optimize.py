def set_smooth(obj):
    """ Enable smooth shading on an mesh object """
    for face in obj.data.polygons:
        face.use_smooth = True


def set_decimate_mod(obj, ratio):
    modifier = obj.modifiers.new('DecimateMod', 'DECIMATE')
    modifier.ratio = ratio

def removeMeshFromMemory(passedName):
    print("removeMeshFromMemory:[%s]." % passedName)
    # Extra test because this can crash Blender if not done correctly.
    result = False
    mesh = bpy.data.meshes.get(passedName)
    if mesh != None:
        if mesh.users == 0:
            try:
                mesh.user_clear()
                can_continue = True
            except:
                can_continue = False

            if can_continue == True:
                try:
                    bpy.data.meshes.remove(mesh)
                    result = True
                    print("removeMeshFromMemory: MESH [" + passedName + "] removed from memory.")
                except:
                    result = False
                    print("removeMeshFromMemory: FAILED to remove [" + passedName + "] from memory.")
            else:
                # Unable to clear users, something is holding a reference to it.
                # Can't risk removing. Favor leaving it in memory instead of risking a crash.
                print(
                    "removeMeshFromMemory: Unable to clear users for MESH, something is holding a reference to it.")
                result = False
        else:
            print(
                "removeMeshFromMemory: Unable to remove MESH because it still has [" + str(mesh.users) + "] users.")
    else:
        # We could not fetch it, it does not exist in memory, essentially removed.
        print("We could not fetch MESH [%s], it does not exist in memory, essentially removed." % passedName)
        result = True
    return result


