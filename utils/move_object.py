import bpy

location = bpy.context.active_object.location.x
offset = 3

for obj in bpy.context.selected_objects:
    obj.location.x = location
    location += offset