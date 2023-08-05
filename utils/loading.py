import bpy
import os
from functools import partial



def show_progress(area):
    if os.path.exists('/home/richi/Scrivania/stop/'):
        area.header_text_set(None)
        print('Created!')
        return
    else:
        msg = "\r{0}{1}".format('Loading', "."*show_progress.n_dot)
        area.header_text_set(msg)
        if (show_progress.n_dot + 1) % 4 == 0:
            show_progress.n_dot = 0
        else:
            show_progress.n_dot += 1
        return 0.8

show_progress.n_dot = 0
bpy.app.timers.register(partial(show_progress, bpy.context.area))
