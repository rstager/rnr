import matplotlib.pyplot as plt
import matplotlib.animation as animation
import atexit
from os import makedirs
import os.path


class MoviePlot:
    def __init__(self,figs={1:'pyplot_movie'},dpi=100,fps=15,path=None):
        if path:
            for n,name in figs.items():
                figs[n]=os.path.join(path,name)
        self.writers=[]
        for fignum,filename in figs.items():
            fig = plt.figure(fignum)
            print("Record figure {} as {}".format(fignum,filename))
            FFMpegWriter = animation.writers['ffmpeg']
            metadata = dict(title='Movie Test', artist='Matplotlib',
                            comment='Movie support!')
            if os.path.dirname(filename)!='':
                makedirs(os.path.dirname(filename), exist_ok=True)
            writer = FFMpegWriter(fps=fps, metadata=metadata)
            writer.setup(fig, "{}.mp4".format(filename), dpi)
            self.writers.append(writer)
        def cleanup(obj):
            obj.finish()
        atexit.register(cleanup,self)

    def grab_frames(self):
        for w in self.writers:
            w.grab_frame()

    def finish(self):
        print("MoviePlot Finish")
        for w in self.writers:
            w.finish()
        # atexit will call finish, so we want to
        # make sure we clear the list of writers
        self.writers=[]