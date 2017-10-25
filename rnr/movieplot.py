import matplotlib.pyplot as plt
import matplotlib.animation as animation
import atexit
from os import makedirs
import os.path


class MoviePlot:
    def __init__(self,figs='pyplot_movie',dpi=100,fps=15,path=None):
        self.figs=figs
        self.path=path
        self.dpi=dpi
        self.fps=fps
        if isinstance(path,dict):
            for n,name in figs.items():
                figs[n]=os.path.join(path,name)
        self.writers={}
        if isinstance(figs,dict):
            for fignum,filename in figs.items():
                self._add_fig(fignum,filename)
        def cleanup(obj):
            obj.finish()
        atexit.register(cleanup,self)

    def _add_fig(self,fignum,filename):
        if filename is None:
            filename=self.figs+'_'+str(fignum)
            if self.path:
                filename = os.path.join(self.path, filename)
        fig = plt.figure(fignum)
        print("Record figure {} as {}".format(fignum, filename))
        FFMpegWriter = animation.writers['ffmpeg']
        metadata = dict(title='Pyplot movie', artist='Matplotlib',
                        comment='')
        if os.path.dirname(filename) != '':
            makedirs(os.path.dirname(filename), exist_ok=True)
        writer = FFMpegWriter(fps=self.fps, metadata=metadata)
        writer.setup(fig, "{}.mp4".format(filename), self.dpi)
        self.writers[fignum] = writer

    def grab_frames(self):
        for w in self.writers.values():
            w.grab_frame()

    def finish(self):
        print("MoviePlot Finish")
        for w in self.writers.values():
            w.finish()
        # atexit will call finish, so we want to
        # make sure we clear the list of writers
        if hasattr(self,'plt'):
            self.plt.pause = self.save_pause
        self.writers={}

    def grab_on_pause(self,plt):
        from types import MethodType
        self.plt=plt
        self.save_pause=plt.pause
        orig_pause=self.save_pause
        movieplotter=self
        def pausewrapper(wrapped_self, *args, **kwargs):
            fig=wrapped_self.gcf()
            if not fig.number in movieplotter.writers and isinstance(movieplotter.figs,str):
                movieplotter._add_fig(fig.number,None)
            if fig.number in movieplotter.writers:
                movieplotter.writers[fig.number].grab_frame()
            orig_pause(*args, **kwargs)
        plt.pause = MethodType(pausewrapper,plt)