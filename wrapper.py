



class AWrap:
    def __init__(self, wrapped):
        self.wrapped=wrapped
        self.__before__={}
        self.__after__={}
    def before(self,name,func):
        self.__before__[name]=func
    def after(self,name,func):
        self.__after__[name]=func
    def __getattr__(self,name):
        before=self.__before__.get(name,None)
        after=self.__after__.get(name,None)
        method=getattr(self.wrapped, name)
        if not before and not after:
            return method

        def func(self,*args,**kwargs):
            a,k = before(self,*args,**kwargs)
            a,k=method(self,*a,**k)
            return after(self,*a,**k)
        return func

