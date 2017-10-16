import matplotlib.pyplot as plt
import numpy as np

a=np.linspace(-10,10,1000)

def smoothstep(x,edge0,edge1):
    x = np.clip((a - edge0)/(edge1 - edge0) , 0.0,1.0)
    v=  x * x * (3 - 2 * x)
    v = -20*np.power(x,7)+70*np.power(x,6)-84*np.power(x,5)+35*np.power(x,4)
    return v

def velocity_profile(a,f,t,w):
    if abs(f-t)>2*w:
        v=smoothstep(a,f,f+w) - smoothstep(a,t-w,t)
    else:
        w=(t-f)/2
        v=(smoothstep(a,f,f+w) - smoothstep(a,t-w,t))*w
    return v
