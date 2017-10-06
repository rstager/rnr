from keras.callbacks import Callback

import keras.backend as K
from keras.models import Model
import numpy as np

# todo: implement 2d softmax
def softmax2d(x):
    # Todo: figure out why we have to save x.shape as a tuple instead of using it directly
    # using it directly causes a keras problem

    old_shape=(-1,int(x.shape[1]),int(x.shape[2]),int(x.shape[3]))
    new_shape=(-1,1,int(x.shape[1])*int(x.shape[2]),old_shape[3])
    nsx = K.reshape(x,new_shape)
    e = K.exp(nsx - K.max(nsx, axis=-2, keepdims=True))
    s = K.sum(e, axis=-2, keepdims=True)
    ns = e / s
    ex = K.reshape(ns,old_shape)
    return ex


# returns x,y coordinates[0-1) of maximum value for each channel
# input should pass through softmax2d before input to this funciton
def softargmax(x):
    xc = K.variable(np.arange(int(x.shape[1]))/(int(x.shape[1])-1))
    x1 = K.variable(np.ones([x.shape[1]]))
    yc = K.variable(np.arange(int(x.shape[2]))/(int(x.shape[2])-1))
    y1 = K.variable(np.ones([x.shape[2]]))
    xx = K.dot(yc, x)
    xx = K.dot(x1, xx)
    xy = K.dot(y1, x)
    xy = K.dot(xc, xy)
    nc=K.stack([xx,xy],axis=-1)
    return nc


# Warning: If the critic uses BatchNormalization, then the actor must also.
def DDPGof(opt):

    class tmp(opt):
        def __init__(self, critic, actor, *args, **kwargs):
            super(tmp, self).__init__(*args, **kwargs)
            self.critic=critic
            self.actor=actor
        def get_gradients(self,loss, params):
            self.combinedloss= -self.critic([self.actor.inputs[0],self.actor.outputs[0]])
            return K.gradients(self.combinedloss,self.actor.trainable_weights)
    return tmp


def listlayers(model,indent="  "):
    print("{}Model {} {}".format(indent,model.name,("trainable" if model.trainable else "frozen")))
    for l in model.layers:
        if isinstance(l,Model):
            listlayers(l,indent=indent+"  ")
        else:
            print ("  {}layer {} {}".format(indent,l.name,("trainable" if l.trainable else "frozen")))
def freezelayers(model,names=None,thaw=False):
    for l in model.layers:
        if not names or l.name in names:
            if l.trainable != thaw:
                print("{}Freeze {}".format(("Un" if thaw else ""),l.name))
                l.trainable=thaw
            if isinstance(l, Model):
                freezelayers(l,names=None,thaw=thaw)
        else:
            if isinstance(l,Model):
                freezelayers(l,names=names,thaw=thaw)

# borrowed from https://github.com/matthiasplappert/keras-rl.git

class RandomProcess(object):
    def reset_states(self):
        pass


class AnnealedGaussianProcess(RandomProcess):
    def __init__(self, mu, sigma, sigma_min, n_steps_annealing):
        self.mu = mu
        self.sigma = sigma
        self.n_steps = 0

        if sigma_min is not None:
            self.m = -float(sigma - sigma_min) / float(n_steps_annealing)
            self.c = sigma
            self.sigma_min = sigma_min
        else:
            self.m = 0.
            self.c = sigma
            self.sigma_min = sigma

    @property
    def current_sigma(self):
        sigma = max(self.sigma_min, self.m * float(self.n_steps) + self.c)
        return sigma


class GaussianWhiteNoiseProcess(AnnealedGaussianProcess):
    def __init__(self, mu=0., sigma=1., sigma_min=None, n_steps_annealing=1000, size=1):
        super(GaussianWhiteNoiseProcess, self).__init__(mu=mu, sigma=sigma, sigma_min=sigma_min, n_steps_annealing=n_steps_annealing)
        self.size = size

    def sample(self):
        sample = np.random.normal(self.mu, self.current_sigma, self.size)
        self.n_steps += 1
        return sample

# Based on http://math.stackexchange.com/questions/1287634/implementing-ornstein-uhlenbeck-in-matlab
class OrnsteinUhlenbeckProcess(AnnealedGaussianProcess):
    def __init__(self, theta, mu=0., sigma=1., dt=1e-2, x0=None, size=1, sigma_min=None, n_steps_annealing=1000):
        super(OrnsteinUhlenbeckProcess, self).__init__(mu=mu, sigma=sigma, sigma_min=sigma_min, n_steps_annealing=n_steps_annealing)
        self.theta = theta
        self.mu = mu
        self.dt = dt
        self.x0 = x0
        self.size = size
        self.reset_states()

    def sample(self):
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + self.current_sigma * np.sqrt(self.dt) * np.random.normal(size=self.size)
        self.x_prev = x
        self.n_steps += 1
        return x

    def reset_states(self):
        self.x_prev = self.x0 if self.x0 is not None else np.zeros(self.size)


def ornstein_exploration(space,theta,nfrac=0.01,**kwrds):
    #todo: add support for unique sigma per dimension
    return OrnsteinUhlenbeckProcess(theta,size=space.shape,
                                 sigma=nfrac*np.max(space.high),sigma_min=nfrac*np.min(space.low),**kwrds)

# borrowed from jiumem https://github.com/fchollet/keras/issues/898
class LrReducer(Callback):
    def __init__(self, patience=0, reduce_rate=0.5, reduce_nb=10, verbose=1):
        super(Callback, self).__init__()
        self.patience = patience
        self.wait = 0
        self.best_score = -1.
        self.reduce_rate = reduce_rate
        self.current_reduce_nb = 0
        self.reduce_nb = reduce_nb
        self.verbose = verbose

    def on_epoch_end(self, epoch, logs={}):
        current_score = logs.get('val_acc',logs.get('episode_reward'))
        if current_score > self.best_score:
            self.best_score = current_score
            self.wait = 0
            if self.verbose > 0:
                print('---current best val accuracy: %.3f' % current_score)
        else:
            if self.wait >= self.patience:
                self.current_reduce_nb += 1
                if self.current_reduce_nb <= 10:
                    lr = self.model.optimizer.lr.get_value()
                    self.model.optimizer.lr.set_value(lr*self.reduce_rate)
                else:
                    if self.verbose > 0:
                        print("Epoch %d: early stopping" % (epoch))
                    self.model.stop_training = True
            self.wait += 1

