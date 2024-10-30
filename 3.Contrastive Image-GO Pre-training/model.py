import torch.optim as optim


class SchedulerBase(object):
    def __init__(self):
        self._is_load_best_weight = True
        self._is_load_best_optim = True
        self._is_freeze_bn=False
        self._is_adjust_lr = True
        self._lr = 0.01
        self._cur_optimizer = None

    def schedule(self, net, epoch, epochs, **kwargs):
        raise Exception('Did not implemented')

    def step(self, net, epoch, epochs):
        optimizer, lr = self.schedule(net, epoch, epochs)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        lr_list = []
        for param_group in optimizer.param_groups:
            lr_list += [param_group['lr']]
        return lr_list

    def is_load_best_weight(self):
        return self._is_load_best_weight

    def is_load_best_optim(self):
        return self._is_load_best_optim

    def is_freeze_bn(self):
        return self._is_freeze_bn

    def reset(self):
        self._is_load_best_weight = True
        self._load_best_optim = True
        self._is_freeze_bn = False

    def is_adjust_lr(self):
        return self._is_adjust_lr


class Adam45(SchedulerBase):
    def __init__(self, params_list=None):
        super(Adam45, self).__init__()
        self._lr = 3e-4
        self._cur_optimizer = None
        self.params_list=params_list

    def schedule(self, net, epoch, epochs, **kwargs):
        lr = 30e-5
        if epoch > 25:
            lr = 15e-5
        if epoch > 30:
            lr = 7.5e-5
        if epoch > 35:
            lr = 3e-5
        if epoch > 40:
            lr = 1e-5
        self._lr = lr
        if self._cur_optimizer is None:
            self._cur_optimizer = optim.Adam(net.parameters(), lr=lr)#, weight_decay=0.0005

        return self._cur_optimizer, self._lr


class Adam55(SchedulerBase):
    def __init__(self, params_list=None):
        super(Adam55, self).__init__()
        self._lr = 3e-4
        self._cur_optimizer = None
        self.params_list=params_list

    def schedule(self,net, epoch, epochs, **kwargs):
        lr = 30e-5
        if epoch > 25:
            lr = 15e-5
        if epoch > 35:
            lr = 7.5e-5
        if epoch > 45:
            lr = 3e-5
        if epoch > 50:
            lr = 1e-5
        self._lr = lr
        if self._cur_optimizer is None:
            self._cur_optimizer = optim.Adam(net.parameters(), lr=lr)#, weight_decay=0.0005
        return self._cur_optimizer, self._lr

class FaceAdam(SchedulerBase):
    def __init__(self,params_list=None):
        super(FaceAdam, self).__init__()
        self._lr = 2e-4
        self._cur_optimizer = None
        self.params_list=params_list

    def schedule(self, net, epoch, epochs, **kwargs):
        lr = 1e-4
        self._lr = lr
        if self._cur_optimizer is None:
            self._cur_optimizer = optim.Adam(net.parameters(), lr=lr)  # , weight_decay=0.0005
        return self._cur_optimizer, self._lr
