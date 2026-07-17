import os
import torch
import torch.nn as nn


class GpuDataParallel(object):
    def __init__(self):
        self.gpu_list = []
        self.output_device = None
        self.device_type = self._device_type()

    def set_device(self, devices):
        if isinstance(devices, list):
            self.gpu_list = [i for i in devices]
        elif isinstance(devices, int):
            self.gpu_list = [devices]
        elif isinstance(devices, str):
            self.gpu_list = list(range(torch.cuda.device_count()))
        else:
            self.gpu_list = []

        if devices is None:
            raise ValueError("Unknown device type: {}".format(devices))

        self.occupy_gpu(self.gpu_list)
        
        if len(self.gpu_list) > 0 and len(self.gpu_list) < 2:
            self.output_device = int(devices)
        elif len(self.gpu_list) > 1:
            self.output_device = int(self.gpu_list[0])
        else:
            self.output_device = "cpu"
        
    def _device_type(self):
        if torch.cuda.is_available():
            return 'cuda'
        elif torch.backends.mps.is_available():
            return 'mps'
        else:
            return 'cpu'

    def model_to_device(self, model):
        # model = convert_model(model)
        model = model.to(self.output_device)
        # if len(self.gpu_list) > 1:
        #     model = nn.DataParallel(
        #         model,
        #         device_ids=self.gpu_list,
        #         output_device=self.output_device)
        return model

    def data_to_device(self, data):
        if isinstance(data, torch.FloatTensor):
            return data.to(self.output_device)
        elif isinstance(data, torch.DoubleTensor):
            return data.float().to(self.output_device)
        elif isinstance(data, torch.ByteTensor):
            return data.long().to(self.output_device)
        elif isinstance(data, torch.LongTensor):
            return data.to(self.output_device)
        elif isinstance(data, list) or isinstance(data, tuple):
            return [self.data_to_device(d) for d in data]
        else:
            raise ValueError(data.shape, "Unknown Dtype: {}".format(data.dtype))

    def criterion_to_device(self, loss):
        return loss.to(self.output_device)

    def occupy_gpu(self, gpus=None):
        """
            make program appear on nvidia-smi.
        """
        if not gpus or (isinstance(gpus, (list, tuple)) and len(gpus) == 0):
            torch.zeros(1).cuda()
        else:
            gpus = [gpus] if isinstance(gpus, int) else list(gpus)
            for g in gpus:
                torch.zeros(1).cuda(g)

