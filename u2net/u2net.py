import os

import numpy as np
import torch
from skimage.transform import resize

from .models import U2NET
from .models import U2NETP


class u2net(object):
    U2NET_MODEL_DIR = os.path.abspath(os.path.expanduser('~/.u2net'))

    def __init__(self, model_name='u2net', model_dir=None):
        """
        Creates a u2net model using either u2net or u2netp (smaller) architectures.
        By default it will use the pretrained network, stored in '~/.u2net'.
        :param model_name: {u2net, u2netp}
        :param model_dir: Specify a different directory to use a custom model.
        """
        if model_name == 'u2net':
            self.model = U2NET(3, 1)
        elif model_name == 'u2netp':
            self.model = U2NETP(3, 1)
        elif model_name == 'people':
            self.model = U2NET(3, 1)
            model_name = 'u2net_human_seg'
        else:
            print("Model {} not found, using default (u2net)".format(model_name))
            model_name = 'u2net'
            self.model = U2NET(3, 1)

        if not model_dir:
            model_dir = os.path.join(u2net.U2NET_MODEL_DIR, model_name + '.pth')
        else:
            model_dir = os.path.join(model_dir, model_name + '.pth')

        if torch.cuda.is_available():
            self.model.load_state_dict(torch.load(model_dir))
            self.model.cuda()
        else:
            self.model.load_state_dict(torch.load(model_dir, map_location='cpu'))
        self.model.eval()

    def predict(self, image, do_resize=True, prediction_size=320):
        """
        Predict segmentation mask from an image.
        :param image: An image that has already been loaded (normalized and RGB format).
        :param do_resize: If true, resizes the input image to perform the segmentation (faster).
        :param prediction_size: If resizing is enabled, selects the size used for the prediction.
        :return: Segmentation mask of the salient object(s). It will be of the same size as input, even if resizing is enabled for the prediction.
        """
        if do_resize:
            h, w = image.shape[:2]

            if h < w:
                new_h, new_w = prediction_size * h // w, prediction_size
            else:
                new_h, new_w = prediction_size, prediction_size * w // h

            image = resize(image, (new_h, new_w), anti_aliasing = True)

        # normalize
        image = image / np.max(image)
        image[:, :, 0] = (image[:, :, 0] - 0.485) / 0.229
        image[:, :, 1] = (image[:, :, 1] - 0.456) / 0.224
        image[:, :, 2] = (image[:, :, 2] - 0.406) / 0.225
        img_tensor = torch.from_numpy(image.transpose((2, 0, 1))).unsqueeze(0).float()

        if torch.cuda.is_available():
            img_tensor.cuda()

        d1, d2, d3, d4, d5, d6, d7 = self.model(img_tensor)

        pred = d1[:, 0, :, :]
        pred = (pred-torch.min(pred))/(torch.max(pred)-torch.min(pred))
        mask = pred.detach().numpy().transpose((1, 2, 0))[..., 0]
        del d1, d2, d3, d4, d5, d6, d7

        if do_resize:
            mask = resize(mask, (h, w), anti_aliasing=True)
        return mask
