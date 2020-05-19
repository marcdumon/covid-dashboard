# --------------------------------------------------------------------------------------------------------
# 2020/01/07
# src - filter_visualisation.py
# md
# --------------------------------------------------------------------------------------------------------

import PIL
import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch as th
from my_tools.pytorch_tools import summary, set_requires_grad
from torchvision import transforms
from torchvision.models import vgg16_bn


class SaveFeatures:
    def __init__(self, module):
        self.hook = module.register_forward_hook(self.hook_fn)

    def hook_fn(self, module, input, output):
        self.features = output

    def close(self):
        self.hook.remove()


class FilterVisualizer:
    def __init__(self):
        vgg = vgg16_bn(pretrained=True)
        summary(vgg.cuda(), (3, 224, 224))
        self.model = vgg.cuda().eval()
        set_requires_grad(self.model, 'all', False)

    def visualize(self, sz, layer, filter, upscaling_steps=12, upscaling_factor=1.2,
                  lr=0.1, opt_steps=20, blur=None, save=False, print_losses=False):
        # img = (np.random.random((sz, sz, 3)) * 20 + 128.) / 255.
        img = np.uint8(np.random.uniform(50, 250, (sz, sz, 3))) / 255
        activations = SaveFeatures(list(self.model.children())[0][layer])  # register hook
        val_tfms = transforms.Compose([transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                       ])
        mu, sigma = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
        denorm = lambda x: x * sigma + mu

        for i in range(upscaling_steps):  # scale the image up upscaling_steps times

            img_var = val_tfms(img)
            img_var = img_var[np.newaxis, :, :, :]
            img_var = img_var.float().to('cuda')
            img_var.requires_grad = True

            optimizer = th.optim.Adam([img_var], lr=lr, weight_decay=1e-6)

            if i > upscaling_steps / 2:
                opt_steps_ = int(opt_steps * 1.3)
            else:
                opt_steps_ = opt_steps

            for n in range(opt_steps_):  # optimize pixel values for opt_steps times
                optimizer.zero_grad()
                self.model(img_var)
                loss = -1 * activations.features[0, filter].mean()
                # [80, 155, 199, 284]
                # loss = -1 * activations.features[0, 80].mean() \
                #        -1 * activations.features[0, 155].mean()\
                #        -1 * activations.features[0, 199].mean()\
                #        -1 * activations.features[0, 284].mean()
                if print_losses:
                    if i % 3 == 0 and n % 5 == 0:
                        print(f'{i} - {n} - {float(loss)}')
                loss.backward()
                optimizer.step()

            img = denorm(img_var.data.cpu().numpy()[0].transpose(1, 2, 0))
            self.output = img
            sz = int(upscaling_factor * sz)  # calculate new image size
            img = cv2.resize(img, (sz, sz), interpolation=cv2.INTER_CUBIC)  # scale image up
            if blur is not None: img = cv2.blur(img, (blur, blur))  # blur image to reduce high frequency patterns
        activations.close()
        return np.clip(self.output, 0, 1)

    # def get_transformed_img(self, img, sz):
    #     # train_tfms, val_tfms = tfms_from_model(resnet34, sz)
    #     val_tfms = transforms.Compose([transforms.ToTensor(),
    #                                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    #     # return val_tfms.denorm(np.rollaxis(to_np(val_tfms(img)[None]), 1, 4))[0]
    #     denorm = DeNormalize([0.1307], [0.3082])
    #     return denorm(np.rollaxis(to_np(val_tfms(img)), 1, 4))[0]

    def most_activated(self, image, layer, limit_top=None):

        # train_tfms, val_tfms = tfms_from_model(resnet34, 224)
        val_tfms = transforms.Compose([transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        transformed = val_tfms(image)
        transformed.requires_grad = True
        transformed = transformed.cuda()
        activations = SaveFeatures(list(self.model.children())[0][layer])  # register hook
        self.model(transformed[None])
        print(activations.features.shape[1])
        print(activations.features[0, 127].mean().data.cpu().numpy())
        mean_act = [activations.features[0, i].mean().data.cpu().numpy() for i in range(activations.features.shape[1])]
        activations.close()
        return mean_act


all_layers = []

if __name__ == '__main__':

    layer = 42
    filter = [80, 155, 199, 284]
    total_filters_in_layer = 512

    FV = FilterVisualizer()
    picture = PIL.Image.open('church.jpg')
    mean_act = FV.most_activated(picture, layer)
    plt.figure(figsize=(7, 5))
    act = plt.plot(mean_act, linewidth=2.)
    extraticks = filter
    ax = act[0].axes
    ax.set_xlim(0, 500)
    # plt.axvline(x=filter, color='grey', linestyle='--')
    ax.set_xlabel(" feature map")
    ax.set_ylabel("mean activation")
    ax.set_xticks([0, 200, 400] + extraticks)
    plt.show()
    thresh = 0.2
    filter = [i for i in range(total_filters_in_layer) if mean_act[i] > thresh]
    print(filter)
    # sleep(20)

    for f in filter:
        x = FV.visualize(56, layer, f, blur=5, upscaling_steps=12, upscaling_factor=1.2, )
        plt.imshow(x[:, :, :])
        plt.show()
