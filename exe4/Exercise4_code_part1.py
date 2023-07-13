import json

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from PIL import Image
import matplotlib.pyplot as plt

import torchvision.transforms as transforms
import torchvision.models as models
import matplotlib
import math
import numpy as np
matplotlib.use('TkAgg')
import copy

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
unloader = transforms.ToPILImage()  # reconvert into PIL image


def imshow(tensor, title=None):
    image = tensor.cpu().clone()  # we clone the tensor to not do changes on it
    image = image.squeeze(0)      # remove the fake batch dimension
    image = unloader(image)
    plt.imshow(image)
    if title is not None:
        plt.title(title)
    plt.pause(0.001) # pause a bit so that plots are updated


# desired size of the output image
imsize = 512 if torch.cuda.is_available() else 128  # use small size if no GPU

loader = transforms.Compose([
    transforms.Resize(imsize),  # scale imported image
    transforms.ToTensor()])  # transform it into a torch tensor

mean_var_loss = False

def image_loader(image_name):
    image = Image.open(image_name)
    image = image.resize((444, 444))  # for 2.1
    # fake batch dimension required to fit network's input dimensions
    image = loader(image).unsqueeze(0)
    return image.to(device, torch.float)


style_img = None
content_img = None

def set_style_and_content_images(section):
    global style_img
    global content_img

    if section in ['1.a']:
        style_img = image_loader("./person_straw_hat_2.jpg") # for 1.a
    else:
        style_img = image_loader("./texture.jpg")  # for 1.b and 1.c

    if section in ['1.a', '1.b', '1.c']:
        content_img = torch.randn(style_img.data.size()).to(device) # for 1.a, 1.b, 1.c

    if section in ['2.a']:
        style_img = image_loader("./picasso.jpg")  # for 2.a
        content_img = image_loader("./person_straw_hat_2_df.jpeg")  # for 2.a

    assert style_img.size() == content_img.size(), \
        "we need to import style and content images of the same size"
    plt.ion()
    plt.figure()
    imshow(style_img, title='Style Image')
    plt.figure()
    imshow(content_img, title='Content Image')


class ContentLoss(nn.Module):

    def __init__(self, target,):
        super(ContentLoss, self).__init__()
        # we 'detach' the target content from the tree used
        # to dynamically compute the gradient: this is a stated value,
        # not a variable. Otherwise the forward method of the criterion
        # will throw an error.
        self.target = target.detach()

    def forward(self, input):
        self.loss = F.mse_loss(input, self.target)
        return input

def gram_matrix(input):
    a, b, c, d = input.size()  # a=batch size(=1)
    # b=number of feature maps
    # (c,d)=dimensions of a f. map (N=c*d)

    features = input.view(a * b, c * d)  # resize F_XL into \hat F_XL

    G = torch.mm(features, features.t())  # compute the gram product

    # we 'normalize' the values of the gram matrix
    # by dividing by the number of element in each feature maps.
    return G.div(a * b * c * d)

def mean_var_matrix(input):
    a, b, c, d = input.size()  # a=batch size(=1)
    # b=number of feature maps
    # (c,d)=dimensions of a f. map (N=c*d)
    features = input.view(a * b, c * d)  # resize F_XL into \hat F_XL
    mean_var_matrix = torch.zeros(b, 2)
    for index, row in enumerate(features):
        mean_var_matrix[index][0] = torch.mean(row)
        mean_var_matrix[index][1] = torch.var(row)

    return mean_var_matrix

class StyleLoss(nn.Module):

    def __init__(self, target_feature):
        super(StyleLoss, self).__init__()
        self.mean_var = mean_var_loss
        self.target = gram_matrix(target_feature).detach() if self.mean_var is False else mean_var_matrix(target_feature).detach()

    def forward(self, input):
        G = gram_matrix(input) if self.mean_var is False else mean_var_matrix(input)
        self.loss = F.mse_loss(G, self.target)
        return input


cnn = models.vgg19(pretrained=True).features.eval()
cnn.to(device)

cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406])
cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225])

# create a module to normalize input image so we can easily put it in a
# ``nn.Sequential``
class Normalization(nn.Module):
    def __init__(self, mean, std):
        super(Normalization, self).__init__()
        # .view the mean and std to make them [C x 1 x 1] so that they can
        # directly work with image Tensor of shape [B x C x H x W].
        # B is batch size. C is number of channels. H is height and W is width.
        self.mean = torch.tensor(mean).view(-1, 1, 1).to(device)
        self.std = torch.tensor(std).view(-1, 1, 1).to(device)

    def forward(self, img):
        # normalize ``img``
        return (img - self.mean) / self.std

def get_style_model_and_losses(cnn, normalization_mean, normalization_std,
                               style_img, content_img,
                               content_layers,
                               style_layers):
    # normalization module
    normalization = Normalization(normalization_mean, normalization_std)

    # just in order to have an iterable access to or list of content/style
    # losses
    content_losses = []
    style_losses = []
    names = []

    # assuming that ``cnn`` is a ``nn.Sequential``, so we make a new ``nn.Sequential``
    # to put in modules that are supposed to be activated sequentially
    model = nn.Sequential(normalization)
    model.to(device)

    i = 0  # increment every time we see a conv
    for layer in cnn.children():
        if isinstance(layer, nn.Conv2d):
            i += 1
            name = 'conv_{}'.format(i)
        elif isinstance(layer, nn.ReLU):
            name = 'relu_{}'.format(i)
            # The in-place version doesn't play very nicely with the ``ContentLoss``
            # and ``StyleLoss`` we insert below. So we replace with out-of-place
            # ones here.
            layer = nn.ReLU(inplace=False)
        elif isinstance(layer, nn.MaxPool2d):
            name = 'pool_{}'.format(i)
        elif isinstance(layer, nn.BatchNorm2d):
            name = 'bn_{}'.format(i)
        else:
            raise RuntimeError('Unrecognized layer: {}'.format(layer.__class__.__name__))

        names.append(name)
        model.add_module(name, layer)

        if name in content_layers:
            # add content loss:
            target = model(content_img).detach()
            content_loss = ContentLoss(target)
            model.add_module("content_loss_{}".format(i), content_loss)
            content_losses.append(content_loss)

        if name in style_layers:
            # add style loss:
            target_feature = model(style_img).detach()
            style_loss = StyleLoss(target_feature)
            model.add_module("style_loss_{}".format(i), style_loss)
            style_losses.append(style_loss)

    # now we trim off the layers after the last content and style losses
    for i in range(len(model) - 1, -1, -1):
        if isinstance(model[i], ContentLoss) or isinstance(model[i], StyleLoss):
            break

    model = model[:(i + 1)]

    return model, style_losses, content_losses

def get_input_optimizer(input_img):
    # this line to show that input is a parameter that requires a gradient
    optimizer = optim.Adam([input_img])
    return optimizer


# input_img = torch.randn(content_img.data.size())
# # add the original input image to the figure:
# plt.figure()
# imshow(input_img, title='Input Image')

def get_input_optimizer(input_img, lr=0.005):
    # this line to show that input is a parameter that requires a gradient
    optimizer = optim.Adam([input_img], lr=lr)  # lr=0.005 for 1.a , lr=0.01 for 1.b
    return optimizer

def run_style_transfer(cnn, normalization_mean, normalization_std,
                       content_img, style_img, input_img, content_layers_default, style_layers_default, num_steps=2000,
                       style_weight=1000000, content_weight=1, lr=0.005, threshold=0.0001):
    """Run the style transfer."""
    print('Building the style transfer model..')
    model, style_losses, content_losses = get_style_model_and_losses(cnn,
        normalization_mean, normalization_std, style_img, content_img, content_layers_default, style_layers_default)

    # We want to optimize the input and not the model parameters so we
    # update all the requires_grad fields accordingly
    input_img.requires_grad_(True)
    # We also put the model in evaluation mode, so that specific layers
    # such as dropout or batch normalization layers behave correctly.
    model.eval()
    model.requires_grad_(False)
    # closure = None
    optimizer = get_input_optimizer(input_img, lr)

    imshow(input_img, title='Output Image')
    # sphinx_gallery_thumbnail_number = 4
    plt.ioff()
    plt.savefig('./plots/question_2_a_input_image')
    print('Optimizing..')
    counter = 0
    run = [0]
    while run[0] <= num_steps:
        def closure():
            # correct the values of updated input image
            with torch.no_grad():
                input_img.clamp_(0, 1)

            optimizer.zero_grad()
            model(input_img)
            style_score = 0
            content_score = 0

            for sl in style_losses:
                style_score += sl.loss
            for cl in content_losses:
                content_score += cl.loss

            style_score *= style_weight
            content_score *= content_weight

            loss = style_score + content_score
            loss.backward()
            closure.__setattr__('loss', loss.item())

            input_img.grad = input_img.grad / torch.mean(torch.abs(input_img.grad))
            run[0] += 1
            if run[0] % 50 == 0:
                print("run {}:".format(run))
                print('Style Loss : {:4f} Content Loss: {:4f}'.format(
                    style_score, content_score))
                print()

            return style_score + content_score

        optimizer.step(closure)

        if closure is not None and closure.loss <= threshold:
            break
        elif closure.loss <= 0.5 and lr > 0.05:
            optimizer = get_input_optimizer(input_img, lr=0.05)

    # a last correction...
    with torch.no_grad():
        input_img.clamp_(0, 1)

    return input_img


def question_one_a():
    set_style_and_content_images('1.a')
    layers = ['conv_1', 'relu_1', 'conv_2', 'relu_2', 'pool_2', 'conv_4', 'relu_4', 'pool_4', 'conv_8', 'conv_12', 'conv_16']
    # layers = ['conv_8', 'conv_12', 'conv_16']
    for layer in layers:
        # style_layers_default = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']
        style_layers_default = []
        content_layers_default = [layer]
        input_img = content_img.clone()
        # style_layers_default = ['conv_5']
        output = run_style_transfer(cnn, cnn_normalization_mean, cnn_normalization_std,
                                style_img, style_img, input_img, content_layers_default, style_layers_default)

        plt.figure()
        imshow(output, title='Output Image')
        # sphinx_gallery_thumbnail_number = 4
        plt.ioff()
        plt.savefig('./plots/question_1_a_'+layer)

    print()


def question_one_b(is_mean_var_loss=False):
    global mean_var_loss
    mean_var_loss = is_mean_var_loss
    set_style_and_content_images('1.c' if is_mean_var_loss else '1.b')
    # layers = [['conv_1'], ['conv_2'], ['conv_3'], ['conv_4'], ['conv_5'], ['conv_6'], ['conv_7'], ['conv_8'], ['conv_9'], ['conv_10'], ['conv_11'], ['conv_12'], ['conv_13'], ['conv_14'], ['conv_15'], ['conv_16']]
    # layers = [['conv_1'], ['conv_2'], ['conv_3'], ['conv_4'], ['conv_5'], ['conv_6'], ['conv_7'], ['conv_8'], ['conv_9'], ['conv_10'], ['conv_11'], ['conv_12'], ['conv_13'], ['conv_14'], ['conv_15'], ['conv_16']]
    layers = [['conv_1'], ['conv_1', 'conv_2'], ['conv_1', 'conv_2', 'conv_3'], ['conv_1', 'conv_2', 'conv_3', 'conv_4'], ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']
              , ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5', 'conv_6'], ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5', 'conv_6', 'conv_7'], ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5', 'conv_6', 'conv_7', 'conv_8']
              , ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5', 'conv_6', 'conv_7', 'conv_8', 'conv_9'], ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5', 'conv_6', 'conv_7', 'conv_8', 'conv_9', 'conv_10']
              , ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5', 'conv_6', 'conv_7', 'conv_8', 'conv_9', 'conv_10', 'conv_11'], ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5', 'conv_6', 'conv_7', 'conv_8', 'conv_9', 'conv_10', 'conv_11', 'conv_12']
              , ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5', 'conv_6', 'conv_7', 'conv_8', 'conv_9', 'conv_10', 'conv_11', 'conv_12', 'conv_13'], ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5', 'conv_6', 'conv_7', 'conv_8', 'conv_9', 'conv_10', 'conv_11', 'conv_12', 'conv_13', 'conv_14']
              , ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5', 'conv_6', 'conv_7', 'conv_8', 'conv_9', 'conv_10', 'conv_11', 'conv_12', 'conv_13', 'conv_14', 'conv_15'],
              ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5', 'conv_6', 'conv_7', 'conv_8', 'conv_9', 'conv_10', 'conv_11', 'conv_12', 'conv_13', 'conv_14', 'conv_15', 'conv_16']]
    for layer in layers:
        # style_layers_default = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']
        style_layers_default = layer
        content_layers_default = []
        input_img = content_img.clone()
        # style_layers_default = ['conv_5']
        output = run_style_transfer(cnn, cnn_normalization_mean, cnn_normalization_std,
                                    style_img, style_img, input_img, content_layers_default, style_layers_default, lr=0.005, num_steps=2000, style_weight=1, threshold=0.00000001)

        plt.figure()
        imshow(output, title='Output Image')
        # sphinx_gallery_thumbnail_number = 4
        plt.ioff()
        plt.savefig(f'./plots/question_1_{"c" if mean_var_loss else "b"}_'+('_'.join(layer)))

    print()

def question_two_a(is_mean_var_loss=False):
    global mean_var_loss
    mean_var_loss = is_mean_var_loss
    set_style_and_content_images('2.a')
    content_layers_default = ['conv_4']
    style_layers_default = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5', 'conv_6', 'conv_7', 'conv_8', 'conv_9', 'conv_10', 'conv_11', 'conv_12', 'conv_13', 'conv_14', 'conv_15', 'conv_16']
    input_img = content_img.clone()
    output = run_style_transfer(cnn, cnn_normalization_mean, cnn_normalization_std,
                            content_img, style_img, input_img, content_layers_default, style_layers_default)

    plt.figure()
    imshow(output, title='Output Image')
    # sphinx_gallery_thumbnail_number = 4
    plt.ioff()
    plt.savefig('./plots/question_2_a')




# question_one_a() # 1.a
# question_one_b() # 1.b
# question_one_b(True) # 1.c
# question_two_a() # 2.a