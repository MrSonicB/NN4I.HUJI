# -*- coding: utf-8 -*-
"""Copy of ex4_code.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/18goczX6BUQZDFGmEE7Iv37A1aIJeKS18

**Connect To Google Drive**
"""

from google.colab import drive
drive.mount('/content/gdrive/',  force_remount=True)

"""**Install Necessary Packages**"""

!pip install transformers --quiet
!pip install diffusers --quiet
!pip install accelerate --quiet

"""**Import Python Packages**"""

from PIL import Image
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import AutoencoderKL, UNet2DConditionModel, PNDMScheduler
from diffusers import UniPCMultistepScheduler
from tqdm.auto import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import torchvision.models as models
import accelerate

"""**Initialize Models And Set Global Variables**"""

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
ROOT_GDRIVE_PATH = "/content/gdrive/MyDrive/"

VAE = AutoencoderKL.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="vae")
TOKENIZER = CLIPTokenizer.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="tokenizer")
TEXT_ENCODER = CLIPTextModel.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="text_encoder")
UNET = UNet2DConditionModel.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="unet")
SCHEDULER = UniPCMultistepScheduler.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="scheduler")

for model in [VAE, TEXT_ENCODER, UNET]:
  model.eval()
  for param in model.parameters():
    param.requires_grad = False

  model.to(DEVICE)

"""**Auxiliary Functions**"""

plt.ion()
plt.figure()

# The Desired size of the output image
imsize = 512 if torch.cuda.is_available() else 128  # use small size if no GPU exists

loader = transforms.Compose([
    transforms.Resize(imsize),  # scale imported image
    transforms.ToTensor()])  # transform it into a torch tensor


def image_loader(image_name):
    image = Image.open(image_name)
    # image = image.resize((444, 444))
    # fake batch dimension required to fit network's input dimensions
    image = loader(image).unsqueeze(0)
    return image.to(DEVICE, torch.float)


def imshow(tensor, title=None):
  image = tensor.detach().clone()  # we clone the tensor to not do changes on it
  image = image.squeeze(0)      # remove the fake batch dimension
  image = transforms.ToPILImage()(image)
  plt.imshow(image)
  if title is not None:
      plt.title(title)
  plt.pause(0.001) # pause a bit so that plots are updated


def convert_latent_to_image(latent, vae_model):
    # Scale and decode the clean image latents with vae
    latent = 1 / 0.18215 * latent

    image = vae_model.decode(latent).sample
    image = (image / 2 + 0.5).clamp(0, 1)
    return image

"""**Style loss**"""

# cnn_normalization_mean and cnn_normalization_std are constant numbers that computed based on milions of images from image net.
cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406])
cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225])
cnn = models.vgg19(pretrained=True).features.eval()
cnn.eval()
cnn.to(DEVICE)


def gram_matrix(input):
    a, b, c, d = input.size()  # a=batch size(=1)
    # b=number of feature maps
    # (c,d)=dimensions of a f. map (N=c*d)

    features = input.view(a * b, c * d)  # resize F_XL into \hat F_XL

    G = torch.mm(features, features.t())  # compute the gram product

    # we 'normalize' the values of the gram matrix
    # by dividing by the number of element in each feature maps.
    return G.div(a * b * c * d)


class StyleLoss(nn.Module):

    def __init__(self, target_feature, mean_var=False):
        super(StyleLoss, self).__init__()
        self.mean_var = mean_var
        self.target = gram_matrix(target_feature).detach()

    def forward(self, input):
        G = gram_matrix(input)
        self.loss = F.mse_loss(G, self.target)
        return input


class Normalization(nn.Module):
    def __init__(self, mean, std):
        super(Normalization, self).__init__()
        # .view the mean and std to make them [C x 1 x 1] so that they can
        # directly work with image Tensor of shape [B x C x H x W].
        # B is batch size. C is number of channels. H is height and W is width.
        self.mean = torch.tensor(mean).view(-1, 1, 1).to(DEVICE)
        self.std = torch.tensor(std).view(-1, 1, 1).to(DEVICE)

    def forward(self, img):
        # normalize ``img``
        return (img - self.mean) / self.std


def get_style_model_and_losses(cnn, normalization_mean, normalization_std,
                               style_img, style_layers):
    # normalization module
    normalization = Normalization(normalization_mean, normalization_std)

    # just in order to have an iterable access to or list of content/style
    # losses
    style_losses = []
    names = []

    # assuming that ``cnn`` is a ``nn.Sequential``, so we make a new ``nn.Sequential``
    # to put in modules that are supposed to be activated sequentially
    model = nn.Sequential(normalization)
    model.to(DEVICE)

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

        if name in style_layers:
            # add style loss:
            target_feature = model(style_img).detach()
            style_loss = StyleLoss(target_feature)
            model.add_module("style_loss_{}".format(i), style_loss)
            style_losses.append(style_loss)

    # now we trim off the layers after the last content and style losses
    for i in range(len(model) - 1, -1, -1):
        if isinstance(model[i], StyleLoss):
            break

    model = model[:(i + 1)]

    return model, style_losses


def compute_style_loss(image, style_image, style_layers):
    model, style_losses = get_style_model_and_losses(cnn, cnn_normalization_mean, cnn_normalization_std, style_image, style_layers)

    # correct the values of updated input image
    with torch.no_grad():
        image.clamp_(0, 1)

    style_score = 0

    # We also put the model in evaluation mode, so that specific layers
    # such as dropout or batch normalization layers behave correctly.
    # Question: Why we needs to change to requires_grad = False
    for param in model.parameters():
      param.requires_grad = False

    model(image)

    for sl in style_losses:
        style_score += sl.loss
    #print("style_losses is {}".format(style_losses))

    return style_score

"""**Creates Text Embeddings Function**"""

def create_text_embeddings(prompt, batch_size):
  # Create conditional text embeddings
  text_input = TOKENIZER(
      prompt, padding="max_length", max_length=TOKENIZER.model_max_length, truncation=True, return_tensors="pt"
  )

  with torch.no_grad():
      text_embeddings = TEXT_ENCODER(text_input.input_ids.to(DEVICE))[0]

  # Create unconditional text embeddings
  max_length = text_input.input_ids.shape[-1]
  uncond_input = TOKENIZER([""] * batch_size, padding="max_length", max_length=max_length, return_tensors="pt")
  uncond_embeddings = TEXT_ENCODER(uncond_input.input_ids.to(DEVICE))[0]

  # Concatenate conditional and unconditional embeddings into a batch
  text_embeddings = torch.cat([uncond_embeddings, text_embeddings])

  return text_embeddings

"""**Create a random noise latent vector**"""

def generate_random_latent_vector(height, width, batch_size):
  generator = torch.manual_seed(0)  # Seed generator to create the inital latent noise

  random_latent = torch.randn(
      (batch_size, UNET.in_channels, height // 8, width // 8),
      generator=generator,
  )
  random_latent = random_latent.to(DEVICE)
  random_latent = random_latent * SCHEDULER.init_noise_sigma

  return random_latent

"""**Perform Image Diffusion**"""

def perform_image_diffusion(prompt, num_inference_steps, guidance_scale, learning_rate, target_style_image, style_layers, style_manipulation=False):
  # Set SCHEDULER's timestemps, lambda_lr and lambda_mu variables
  SCHEDULER.set_timesteps(num_inference_steps)
  lambda_lr = lambda timestep: SCHEDULER.sigma_t[timestep] * learning_rate
  lambda_mu = lambda index: 0.01 * index + 1

  # Compute text_embeddings based on the given prompt
  batch_size = len(prompt)
  text_embeddings = create_text_embeddings(prompt, batch_size)

  # Initialize random latent vector
  height = target_style_image.size()[2]
  width = target_style_image.size()[3]
  latent = generate_random_latent_vector(height, width, batch_size)

  # Main loop - Image diffusion
  for index, timestep in enumerate(tqdm(SCHEDULER.timesteps)):
    # Expand the latents if we are doing classifier-free guidance to avoid doing two forward passes.
    latent_model_input = torch.cat([latent] * 2)
    latent_model_input = SCHEDULER.scale_model_input(latent_model_input, timestep=timestep)

    # Predict the noise residual
    with torch.no_grad():
      pred_noise = UNET(latent_model_input, timestep, encoder_hidden_states=text_embeddings).sample

    # Perform guidance
    pred_noise_uncond, pred_noise_text = pred_noise.chunk(2)
    pred_noise = pred_noise_uncond + guidance_scale * (pred_noise_text - pred_noise_uncond)

    if style_manipulation:
      # Initialized style noise and SGD optimizer
      pred_noise_style = pred_noise.detach().clone()
      pred_noise_style.requires_grad = True
      optimizer = optim.SGD([pred_noise_style], lambda_lr(timestep) * lambda_mu(index))
      optimizer.zero_grad()

      # Compute the clean image's letant vector based on the current noise x_t -> x_0
      clean_latent = SCHEDULER.convert_model_output(pred_noise_style, timestep, latent)

      # Use vae model to compute the clean image based on that latent vector
      clean_image = convert_latent_to_image(clean_latent, VAE)

      # Compute the stype loss on this spesific clean image
      style_loss = compute_style_loss(clean_image, target_style_image, style_layers)

      # Print the current clean image and its style loss (for debugging purposes)
      if index % 20 == 0:
        print("index = {}, timestep = {}, style_loss = {}".format(index, timestep, style_loss))
        imshow(clean_image, title="Intermediate clean image for index = {}, timestep = {}".format(index, timestep))

      # Compute the stype loss gradient w.r.t pred_noise_style
      style_loss.backward()

      # Normalize the gradient
      with torch.no_grad():
        pred_noise_style.grad = pred_noise_style.grad / torch.mean(torch.abs(pred_noise_style.grad))

      # Use that gradient to update pred_noise_style
      optimizer.step()

      # Update pred_nosie variable
      pred_noise = pred_noise_style.detach().clone()

    else:
      if index % 20 == 0:
        # Compute the clean image's letant vector based on the current noise x_t -> x_0
        clean_latent = SCHEDULER.convert_model_output(pred_noise, timestep, latent)

        # Use vae model to compute the clean image based on that latent vector
        clean_image = convert_latent_to_image(clean_latent, VAE)

        # Compute the stype loss on this spesific clean image
        style_loss = compute_style_loss(clean_image, target_style_image, style_layers)

        # Print the current clean image and its style loss (for debugging purposes)
        print("index = {}, timestep = {}, style_loss = {}".format(index, timestep, style_loss))
        imshow(clean_image, title="Intermediate clean image for index = {}, timestep = {}".format(index, timestep))

    # Based on the predicted noise, compute the previous noisy sample x_t -> x_t-1
    latent = SCHEDULER.step(pred_noise, timestep, latent).prev_sample

  return latent

"""**Part 2 - Style Manipulation Using Denoising Diffusion**"""

# Note for tester: This code containd only the diffution parts.
# To see how we created the final images using Neural Style Transfer, please check the part1.py

# Set diffution parameters
num_inference_steps = 500  # Number of denoising steps
guidance_scale = 7.5  # Scale for classifier-free guidance
learning_rate = 0.8 # The 'S' parameter

# load target style image
target_style_image = image_loader(ROOT_GDRIVE_PATH + "seated_nude.jpeg")
target_style_image.to(DEVICE)
imshow(target_style_image, title='Target Style Image')

# Set VGG19 style layers to be used when computing style loss
style_layers = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5', 'conv_6', 'conv_7', 'conv_8', 'conv_9', 'conv_10']

# Perform Style Manipulation Using Denoising Diffusion
prompt1 = ["a man wearing a straw hat"]
final_latent1 = perform_image_diffusion(prompt1, num_inference_steps, guidance_scale, learning_rate, target_style_image, style_layers, True)

# Scale and decode the final latent with VAE model and compute it's stype loss
final_image1 = convert_latent_to_image(final_latent1, VAE)
style_loss1 = compute_style_loss(final_image1, target_style_image, style_layers)
print("index = {}, timestep = 0, style_loss = {}".format(num_inference_steps, style_loss1))
imshow(final_image1, title="Resulted Image")

# Perform ordinary text-to-image stable diffusion with the text-based style description added to the prompt
prompt2 = ["a man wearing a straw hat with similar texture to the painting named 'seated nude' by Picasso"]
final_latent2 = perform_image_diffusion(prompt2, num_inference_steps, guidance_scale, learning_rate, target_style_image, style_layers, False)

# Scale and decode the final latent with VAE model and compute it's stype loss
final_image2 = convert_latent_to_image(final_latent2, VAE)
style_loss2 = compute_style_loss(final_image2, target_style_image, style_layers)
print("index = {}, timestep = 0, style_loss = {}".format(num_inference_steps, style_loss2))
imshow(final_image2, title="Resulted Image")

# Clean cuda cashe (form memory purposes)
with torch.no_grad():
    torch.cuda.empty_cache()

