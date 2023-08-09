from pytorch_grad_cam import GradCAM, HiResCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
from torchvision.models import resnet50
from torchvision import transforms
import torch
from PIL import Image
import cv2
import numpy as np
from pytorch_grad_cam import GuidedBackpropReLUModel
from pytorch_grad_cam.utils.image import show_cam_on_image, \
    deprocess_image, \
    preprocess_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
import argparse
from config import cfg
import torchreid
from  model import make_model
from torchvision.transforms import functional as TF

from vit_grad_rollout import VITAttentionGradRollout
import timm

from vit_rollout import VITAttentionRollout

def show_mask_on_image(img, mask):
    img = np.float32(img) / 155
    heatmap = cv2.applyColorMap(np.uint8(200 * mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    cam = heatmap + np.float32(img)
    cam = cam / np.max(cam)
    return np.uint8(255 * cam)

def show_mask(mask):
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    #heatmap = heatmap / np.max(heatmap)
    return np.uint8(heatmap)


def vit_cam2(opt, image_path):


    # print('using VIT to extract features'
    discard_ratio = 0.8
    category_index = None

    device = "cpu"
    model = make_model(cfg, num_class=1041, camera_num=15, view_num =15)
    model.load_param(cfg.TEST.WEIGHT)
    model.to(device)
    model.eval()


    rgb_img = cv2.imread(image_path, 1)
    #osize = rgb_img.shape[:2]
    #print(osize)
   
    rgb_img = cv2.resize(rgb_img, (128, 256))
    #rgb_img = np.float32(rgb_img) / 255
    
    rgb＿tf = TF.to_tensor(rgb_img)

    print('rgb_tf:' ,rgb_tf.shape)


    input_tensor = TF.normalize(rgb_tf, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    input_tensor = input_tensor.unsqueeze(0)
    print(input_tensor.shape)
   
    if opt.grad:
        print("Doing Gradient Attention Rollout")
        grad_rollout = VITAttentionGradRollout(model, discard_ratio= discard_ratio)
        mask = grad_rollout(input_tensor, category_index)
    else:
        attention_rollout = VITAttentionRollout(model, head_fusion="mean", 
            discard_ratio= discard_ratio)
        mask = attention_rollout(input_tensor)

   
   
    mask = cv2.resize(mask, (128, 256))

    #print('image:', rgb_tf.shape) 

    #rgb_tf = rgb_tf.permute(1,2,0)
    #print('image permute:', rgb_tf.shape) 

   # np_img = np.array(rgb_tf)
    np_img = rgb_img
    print(np_img.shape)
    mask_save = show_mask(mask)
    mask = show_mask_on_image(np_img, mask)
    
    #cv2.imwrite("./vis_result/transreid_mask2.png", mask_save)
    cv2.imwrite(opt.o, mask)



def reshape_transform(tensor, height=14, width=14):
    result = tensor[:, 1:, :].reshape(tensor.size(0),
                                      height, width, tensor.size(2))

    # Bring the channels to the first dimension,
    # like in CNNs.
    result = result.transpose(2, 3).transpose(1, 2)
    return result

def vit_cam(opt, image_path):
    img = cv2.imread(image_path, 1)
    osize = img.shape[:2]
    img = np.float32(img) / 255
    print('size:', osize)
    rgb_img = cv2.resize(img, (224, 224))
    input_tensor = preprocess_image(rgb_img, mean=[0.5, 0.5, 0.5],
                                    std=[0.5, 0.5, 0.5])


    # If None, returns the map for the highest scoring category.
    # Otherwise, targets the requested category.


    targets = 0 #[ClassifierOutputTarget(770)]
    model = torch.hub.load('facebookresearch/deit:main',
                           'deit_base_patch16_224', pretrained=True)
    model.eval()

    target_layers = [model.blocks[-1].norm1]
    
    # AblationCAM and ScoreCAM have batched implementations.
    # You can override the internal batch size for faster computation.
    cam =  GradCAMPlusPlus(model=model,
                        target_layers=target_layers,
                        use_cuda=False,
                        reshape_transform=reshape_transform)
    
    grayscale_cam = cam(input_tensor=input_tensor,
                        targets=targets,
                        eigen_smooth=False,
                        aug_smooth=False)
    
    grayscale_cam = grayscale_cam[0, :]
    
    grayscale_cam = cv2.resize(grayscale_cam, (osize[1], osize[0]))

    cam_image = show_cam_on_image(img, grayscale_cam, use_rgb=True)
    cam_image = cv2.cvtColor(cam_image, cv2.COLOR_RGB2BGR)

    cv2.imwrite("./cmresult_Vit2.png", cam_image)







def OSnet_cam(image_path, opt):
    #resnet on imageNet
    #model = resnet50(pretrained=True)
    # OSnet for reid
    print('using OSnet to extract features')
    # model = torchreid.models.build_model(
    #     name="resnet50",
    #     num_classes= 0,
    #     loss="softmax",
    #     pretrained=True
    # )
    model =  torchreid.models.build_model('osnet_x1_0', 1)
    torchreid.utils.load_pretrained_weights(model, '../weights/osnet_x1_0_MS_D_C.pth')
    model = model.to('cpu')
    print(model)
    model.eval()
    #print(model)
    target_layers = [model.conv5]
    #resnet
    rgb_img = cv2.imread(image_path, 1)[:, :, ::-1]
    rgb_img = np.float32(rgb_img) / 255
    input_tensor = preprocess_image(rgb_img,
                                    mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])# Create an input tensor image for your model..
    
    cam = GradCAMPlusPlus(model=model, target_layers=target_layers, use_cuda=False)
    targets = None #shirt
    grayscale_cam = cam(input_tensor=input_tensor, targets=targets,eigen_smooth=False,
                        aug_smooth=False)
        # In this example grayscale_cam has only one image in the batch:
    grayscale_cam = grayscale_cam[0, :]
    cam_image = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
    cam_image = cv2.cvtColor(cam_image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(opt.o, cam_image)



if __name__ == '__main__':


    image_path = './fc2.png'

    
    parser = argparse.ArgumentParser()
    parser.add_argument('-cnn', default=False, help='use cnn')
    parser.add_argument('-vit', default=False, help='use transformer')
    parser.add_argument('-grad', default=False, help='grad rollout')
    parser.add_argument(
        "--o", default="", help="output name", type=str)
    parser.add_argument(
        "--c", default="", help="path to config file", type=str)
    
    opt = parser.parse_args()


    if opt.c != "":
        cfg.merge_from_file(opt.c)
        cfg.freeze()

    if opt.cnn:
        OSnet_cam(image_path, opt)

    if opt.vit:
        vit_cam2(opt, image_path)

