import numpy as np
import torch
from tqdm import tqdm
import math

import torch
import lpips

# ignore torchvision UserWarning of 'weights'
import warnings
warnings.filterwarnings("ignore")

spatial = True         # Return a spatial map of perceptual distance.

# Linearly calibrated models (LPIPS)
loss_fn = lpips.LPIPS(net='vgg', spatial=spatial) # Can also set net = 'squeeze' or 'vgg'
# loss_fn = lpips.LPIPS(net='alex', spatial=spatial, lpips=False) # Can also set net = 'squeeze' or 'vgg'

def trans(x):
    # if greyscale images add channel
    if x.shape[-3] == 1:
        x = x.repeat(1, 1, 3, 1, 1)

    # value range [0, 1] -> [-1, 1]
    x = x * 2 - 1

    return x

import open_clip

from dreamsim import dreamsim
from torchvision import transforms

def calculate_dreamsim_loss(videos1, videos2, device, only_final=False, dreamsim_model=None):
    print("calculate_dreamsim_loss...")
    
    assert videos1.shape == videos2.shape
    
    # Only load model if not provided
    if dreamsim_model is None:
        dreamsim_model, _ = dreamsim(pretrained=True, device=device)
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    
    dreamsim_loss_results = []
    
    for video_num in tqdm(range(videos1.shape[0])):
        video1 = videos1[video_num]
        video2 = videos2[video_num]
        
        dreamsim_loss_results_of_a_video = []
        for clip_timestamp in range(len(video1)):
            img1 = transforms.Resize((224, 224))(video1[clip_timestamp])
            img2 = transforms.Resize((224, 224))(video2[clip_timestamp])
            
            img1_tensor = img1.unsqueeze(0).to(device)
            img2_tensor = img2.unsqueeze(0).to(device)
            
            loss = dreamsim_model(img1_tensor, img2_tensor).item()
            
            dreamsim_loss_results_of_a_video.append(loss)
        
        dreamsim_loss_results.append(dreamsim_loss_results_of_a_video)
    
    dreamsim_loss_results = np.array(dreamsim_loss_results)
    
    dreamsim_loss = []
    dreamsim_loss_std = []
    
    if only_final:
        dreamsim_loss.append(np.mean(dreamsim_loss_results))
        dreamsim_loss_std.append(np.std(dreamsim_loss_results))
    else:
        for clip_timestamp in range(len(video1)):
            dreamsim_loss.append(np.mean(dreamsim_loss_results[:, clip_timestamp]))
            dreamsim_loss_std.append(np.std(dreamsim_loss_results[:, clip_timestamp]))
    
    result = {
        "value": dreamsim_loss,
        "value_std": dreamsim_loss_std,
    }
    
    # return mean for each video
    return result, dreamsim_loss_results.mean(axis=1)


def calculate_clip_loss(videos1, videos2, device, only_final=False, clip_model=None, clip_preprocess=None):
    print("Calculating CLIP loss with OpenCLIP's best model...")
    
    assert videos1.shape == videos2.shape, "Input videos must have the same shape."
    
    # Only load model if not provided
    if clip_model is None or clip_preprocess is None:
        model, _, preprocess = open_clip.create_model_and_transforms("ViT-bigG-14", pretrained="laion2B_s39B_b160k")
        model = model.to(device).eval()
    else:
        model = clip_model
        preprocess = clip_preprocess
    
    clip_loss_results = []
    
    for video_num in tqdm(range(videos1.shape[0])):
        video1 = videos1[video_num]
        video2 = videos2[video_num]
        
        clip_loss_results_of_a_video = []
        for clip_timestamp in range(len(video1)):
            img1 = video1[clip_timestamp].cpu()
            img2 = video2[clip_timestamp].cpu()
            
            # Convert numpy images to PIL images for preprocessing
            img1 = preprocess(transforms.ToPILImage()(torch.tensor(img1)))
            img2 = preprocess(transforms.ToPILImage()(torch.tensor(img2)))
            
            # Stack and move to device
            images = torch.stack([img1, img2]).to(device)
            
            # Compute image features
            with torch.no_grad():
                image_features = model.encode_image(images)
                image_features = torch.nn.functional.normalize(image_features, dim=-1)
            
            # Compute cosine similarity
            similarity = torch.cosine_similarity(image_features[0], image_features[1], dim=0)
            clip_loss_results_of_a_video.append(similarity.cpu().item())
        
        clip_loss_results.append(clip_loss_results_of_a_video)
    
    clip_loss_results = np.array(clip_loss_results)
    clip_loss = np.mean(clip_loss_results, axis=0) if not only_final else [np.mean(clip_loss_results)]
    clip_loss_std = np.std(clip_loss_results, axis=0) if not only_final else [np.std(clip_loss_results)]
    
    return {"value": clip_loss, "value_std": clip_loss_std}, clip_loss_results.mean(axis=1)


def calculate_lpips(videos1, videos2, device, only_final=False, lpips_model=None):
    # image should be RGB, IMPORTANT: normalized to [-1,1]
    print("calculate_lpips...")

    assert videos1.shape == videos2.shape

    # videos [batch_size, timestamps, channel, h, w]

    # support grayscale input, if grayscale -> channel*3
    # value range [0, 1] -> [-1, 1]
    videos1 = trans(videos1)
    videos2 = trans(videos2)

    lpips_results = []

    # Use provided model or global loss_fn
    if lpips_model is None:
        lpips_model = loss_fn
    
    lpips_model.to(device)

    for video_num in tqdm(range(videos1.shape[0])):
        # get a video
        # video [timestamps, channel, h, w]
        video1 = videos1[video_num]
        video2 = videos2[video_num]

        lpips_results_of_a_video = []
        for clip_timestamp in range(len(video1)):
            # get a img
            # img [timestamps[x], channel, h, w]
            # img [channel, h, w] tensor

            img1 = video1[clip_timestamp].unsqueeze(0).to(device)
            img2 = video2[clip_timestamp].unsqueeze(0).to(device)

            # calculate lpips of a video
            lpips_results_of_a_video.append(lpips_model.forward(img1, img2).mean().detach().cpu().tolist())
        lpips_results.append(lpips_results_of_a_video)
    
    lpips_results = np.array(lpips_results)
    
    lpips = []
    lpips_std = []

    if only_final:

        lpips.append(np.mean(lpips_results))
        lpips_std.append(np.std(lpips_results))

    else:

        for clip_timestamp in range(len(video1)):
            lpips.append(np.mean(lpips_results[:,clip_timestamp]))
            lpips_std.append(np.std(lpips_results[:,clip_timestamp]))

    result = {
        "value": lpips,
        "value_std": lpips_std,
    }

    return result, lpips_results.mean(axis=1)

# test code / using example

def main():
    NUMBER_OF_VIDEOS = 8
    VIDEO_LENGTH = 30
    CHANNEL = 3
    SIZE = 64
    videos1 = torch.zeros(NUMBER_OF_VIDEOS, VIDEO_LENGTH, CHANNEL, SIZE, SIZE, requires_grad=False)
    videos2 = torch.ones(NUMBER_OF_VIDEOS, VIDEO_LENGTH, CHANNEL, SIZE, SIZE, requires_grad=False)
    device = torch.device("cuda")
    # device = torch.device("cpu")

    result = calculate_lpips(videos1, videos2, device)
    print("[lpips avg]", result["value"])
    print("[lpips std]", result["value_std"])

if __name__ == "__main__":
    main()