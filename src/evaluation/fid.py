from accelerate import Accelerator
from torchmetrics.image.fid import FrechetInceptionDistance
from src.data.dataloader import trainloader
import copy
from src.evaluation.multi_image_sampler import sample_batch, accelerator

fid=FrechetInceptionDistance(
    feature=2048,
    normalize=True
).to(accelerator.device)

fid.reset()

real_count=0
target_images=5000//accelerator.num_processes

trainloader=accelerator.prepare(trainloader)
for images, _ in trainloader:

    images=(images+1)/2
    images=images.clamp(0,1)

    fid.update(images, real=True)

    real_count+=images.shape[0]

    if real_count>=target_images:
        break

def fid_fn(ema_model, base_fid, alpha, alpha_bar, posterior_variance, target_images=5000):
    fake_count=0
    batch_size=64
    
    fid=copy.deepcopy(base_fid)

    target_fake_images=5000//accelerator.num_processes
    
    xyz=0
    while fake_count<target_fake_images:
    
        fake_images=sample_batch(
            ema_model,
            alpha,
            alpha_bar,
            posterior_variance,
            batch_size=batch_size
        )
    
        fake_images=(fake_images+1)/2
        fake_images=fake_images.clamp(0, 1)
    
        fid.update(fake_images, real=False)
    
        fake_count+=batch_size

        if xyz%10==0:
            print(f"fake count{fake_count* accelerator.num_processes}")
        xyz=xyz+1

    return fid.compute().item()
