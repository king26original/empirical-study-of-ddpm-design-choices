import torchvision
import torchvision.transforms as transforms
import torch

"""
The snr is hardcoded to be in -1 to 1 but if the image is in 0 to 255 then the image completely overpowers the noise which would have very less
effect. thus even at the last step xt would not be pure noise and hence the model never learns to generate from pure noise.

To solve the above issue we can either change the snr schedule or we can just normalize the input data. In the first method if the data is within
0 to 1 then the noise is dominating here, thus we would need to damp the Beta t ie instead of bt we use c*bt where c is the damping constant
proportional to variance of the data (c<1).

The mean and std are set to 0.5 to map from [0,1] to exactly [-1, 1] instead of the actual cifar10 statistic, this is because the gaussian noise
also varies from [-1, 1], this makes noise addition symmetric. at t=0 the image statistic will dominate and at t=T, noise statistic will dominate
noise is always gaussian and we require the image to also have the same statistic else the unet will have to waste some of its parameter to learn
learn this change in statistic across timesteps.
"""
transform=transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        [0.5, 0.5, 0.5],#mean
        [0.5, 0.5, 0.5]#std
    )
])

"""
data augmentation was not used here as with random crops we image might contain some useless part of the image which the model might learn
to output, though horizontal flips may be used later but vertical flips might cause harm to the model.
"""

trainset=torchvision.datasets.CIFAR10(
    root='./data',
    train=True,
    download=True,
    transform=transform
)

trainloader=torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True)
