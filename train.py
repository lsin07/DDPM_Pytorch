from pathlib import Path
from torch.optim import Adam
import matplotlib.pyplot as plt

from ddpm_pytorch.network import *
from ddpm_pytorch.forward_diffusion import *
from ddpm_pytorch.reversed_diffusion import *
from torchvision.utils import save_image

def num_to_groups(num, divisor):
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr

results_folder = Path("./results")
results_folder.mkdir(exist_ok = True)
save_and_sample_every = 1000

device = "cuda" if torch.cuda.is_available() else "cpu"

model = Unet(
    dim=image_size,
    channels=channels,
    dim_mults=(1, 2, 4,)
)
model.to(device)

optimizer = Adam(model.parameters(), lr=1e-3)

epochs = 10

for epoch in range(epochs):
    for step, batch in enumerate(dataloader):
        optimizer.zero_grad()

        batch_size = batch["pixel_values"].shape[0]
        batch = batch["pixel_values"].to(device)

        # Algorithm 1 line 3: sample t uniformally for every example in the batch
        t = torch.randint(0, timesteps, (batch_size,), device=device).long()

        loss = p_losses(model, batch, t, loss_type="huber")

        print(step)
        if step % 100 == 0:
            print("Loss:", loss.item())

        loss.backward()
        optimizer.step()

        # save generated images
        # 작동 안됨, 이유 모르겠음 ㅅㅂ
        # if step % save_and_sample_every == 0: # step != 0 and
        #     milestone = step // save_and_sample_every
        #     batches = num_to_groups(4, batch_size)
        #     all_images_list = list(map(lambda n: sample(model, image_size, batch_size=n, channels=channels), batches))
        #     all_images = torch.cat(all_images_list, dim=0)
        #     all_images = (all_images + 1) * 0.5
        #     save_image(all_images, str(results_folder / f'sample-{milestone}.png'), nrow = 6)

samples = sample(model, image_size=image_size, batch_size=64, channels=channels)
random_index = 5
plt.imshow(samples[-1][random_index].reshape(image_size, image_size, channels), cmap="gray")
plt.show()