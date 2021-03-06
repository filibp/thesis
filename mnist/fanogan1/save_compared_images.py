import os
import torch
from torchvision.utils import save_image
import pickle

def save_compared_images(opt, datastamp, generator, encoder, dataloader, device):

    print("----------  comparing stage  ----------")
    pck_real = []
    pck_fake = []
    pck_diff = []
    pck_img = []
    generator.load_state_dict(torch.load("results"+datastamp+"/generator"))
    encoder.load_state_dict(torch.load("results"+datastamp+"/encoder"))

    generator.to(device).eval()
    encoder.to(device).eval()

    os.makedirs("results"+datastamp+"/images_diff", exist_ok=True)

    for i, (img, label) in enumerate(dataloader):
        real_img = img.to(device)

        real_z = encoder(real_img)
        fake_img = generator(real_z)
        compared_images = torch.empty(real_img.shape[0] * 3,
                                      *real_img.shape[1:])
        compared_images[0::3] = real_img
        compared_images[1::3] = fake_img
        compared_images[2::3] = real_img - fake_img

        # pck_real.append(real_img)
        # pck_diff.append(real_img - fake_img)
        # pck_fake.append(fake_img)
        pck_img.append((real_img,real_img - fake_img, fake_img, label))

        save_image(compared_images.data,
                   "results"+datastamp+f"/images_diff/{opt.n_grid_lines*(i+1):06}.png",
                   nrow=3, normalize=True)

        if opt.n_iters is not None and opt.n_iters == i:
            break
    # with open("results"+datastamp+'/diff_img.pck','wb') as f:
    #     pickle.dump(pck_diff, f)
    # with open("results"+datastamp+'/real_img.pck','wb') as f:
    #     pickle.dump(pck_real, f)
    # with open("results"+datastamp+'/fake_img.pck','wb') as f:
    #     pickle.dump(pck_fake, f)
    with open("results"+datastamp+'/pck_img.pck','wb') as f:
        pickle.dump(pck_img, f)
