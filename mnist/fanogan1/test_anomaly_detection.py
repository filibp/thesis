import torch
import torch.nn as nn
from torch.utils.model_zoo import tqdm


def test_anomaly_detection(opt, datastamp, generator, discriminator, encoder,
                           dataloader, device, kappa=1.0):


    print("----------  anomaly detection stage  ----------")


    generator.load_state_dict(torch.load("results"+datastamp+"/generator"))
    discriminator.load_state_dict(torch.load("results"+datastamp+"/discriminator"))
    encoder.load_state_dict(torch.load("results"+datastamp+"/encoder"))

    generator.to(device).eval()
    discriminator.to(device).eval()
    encoder.to(device).eval()
    criterion = nn.MSELoss()

    with open("results"+datastamp+"/score.csv", "w") as f:
        f.write("label,img_distance,anomaly_score,z_distance,loss_feature\n")

    for (img, label) in dataloader:
        real_img = img.to(device)

        real_z = encoder(real_img)
        fake_img = generator(real_z)
        fake_z = encoder(fake_img)
        real_feature = discriminator.forward_features(real_img)
        fake_feature = discriminator.forward_features(fake_img)
        # Scores for anomaly detection
        img_distance = criterion(fake_img, real_img)
        loss_feature = criterion(fake_feature, real_feature)
        anomaly_score = img_distance + kappa * loss_feature
        z_distance = criterion(fake_z, real_z)
        with open("results"+datastamp+"/score.csv", "a") as f:
            f.write(f"{label.item()},{img_distance},"
                    f"{anomaly_score},{z_distance},{loss_feature}\n")
