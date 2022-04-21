import fanogan1.save_compared_images as s
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

from fanogan1.train_wgangp import train_wgangp

from model import Generator, Discriminator, Encoder
from tools import SimpleDataset, load_mnist

from fanogan1.train_encoder_izif import train_encoder_izif

from fanogan1.test_anomaly_detection import test_anomaly_detection

from fanogan1.save_compared_images import save_compared_images


import datetime
import time
import pickle

# from fanogan1.test_anomaly_detection import test_anomaly_detection



def main(opt):
    
    start_time = time.time()
    if type(opt.seed) is int:
        torch.manual_seed(opt.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    datastamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    print("----------  dataset loading stage  ----------")
    (x_train, y_train), (x_test, y_test) = load_mnist(image_size=opt.img_size,
                                       trainset_path=opt.path,
                                       training_label=opt.training_label,
                                       split_rate=opt.split_rate)
    train_mnist = SimpleDataset(x_train, y_train,
                                transform=transforms.Compose(
                                    [transforms.ToPILImage(),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.5], [0.5])])
                                )
    train_dataloader = DataLoader(train_mnist, batch_size=opt.batch_size,
                                  shuffle=True)

    generator = Generator(opt)
    discriminator = Discriminator(opt)
    train_wgangp(opt, generator, discriminator, train_dataloader, device, datastamp)


    train_mnist = SimpleDataset(x_train, y_train,
                                transform=transforms.Compose(
                                    [transforms.ToPILImage(),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.5], [0.5])])
                                )
    train_dataloader = DataLoader(train_mnist, batch_size=opt.batch_size,
                                  shuffle=True)

    generator = Generator(opt)
    discriminator = Discriminator(opt)
    encoder = Encoder(opt)

    train_encoder_izif(opt, datastamp, generator, discriminator, encoder,
                       train_dataloader, device)


    test_mnist = SimpleDataset(x_test, y_test,
                               transform=transforms.Compose(
                                   [transforms.ToPILImage(),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.5], [0.5])])
                               )
    test_dataloader = DataLoader(test_mnist, batch_size=1, shuffle=False)

    generator = Generator(opt)
    discriminator = Discriminator(opt)
    encoder = Encoder(opt)
    
    test_anomaly_detection(opt, datastamp, generator, discriminator, encoder,
                           test_dataloader, device)


    test_mnist = SimpleDataset(x_test, y_test,
                               transform=transforms.Compose(
                                   [transforms.ToPILImage(),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.5], [0.5])])
                               )
    test_dataloader = DataLoader(test_mnist, batch_size=opt.n_grid_lines,
                                 shuffle=True)

    generator = Generator(opt)
    encoder = Encoder(opt)

    save_compared_images(opt, datastamp, generator, encoder, test_dataloader, device)

    print('total time in seconds = ' + str(time.time() - start_time))

    with open("results"+datastamp +"/run_report.txt",'w') as f:
        f.write('opt:  \n\n')
        f.write(opt.__str__())
        f.write('\n\ngenerator:  \n\n')
        f.write(generator.__str__())
        f.write('\n\ndiscriminator:  \n\n')
        f.write(discriminator.__str__())
        f.write('\n\nencoder:  \n\n')
        f.write(encoder.__str__())
        f.write('\n\ntotal time in seconds = ' + str(time.time() - start_time))




if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_epochs", type=int, default=200,
                        help="number of epochs of training")
    parser.add_argument("--batch_size", type=int, default=16,
                        help="size of the batches")
    parser.add_argument("--lr", type=float, default=0.0002,
                        help="adam: learning rate")
    parser.add_argument("--b1", type=float, default=0.5,
                        help="adam: decay of first order momentum of gradient")
    parser.add_argument("--b2", type=float, default=0.999,
                        help="adam: decay of first order momentum of gradient")
    parser.add_argument("--latent_dim", type=int, default=200,
                        help="dimensionality of the latent space")
    parser.add_argument("--img_size", type=int, default=300,
                        help="size of each image dimension")
    parser.add_argument("--channels", type=int, default=1,
                        help="number of image channels")
    parser.add_argument("--n_critic", type=int, default=5,
                        help="number of training steps for "
                             "discriminator per iter")
    parser.add_argument("--sample_interval", type=int, default=400,
                        help="interval betwen image samples")
    parser.add_argument("--training_label", type=int, default=0,
                        help="label for normal images")
    parser.add_argument("--split_rate", type=float, default=0.8,
                        help="rate of split for normal training data")
    parser.add_argument("--seed", type=int, default=None,
                        help="value of a random seed")
    parser.add_argument("--path", type=str, default='../imagess',
                        help="trainset path")   
    parser.add_argument("--n_grid_lines", type=int, default=10,
                        help="number of grid lines in the saved image")
    parser.add_argument("--n_iters", type=int, default=None,
                        help="value of stopping iterations") 
    opt = parser.parse_args()

    main(opt)