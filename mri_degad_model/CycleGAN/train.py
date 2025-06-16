import torch
from dataset import NogadGadDataset
import sys
from utils import save_checkpoint, load_checkpoint
from torchvision import transforms 
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import torch.nn as nn
import torch.optim as optim 
import config
from tqdm import tqdm
from torchvision.utils import save_image
from discriminator_model import Discriminator
from generator_model import Generator 


def train_fn(
    discH, discZ, genZ, genH, loader, opt_disc, opt_gen, l1, mse, d_scaler, g_scaler
):
    H_reals = 0
    H_fakes = 0

    # progress bar
    loop = tqdm(loader, leave=True)

    for idx, (nogad, gad) in enumerate(loop):
        nogad = nogad.to(config.DEVICE)
        gad = gad.to(config.DEVICE)

        # Train Discriminators H and Z
        with torch.cuda.amp.autocast(): # for flt16
            fake_gad = genH(nogad)
            D_H_real = discH(gad)
            D_H_fake = discH(fake_gad.detach())
            H_reals += D_H_real.mean().item()
            H_fakes += D_H_fake.mean().item()
            D_H_real_loss = mse(D_H_real, torch.ones_like(D_H_real))
            D_H_fake_loss = mse(D_H_fake, torch.zeros_like(D_H_fake))
            D_H_loss = D_H_real_loss + D_H_fake_loss

            fake_nogad = genZ(gad)
            D_Z_real = discZ(nogad)
            D_Z_fake = discZ(fake_nogad.detach())
            D_Z_real_loss = mse(D_Z_real, torch.ones_like(D_Z_real))
            D_Z_fake_loss = mse(D_Z_fake, torch.zeros_like(D_Z_fake))
            D_Z_loss = D_Z_real_loss + D_Z_fake_loss

            # together
            D_loss = (D_H_loss + D_Z_loss) / 2

        opt_disc.zero_grad()
        d_scaler.scale(D_loss).backward()
        d_scaler.step(opt_disc)
        d_scaler.update()

        # Train Generators H and Z
        with torch.cuda.amp.autocast():
            # adversarial loss for both generators
            D_H_fake = discH(fake_gad)
            D_Z_fake = discZ(fake_nogad)
            loss_G_H = mse(D_H_fake, torch.ones_like(D_H_fake))
            loss_G_Z = mse(D_Z_fake, torch.ones_like(D_Z_fake))

            # cycle loss
            cycle_nogad = genZ(fake_gad)
            cycle_gad = genH(fake_nogad)
            cycle_nogad_loss = l1(nogad, cycle_nogad)
            cycle_gad_loss = l1(gad, cycle_gad)

            # identity loss (remove these for efficiency if you set lambda_identity=0)
            identity_nogad = genZ(nogad)
            identity_gad = genH(gad)
            identity_nogad_loss = l1(nogad, identity_nogad)
            identity_gad_loss = l1(gad, identity_gad)

            # add all togethor
            G_loss = (
                loss_G_Z
                + loss_G_H
                + cycle_nogad_loss * config.LAMBDA_CYCLE
                + cycle_gad_loss * config.LAMBDA_CYCLE
                + identity_gad_loss * config.LAMBDA_IDENTITY
                + identity_nogad_loss * config.LAMBDA_IDENTITY
            )

        opt_gen.zero_grad()
        g_scaler.scale(G_loss).backward()
        g_scaler.step(opt_gen)
        g_scaler.update()
        
        if idx % 500 == 0:
            save_image(fake_gad * 0.5 + 0.5, f"saved_images/gad_{idx}.png")
            save_image(fake_nogad * 0.5 + 0.5, f"saved_images/nogad_{idx}.png")

        loop.set_postfix(H_real=H_reals / (idx + 1), H_fake=H_fakes / (idx + 1))


def main():
    discH = Discriminator(in_channels=1).to(config.DEVICE)
    discZ = Discriminator(in_channels=1).to(config.DEVICE) 
    genH = Generator(img_channels=1, num_features=64, num_residuals=9).to(config.DEVICE)
    genZ = Generator(img_channels=1, num_features=64, num_residuals=9).to(config.DEVICE)

    opt_disc = optim.Adam(
        list(discH.parameters()) + list(discZ.parameters()),
        lr = config.LEARNING_RATE,
        betas = (0.5, 0.999),
    )
    opt_gen = optim.Adam(
        list(genZ.parameters()) + list(genH.parameters()),
        lr = config.LEARNING_RATE,
        betas = (0.5, 0.999),
    )

    L1 = nn.L1Loss()
    mse = nn.MSELoss()

    if config.LOAD_MODEL:
        load_checkpoint(
            config.CHECKPOINT_GEN_H, genH, opt_gen, config.LEARNING_RATE,
        )
        load_checkpoint(
            config.CHECKPOINT_GEN_Z, genZ, opt_gen, config.LEARNING_RATE,
        )
        load_checkpoint(
            config.CHECKPOINT_DISC_H, discH, opt_disc, config.LEARNING_RATE,
        )
        load_checkpoint(
            config.CHECKPOINT_DISC_Z, discZ, opt_disc, config.LEARNING_RATE,
        )
    
    dataset = NogadGadDataset(
        root_nogad = config.TRAIN_DIR + "/train_b_2",
        root_gad = config.TRAIN_DIR + "/train_a_2",
        transform = config.transform_pipeline
    )
    loader = DataLoader(
        dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
    )
    
    g_scaler = torch.cuda.amp.GradScaler()
    d_scaler = torch.cuda.amp.GradScaler()

    for epoch in range(config.NUM_EPOCHS):
        train_fn(
            discH,
            discZ,
            genZ,
            genH,
            loader,
            opt_disc,
            opt_gen,
            L1,
            mse,
            d_scaler,
            g_scaler,
        )

        if config.SAVE_MODEL:
            save_checkpoint(genH, opt_gen, filename=config.CHECKPOINT_GEN_H)
            save_checkpoint(genZ, opt_gen, filename=config.CHECKPOINT_GEN_Z)
            save_checkpoint(discH, opt_disc, filename=config.CHECKPOINT_DISC_H)
            save_checkpoint(discZ, opt_disc, filename=config.CHECKPOINT_DISC_Z)


if __name__ == "__main__":
    main()

