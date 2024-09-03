import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from torch import Tensor
from torch.utils.tensorboard import SummaryWriter
from torchsummary import summary
import numpy as np
import datetime
import os
import pickle
from tqdm import tqdm
from sklearn.metrics import r2_score

from loss import WeightedHuberLoss

from torcheval.metrics import R2Score

# Main inspirations to develop this code:
# https://github.com/phillipi/pix2pix
# https://github.com/akanametov/pix2pix

# Import models architectures as they are defined in a separate file (change imports if you want to use different ones)
from pytorch_models import SimilHuber_Generator as Generator
from pytorch_models import SimpleDiscriminator as Discriminator

# torch.autograd.set_detect_anomaly(True)


def generator_loss(input, target, gen, disc):
    with torch.cuda.amp.autocast():
        gen_output = gen(input)
        fake_output = disc(torch.cat((input, gen_output), dim=1))

        # adversarial loss between the discriminator's output and a tensor of ones
        # (indicating that the generator sho
        # uld aim to produce data that is classified as real)
        gan_loss_g = loss_obj(fake_output, torch.ones_like(fake_output)).to(device)

        # L1 loss between the generated data and the target data
        whuber_l = WeightedHuberLoss()
        # l1_loss = recon_criterion(target, gen_output).to(device)
        l1_loss_g = whuber_l(target, gen_output).to(device)

        # combination of the losses weighted by an hyperparameter LAMBDA
        total_gen_loss = gan_loss_g + LAMBDA * l1_loss_g

    return total_gen_loss, gan_loss_g, l1_loss_g


def discriminator_loss(input, target, gen, disc):
    with torch.cuda.amp.autocast():
        gen_output = gen(input)
        real_output = disc(torch.cat((input, target), dim=1))
        fake_output = disc(torch.cat((input, gen_output), dim=1))

        # loss between the discriminator's output for real data and a tensor of ones (indicating real data)
        real_loss = loss_obj(real_output, torch.ones_like(real_output)).to(device)

        # loss between the discriminator's output for fake data and a tensor of zeros (indicating fake data)
        generated_loss = loss_obj(fake_output, torch.zeros_like(fake_output)).to(device)

    # discriminator loss is halved in order to slow down the training process
    # (too fast compared to the generator training)
    return (real_loss + generated_loss) * 0.5


# Training function for a single epoch
def train_epoch(train_dataset, gen, disc, optimizer_g, optimizer_d, epoch_num):
    # set training mode
    gen.train()
    disc.train()

    # Initialize losses
    total_gen_loss = 0.0
    total_gan_loss = 0.0
    total_l1_loss = 0.0
    total_disc_loss = 0.0

    loop = tqdm(train_dataset)

    num_batches = 0
    for step, (input_batch, target_batch) in enumerate(loop):  # iterates through the training dataset in batches
        # moves the input and target batches to the device (GPU or CPU) if not already on that device
        x_ = input_batch.to(device)
        y_ = target_batch.to(device)

        # ==== Discriminator ====
        optimizer_d.zero_grad()
        disc_loss = discriminator_loss(x_, y_, gen, disc)  # Compute discriminator loss

        # Backpropagate and update discriminator weights
        disc_loss.backward()
        torch.nn.utils.clip_grad.clip_grad_norm_(disc.parameters(), 0.01, error_if_nonfinite=True)
        optimizer_d.step()
        # =======================

        # ====== Generator ======
        optimizer_g.zero_grad()
        gen_loss, gan_loss, l1_loss = generator_loss(x_, y_, gen, disc)  # Compute generator losses

        # Backpropagate and update generator weights
        gen_loss.backward()  # (only use gen_loss)
        torch.nn.utils.clip_grad.clip_grad_norm_(gen.parameters(), 0.01, error_if_nonfinite=True)
        optimizer_g.step()
        # =======================

        # Update total losses
        total_gen_loss += gen_loss
        total_gan_loss += gan_loss
        total_l1_loss += l1_loss
        total_disc_loss += disc_loss

        num_batches += 1

        loop.set_description(f"Epoch [{epoch_num + 1}/{n_epochs}]")
        loop.set_postfix(total_gen_loss=total_gen_loss.item() / num_batches,
                         total_gan_loss=total_gan_loss.item() / num_batches,
                         total_l1_loss=total_l1_loss.item() / num_batches,
                         total_disc_loss=total_disc_loss.item() / num_batches)

    # Compute train losses for the epoch
    avg_gen_loss = total_gen_loss / num_batches
    avg_gan_loss = total_gan_loss / num_batches
    avg_l1_loss = total_l1_loss / num_batches
    avg_disc_loss = total_disc_loss / num_batches

    return avg_gen_loss, avg_gan_loss, avg_l1_loss, avg_disc_loss


# Validation function
def validate(valid_dataset, gen):
    gen.eval()

    mse = nn.MSELoss()
    total_mean_squared_error = 0.0
    total_r2 = 0.0

    num_batches = 0
    with torch.no_grad():
        for _, (input_batch, target_batch) in enumerate(valid_dataset):
            x_ = input_batch.to(device)
            y_ = target_batch.to(device)

            # Generate fake data
            fake = gen(x_)

            # Compute MSE
            batch_mean_squared_error = mse(fake, y_).to(device)
            total_mean_squared_error += batch_mean_squared_error

            # Compute R2
            batch_r2 = r2_score(y_.squeeze().cpu().detach().numpy(),
                                fake.squeeze().cpu().detach().numpy())
            # batch_r2 = R2Score().update(fake.squeeze(), y_.squeeze()).compute()
            total_r2 += batch_r2

            num_batches += 1

    # Compute the RMSE
    mean_squared_error = total_mean_squared_error / num_batches
    rmse = torch.sqrt(mean_squared_error)
    r2 = total_r2/num_batches

    return rmse, r2


# Method to initialize NNs weights
def init_weights(m):
    classname = m.__class__.__name__
    if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
        nn.init.normal_(m.weight, 0.0, 0.02)
        if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias.data, 0.0)


if __name__ == '__main__':
    # Define directory to save the trained models
    model_dir = './models'

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    # hyperparameters
    LAMBDA = 50  # weight for total_gen_loss

    n_epochs = 50
    batch_size = 64  # consider changing this value depending on the avilable memory

    # optimizers parameters
    lr = 2e-4
    beta_1 = 0.5
    beta_2 = 0.999

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # set device (GPU:0 if available)
    torch.set_float32_matmul_precision('medium')  # reduce matmul precision to increase performance

    # instantiate Generator and print summary
    generator = Generator(1000).to(device)
    summary(generator, input_size=(1, 1000), batch_size=-1)

    # instantiate Discriminator and print summary
    discriminator = Discriminator().to(device)
    summary(discriminator, input_size=(2, 1000), batch_size=-1)

    # Define the losses
    loss_obj = nn.HuberLoss()
    recon_criterion = nn.L1Loss()  # reconstruction loss

    # Define the optimizers
    gen_optimizer = optim.Adam(generator.parameters(), lr=lr, betas=(beta_1, beta_2))
    disc_optimizer = optim.Adam(discriminator.parameters(), lr=lr/2000, betas=(beta_1, beta_2))

    # Load dataset
    data_path = './data/dataset_1000.pkl'  # dataset 200k train - 30k valid

    with open(data_path, 'rb') as infile:
        dataset_dict = pickle.load(infile)

    # Prepare training dataset
    train_tensor = TensorDataset(Tensor(dataset_dict['X_train'].astype(np.float32))[:, None, :],
                                 Tensor(dataset_dict['y_train'].astype(np.float32))[:, None, :])
    train_dataset = DataLoader(train_tensor, batch_size=batch_size, num_workers=0)

    # Prepare validation dataset
    valid_tensor = TensorDataset(Tensor(dataset_dict['X_valid'].astype(np.float32))[:, None, :],
                                 Tensor(dataset_dict['y_valid'].astype(np.float32))[:, None, :])
    valid_dataset = DataLoader(valid_tensor, batch_size=batch_size, num_workers=0)

    # Initialize generator weights
    generator.apply(init_weights)

    # Initialize discriminator weights
    discriminator.apply(init_weights)

    # Name of the trained model
    # SUGGEST TO CHANGE THIS AT EACH TRAINING TO AVOID OVERWRITING WHEN SAVING MODEL
    model_name = 'model_name'


    # Comment the following section if you don't want to save the training metrics evolution
    # ====== TensorBoard =======
    # Set up the log directory
    log_dir = "TensorBoard_logs"
    os.makedirs(log_dir, exist_ok=True)

    # Set up the summary writers
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    model_log_dir = os.path.join(log_dir, current_time + '_' + model_name)

    # Initialize TensorBoard writer
    writer = SummaryWriter(model_log_dir)
    # ==========================

    best_val_metric = float('-inf')  # initialize validation metric to inf
    best_rmse = float('-inf')

    # Train the model
    for epoch in range(n_epochs):
        gen_loss, gan_loss, l1_loss, disc_loss = train_epoch(train_dataset, generator, discriminator,
                                                             gen_optimizer, disc_optimizer, epoch)
        val_metric, r2_metric = validate(valid_dataset, generator)

        # Write losses and validation RMSE to TensorBoard
        writer.add_scalar('gen_loss', gen_loss, epoch)
        writer.add_scalar('gan_loss', gan_loss, epoch)
        writer.add_scalar('l1_loss', l1_loss, epoch)
        writer.add_scalar('disc_loss', disc_loss, epoch)
        writer.add_scalar('valid_RMSE', val_metric, epoch)
        writer.add_scalar('valid_R2', r2_metric, epoch)

        print(val_metric, r2_metric)

        # Save the model if it has the best validation metric so far
        if (r2_metric > best_val_metric) & (r2_metric <= 0.85):
            print(f'\tSaving new best model. valid_RMSE={val_metric:.6f}')
            best_val_metric = r2_metric
            best_rmse = val_metric
            torch.save(generator.state_dict(), os.path.join(model_dir, f'{model_name}.pt'))
        elif (r2_metric >= 0.85) & bool(val_metric < best_rmse):
            print(f'\tSaving new best model. valid_RMSE={val_metric:.6f}')
            best_val_metric = r2_metric
            best_rmse = val_metric
            torch.save(generator.state_dict(), os.path.join(model_dir, f'{model_name}.pt'))