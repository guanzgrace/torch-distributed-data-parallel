from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp
from tqdm import tqdm

from base import *
from dataset import *
from logistic import *

def main(rank, world_size):
    # Setup the process groups
    setup(rank, world_size)

    # Prepare the dataloader
    dataloader = prepare(rank, world_size)
    
    # Instantiate the model and move it to the right device
    input_dim = dataloader.dataset.X.shape[1]
    output_dim = 1
    model = LogisticRegression(input_dim, output_dim).to(rank)
    
    # Wrap the model with DDP
    model = DDP(model, device_ids=[rank], output_device=rank, find_unused_parameters=False)
   
    # Set up our optimizer and loss functions
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = torch.nn.BCELoss(reduction="sum")

    # Train the model in batches
    for epoch in tqdm(range(1, int(epochs)+1), desc="Training Epochs", smoothing=0.0):
        dataloader.sampler.set_epoch(epoch)       
        
        for _, x in enumerate(dataloader):
            optimizer.zero_grad(set_to_none=True)
            
            pred = model(x[0])
            label = x[1].to(pred.device)
            
            loss = loss_fn(torch.squeeze(pred), label) / x[0].shape[0]
            loss.backward()
            optimizer.step()
        
        # Every 100 epochs, print some statistics
        if rank == 0 and epoch % 100 == 0:
            print('Epoch %d, Most recent batch loss: %.9f' % (epoch, loss), flush=True)
            # To get full training loss, we would have to add up the losses distributed across the GPUs
    cleanup()

if __name__ == '__main__':
    world_size = 2 # Number of GPUs / processes in the group
    mp.spawn(
        main,
        args=(world_size,), # Rank is each process' identificaiton number, from 0 to world_size-1
        nprocs=world_size
    )