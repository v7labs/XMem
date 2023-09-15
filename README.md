# XMEM V7 Fork

To launch training: `python -m torch.distributed.launch --master_port 25763 --nproc_per_node=2 train.py --exp_id EXPERIMENT_NAME --stage 3 --load_network saves/XMem-s012.pth`

### Training data ###
Original training data has been moved to Dodo at `/data/thom/xmem_training_data`
You configure the dataloaders in `train.py` in the functions named `renew_<datasetname>_loader`
The format for each "video" is a folder of Annotations (.png images, 2D, 0,1) and a folder of images.
For in context learning you would just have 1 video.