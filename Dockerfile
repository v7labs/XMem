FROM pytorch/pytorch:1.11.0-cuda11.3-cudnn8-devel

RUN apt-key adv --keyserver keyserver.ubuntu.com --recv-keys A4B469963BF863CC
RUN apt-get update
RUN apt-get install -y git
RUN apt-get -y install curl
RUN apt install -y libglu1-mesa-dev libglib2.0-0

RUN mkdir /app
WORKDIR /app

ADD ./requirements.txt ./requirements.txt

RUN pip install -r ./requirements.txt

WORKDIR /app/XMem

RUN git config --global --add safe.directory '*'

# ENTRYPOINT python -m torch.distributed.launch --master_port 25763 --nproc_per_node=2 train.py --exp_id finetune_medical --stage 3 --load_network saves/XMem-s012.pth