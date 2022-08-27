import os

from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
from torch.utils.tensorboard import SummaryWriter
from torchviz import make_dot

import src.utils.prednet as prednet
from src.utils.corr_wise import CorrWise
from src.utils.dataset import ImageListDataset, ImageHDF5Dataset
from src.utils.utils import load_list, write_image, write_outputs


def train(images,
          sequences,
          root,
          tensorboard,
          bprop,
          channels,
          period,
          lr,
          lr_rate,
          min_lr,
          saveimg,
          save,
          loss,
          up_down_up,
          amp,
          omg,
          initmodel,
          useamp,
          size,
          color_space,
          batchsize,
          shuffle,
          num_workers,
          input_len,
          device=torch.device('cpu')):

    if sequences == '':
        sequencelist = [images]
    else:
        sequencelist = load_list(sequences, root)

    logf = open('log_t.txt', 'w')
    writer = SummaryWriter() if tensorboard else None
    time_loss_weights = 1./(bprop - 1) * torch.ones(bprop, 1)
    time_loss_weights[0] = 0
    time_loss_weights = time_loss_weights.to(device)
    net = prednet.PredNet(
        channels,
        round_mode="up_donw_up" if up_down_up else "down_up_down",
        device=device,
        amp=amp,
        omg=omg,
    ).to(device)

    base_loss = nn.L1Loss()
    loss = CorrWise(base_loss, flow_method="FBFlow", return_warped=False, reduction_clip=False, flow_cycle_loss=True, scale_clip=True, device=device)
    if initmodel:
        print('Load model from', initmodel)
        net.load_state_dict(torch.load(initmodel))
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    print("Automatic Mixed Precision: {}".format(useamp))
    scaler = torch.cuda.amp.GradScaler(enabled=useamp)
    count = 0
    seq = 0
    lr_maker = lr_scheduler.StepLR(optimizer=optimizer, step_size=100, gamma=lr_rate)

    # create a dataset from initial imagelist
    if sequencelist[seq].endswith(('.h5', '.hdf5')):
        img_dataset = ImageHDF5Dataset(img_size=(size[0], size[1]),
                                       input_len=bprop, channels=channels[0])
        img_dataset.load_hdf5(sequencelist[seq], c_space=color_space)
        imagelist = list(img_dataset.hf_data.keys())
    else:
        imagelist = load_list(sequencelist[seq], root)
        img_dataset = ImageListDataset(img_size=(size[0], size[1]),
                                       input_len=bprop, channels=channels[0])
        img_dataset.load_images(img_paths=imagelist, c_space=color_space)
    # data loader
    data_loader = DataLoader(img_dataset, batch_size=batchsize, shuffle=shuffle, num_workers=num_workers)
    print("shuffle: ", shuffle)
    print("num_workers: ", num_workers)
    while count <= period:
        print("seqNo: {}".format(seq))
        if seq > 0:
            if sequencelist[seq].endswith(('.h5', '.hdf5')):
                img_dataset = ImageHDF5Dataset(img_size=(size[0], size[1]),
                                               input_len=bprop, channels=channels[0])
                img_dataset.load_hdf5(sequencelist[seq], c_space=color_space)
                imagelist = list(img_dataset.hf_data.keys())
            else:
                imagelist = load_list(sequencelist[seq], root) 
                # update dataset and loader 
                img_dataset = ImageListDataset(img_size=(size[0], size[1]),
                                        input_len=bprop, channels=channels[0])
                img_dataset.load_images(img_paths=imagelist, c_space=color_space)
            data_loader = DataLoader(img_dataset, batch_size=batchsize, shuffle=shuffle, num_workers=num_workers)

        if len(imagelist) == 0:
            print("Not found images.")
            return
        fn = 0
        for data in tqdm(data_loader, unit="batch"):
            print("frameNo: {}".format(fn))
            print("total frames: {}".format(count))
            with torch.cuda.amp.autocast(enabled=useamp):
                data = data.to(device)
                pred, errors, _ = net(data)
                if loss == 'corr_wise':
                    mean_error= loss(pred, data[:, -1])
                elif loss == 'ensemble':
                    corr_wise_error = loss(pred, data[:, -1])
                    mean_error = corr_wise_error + errors.mean()
                else:
                    mean_error = errors.mean()
                
            optimizer.zero_grad()
            scaler.scale(mean_error).backward()
            scaler.step(optimizer)
            scaler.update()
            if lr_maker.get_last_lr()[0] > min_lr:
                lr_maker.step()
            else:
                lr_maker.optimizer.param_groups[0]['lr'] = min_lr
            if saveimg:
                for j in range(len(data)):
                    write_image(data[j, -1].detach().cpu().numpy(), 'result/' + str(count) + '_' + str(fn / input_len + j) + 'x',
                                img_dataset.mode, color_space)
                    write_image(pred[j].detach().cpu().numpy(), 'result/' + str(count) + '_' + str(fn / input_len + j) + 'y',
                                img_dataset.mode, color_space)
            print("loss: ", mean_error.detach().cpu().numpy())
            logf.write(str(count) + ', ' + str(mean_error.detach().cpu().numpy()) + '\n')
            if writer is not None:
                writer.add_scalar("loss", mean_error.detach().cpu().numpy(), count)

            if count % save < len(data) * input_len:
                print("Save the model")
                torch.save(net.state_dict(), os.path.join("models", str(count) + ".pth"))
                if writer is not None:
                    for name, param in net.named_parameters():
                        writer.add_histogram(name, param.clone().cpu().data.numpy(), count)
                    write_outputs(writer, net.outputs, count)

            if count > period:
                break
            count += len(data) * input_len
            fn += len(data) * input_len
        seq = (seq + 1) % len(sequencelist)

    if writer is not None:
        print("Save tensorboard graph...")
        dummy_input = torch.zeros((1, 2, 3, size[0], size[1])).to(device)
        net.output_mode = 'prediction'
        writer.add_graph(net, dummy_input)
        writer.close()
    dot = make_dot(pred, params=dict(net.named_parameters()))
    f = open('model.dot', 'w')
    f.write(dot.source)
