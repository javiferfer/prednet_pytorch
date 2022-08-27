from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import src.utils.prednet as prednet
from src.utils.dataset import ImageListDataset
from src.utils.utils import load_list, write_image, write_outputs


def test(images,
          sequences,
          root,
          tensorboard,
          channels,
          initmodel,
          useamp,
          size,
          color_space,
          batchsize,
          num_workers,
          input_len,
          ext,
          device=torch.device('cpu')):

    if sequences == '':
        sequencelist = [images]
    else:
        sequencelist = load_list(sequences, root)
    logf = open('log_p.txt', 'w')
    writer = SummaryWriter() if tensorboard else None
    net = prednet.PredNet(channels, device=device).to(device)
    net.eval()
    if initmodel:
        print('Load model from', initmodel)
        net.load_state_dict(torch.load(initmodel))
    
    for seq in range(len(sequencelist)):
        imagelist = load_list(sequencelist[seq], root) 
        # update dataset and loader 
        img_dataset = ImageListDataset(img_size=(size[0], size[1]),
                                       input_len=input_len, channels=channels[0])
        img_dataset.load_images(img_paths=imagelist, c_space=color_space)
        data_loader = DataLoader(img_dataset, batch_size=batchsize, shuffle=False, num_workers=num_workers)
               
        if len(imagelist) == 0:
            print("Not found images.")
            return

        for i, data in enumerate(tqdm(data_loader, unit="batch")):
            for j in range(len(data)):
                for k in range(input_len):
                    x_batch = data[j, :k+2].view(1, k+2, channels[0], size[1], size[0])
                    with torch.no_grad():
                        with torch.cuda.amp.autocast(enabled=useamp):
                            pred, errors, eval_index = net(x_batch.to(device))
                    write_image(data[j, k].detach().cpu().numpy(), 'result/test_' + str(k + (i * batchsize + j) * input_len ) + 'x',
                                img_dataset.mode, color_space)
                    write_image(pred[0].detach().cpu().numpy(), 'result/test_' + str(k + (i * batchsize + j) * input_len ) + 'y_0',
                                img_dataset.mode, color_space)
                    if writer is not None:
                        prefix = f"test_{i}_{j}"
                        write_outputs(writer, net.outputs, k, prefix)
                    s = str(k + (i * batchsize + j) * input_len )
                    for l in range(net.n_layers):
                        s += ', ' + str(eval_index[l][0].detach().cpu().numpy())
                    logf.write(s + '\n')
                exts = [pred[0].view(1, 1, channels[0], size[1], size[0])]
                y_batch = data[j, input_len:].view(1, 1, channels[0], size[1], size[0])
                for k in range(ext):
                    with torch.no_grad():
                        pred_ext, _, _ = net(torch.cat([x_batch.to(device)] + exts + [y_batch.to(device)], axis=1))
                    exts.append(pred_ext.unsqueeze(0))
                    write_image(pred_ext[0].detach().cpu().numpy(), 'result/test_' + str((i * batchsize + j + 1) * input_len - 1) + 'y_' + str(k + 1),
                                img_dataset.mode, color_space)
                    if writer is not None:
                        prefix = f"text_ext_{i}_{j}"
                        write_outputs(writer, net.outputs, k, prefix)
