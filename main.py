import os, sys
import torch
from architectures.SDNs.SDNConfig import SDNConfig
from architectures.SDNs.SDNDenseNet121 import SDNDenseNet121
from settings import *
# def build_SDN_from_TrojAI_model(model, input_size, num_classes=5):
#     # if isinstance(model, inception.Inception3):
#     #     return
#     if isinstance(model, densenet.DenseNet):
#         return TrojAI_DenseNet121_to_SDN(model, input_size, num_classes)
#     raise RuntimeError('Unknown TrojAI architecture')


if __name__ == '__main__':
    os.chdir('/mnt/storage/Cloud/MEGA/TrojAI/data/trojai-round0-dataset')
    id_dense = 5
    id_resnet = 3
    id_inception = 2

    for i in [id_dense, id_resnet, id_inception]:
        model_name = f'id-{i:08d}/model.pt'
        print(model_name)
        sys.exit(666)
        model = torch.load(model_name)
        model = model.cpu().eval()

        trojai_model = SDNDenseNet121(model, TrojAI_input_size, TrojAI_num_classes, SDNConfig.DenseNet_end_DenseBlock)
        acts, out = trojai_model.forward_w_acts(torch.zeros(TrojAI_input_size))
        print(trojai_model.get_layerwise_model_params())

        # params = build_SDN_from_TrojAI_model(model, (1, 3, 224, 224))
        # print('Total ICs:', len(params))
        # print(params)
        break
        # print(summary(model.cuda(), (3, 224, 224)))
        # for m in model.modules():
        # for name, param in model.named_parameters()
        # print('--------------------------------------------------------')
