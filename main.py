import os, sys
import torch
from architectures.SDNs.SDNConfig import SDNConfig
from architectures.SDNs.SDNDenseNet121 import SDNDenseNet121
from data import TrojAI
from settings import *
# def build_SDN_from_TrojAI_model(model, input_size, num_classes=5):
#     # if isinstance(model, inception.Inception3):
#     #     return
#     if isinstance(model, densenet.DenseNet):
#         return TrojAI_DenseNet121_to_SDN(model, input_size, num_classes)
#     raise RuntimeError('Unknown TrojAI architecture')

def test_SDN_for_TrojAI():
    os.chdir('/mnt/storage/Cloud/MEGA/TrojAI/data/trojai-round0-dataset')
    id_dense = 5
    id_resnet = 3
    id_inception = 2
    device = 'cuda'

    for i in [id_dense, id_resnet, id_inception]:
        model_name = f'id-{i:08d}/model.pt'
        model = torch.load(model_name)
        model = model.to(device).eval()

        trojai_model = SDNDenseNet121(model,
                                      TrojAI_input_size,
                                      TrojAI_num_classes,
                                      SDNConfig.DenseNet_inside_DenseBlock,
                                      device)
        acts, out = trojai_model.forward_w_acts(torch.zeros(TrojAI_input_size).to(device))
        print(len(acts)+1)
        print(trojai_model.get_layerwise_model_params())
        break

def test_TrojAI_dataset():
    dataset = TrojAI(folder='/mnt/storage/Cloud/MEGA/TrojAI/data/trojai-round0-dataset/id-00000000/example_data', batch_size=10)
    for batch in dataset.train_loader:
        print(batch[0].size(), batch[1].size())
        break


if __name__ == '__main__':
    test_SDN_for_TrojAI()
    # test_TrojAI_dataset()