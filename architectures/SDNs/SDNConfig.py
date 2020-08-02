class SDNConfig:
    DenseNet_end_DenseBlock = 0    # attaches an IC at each DenseBlock end
    DenseNet_inside_DenseBlock = 1 # attaches an IC at every DenseLayer in a DenseBlock

    DenseNet = [DenseNet_end_DenseBlock, DenseNet_inside_DenseBlock]