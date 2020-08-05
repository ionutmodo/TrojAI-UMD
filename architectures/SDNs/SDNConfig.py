class SDNConfig:
    DenseNet_attach_to_DenseBlocks = 0    # attaches an IC at each DenseBlock end
    DenseNet_attach_to_DenseLayers = 1 # attaches an IC at every DenseLayer in a DenseBlock

    DenseNet = [DenseNet_attach_to_DenseBlocks, DenseNet_attach_to_DenseLayers]