class SDNConfig:
    DenseNet_blocks = 0    # attaches an IC at each DenseBlock end
    DenseNet_layers = 1 # attaches an IC at every DenseLayer in a DenseBlock
    ResNet = 2
    WideResNet = 3
    Inception3 = 4
    GoogLeNet = 5
    MobileNet2 = 6
    ShuffleNet = 7
    SqueezeNet = 8
    VGG = 9

    DenseNet_Mask = [True, True, True, True]
    DenseNet = [DenseNet_blocks, DenseNet_layers]
