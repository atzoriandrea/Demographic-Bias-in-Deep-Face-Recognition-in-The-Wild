# Degradation Gan Setting

1) First of all, you have to set your High Resolution and Low Resolution datasets in data.py.
Rememeber: datasets in both variables MUST be included in a list.
2) Simply execute `python3 train.py`. This will train both H2L and L2H GANs. In order to obtain coherent results, it is mandatory to train both of them at the same time.
