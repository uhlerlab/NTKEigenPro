import numpy  as np
import dataset
import eigenpro
import torch

"""
This is the kernel for a 1 hidden layer fully connected network
with a bias c = 1/sqrt(d) * Z (Z ~ standard normal) in the first layer,
d is the # of input dimensions.

This does not require data to be normalized, but making sure your data has
norm 1 will help make EigenPro more numerically stable.
"""

def kernel(pair1, pair2):

    out = pair1 @ pair2.transpose(1, 0) + 1
    N1 = torch.sum(torch.pow(pair1, 2), dim=-1).view(-1, 1) + 1
    N2 = torch.sum(torch.pow(pair2, 2), dim=-1).view(-1, 1) + 1

    XX = torch.sqrt(N1 @ N2.transpose(1, 0))
    out = out / XX

    out = torch.clamp(out, -1, 1)

    first = 1/np.pi * (out * (np.pi - torch.acos(out)) \
                       + torch.sqrt(1. - torch.pow(out, 2))) * XX
    sec = 1/np.pi * out * (np.pi - torch.acos(out))
    out = first + sec

    # Set C below as large as possible for fast convergence
    # C = 1 on real data usually works well
    # set C > 1 if EigenPro is not converging
    C = 1
    return out / C


def main():

    SEED = 17
    np.random.seed(SEED)
    X, y = dataset.make_data()
    train_X = X
    train_y = y
    test_X = X  # Adjust as desired
    test_y = y  # Adjust as desired

    num_classes = y.shape[-1]
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    model = eigenpro.FKR_EigenPro(kernel, train_X, train_y.shape[-1], device=device)
    MAX_EPOCHS = 10
    epochs = list(range(MAX_EPOCHS))
    model.fit(train_X, train_y, test_X, test_y, epochs=epochs, mem_gb=12)



if __name__ == "__main__":
    main()
