import math
from einops.einops import reduce, rearrange
import torch.nn as nn
import numpy as np
from matplotlib import pyplot as plt
import cv2
import torch
from sklearn.decomposition import PCA
from sklearn.cluster import AgglomerativeClustering
import scipy.cluster.hierarchy as hcluster

from itertools import repeat
import collections.abc as container_abcs

from torchvision import transforms

from pl_bolts.transforms.dataset_normalizations import cifar10_normalization

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def plot_islands_agreement(levels, image):
    image_cpu = image.permute(1,2,0).detach().cpu().numpy()
    lin = nn.Linear(levels.shape[-1], 2).cuda()
    levels_2 = lin(levels)
    levels_2 = rearrange(levels_2,'(w h) l a -> w h l a', w = int(math.sqrt(levels.detach().cpu().numpy().shape[0])))
    levels_cpu_2 = levels_2.detach().cpu().numpy()

    mylevels = []
    for l in range(levels_cpu_2.shape[2]):
        mylevels.append(levels_cpu_2[:,:,l,:])

    fig, axs = plt.subplots(1, len(mylevels) + 1)
    plt.rcParams["figure.figsize"] = (25,3)
    axs[-1].imshow(image_cpu)
    axs[-1].set_box_aspect(1)
    axs[-1].grid(False)
    axs[-1].axes.xaxis.set_visible(False)
    axs[-1].axes.yaxis.set_visible(False)
    for i, matrice in enumerate(mylevels):
        x = np.arange(0.5, matrice.shape[0], 1)
        y = np.arange(0.5, matrice.shape[0], 1)
        xx, yy = np.meshgrid(x, y)
        r = np.power(np.add(np.power(matrice[:,:,0],2), np.power(matrice[:,:,1],2)),0.5)
        axs[i].imshow(r, cmap='inferno', interpolation='nearest')

        axs[i].set_box_aspect(1)
        axs[i].grid(False)
        axs[i].axes.xaxis.set_visible(False)
        axs[i].axes.yaxis.set_visible(False)

    plt.savefig("islands.png")
    # plt.show()

def accuracy(output, target, topk=(1,)):
        """Computes the precision@k for the specified values of k"""
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

def adjust_learning_rate(optimizer, epoch, max_epochs, FLAGS):
    """Decay the learning rate with half-cycle cosine after warmup"""
    if epoch < FLAGS.warmup:
        lr = FLAGS.learning_rate * epoch / FLAGS.warmup
    else:
        lr = FLAGS.min_lr + (FLAGS.learning_rate - FLAGS.min_lr) * 0.5 * \
            (1. + math.cos(math.pi * (epoch - FLAGS.warmup) / (max_epochs - FLAGS.warmup)))
    for param_group in optimizer.param_groups:
        if "lr_scale" in param_group:
            param_group["lr"] = lr * param_group["lr_scale"]
        else:
            param_group["lr"] = lr
    return lr

def apply_transforms(dm, FLAGS):
    cifar_mean = [0.4914, 0.4822, 0.4465]
    cifar_std = [0.2470, 0.2435, 0.2616]
    normalize_cifar = transforms.Normalize(mean=cifar_mean,
                                     std=cifar_std)
    s=1
    color_jitter = transforms.ColorJitter(
           0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s
       )
    # 10% of the image
    blur = transforms.GaussianBlur((3, 3), (0.1, 2.0))

    if not FLAGS.supervise:
        dm.train_transforms = transforms.Compose(
            [
                transforms.RandomResizedCrop(size=32),
                transforms.RandomHorizontalFlip(p=0.5),  # with 0.5 probability
                transforms.RandomApply([color_jitter], p=0.8),
                transforms.RandomApply([blur], p=0.5),
                transforms.RandomGrayscale(p=0.2),

                transforms.ToTensor(),
                cifar10_normalization()
            ]
        )
        dm.val_transforms = transforms.Compose(
            [
                transforms.ToTensor(),
                cifar10_normalization()
            ]
        )
        dm.test_transforms = transforms.Compose(
            [
                transforms.ToTensor(),
                cifar10_normalization()
            ]
        )


    else:
        dm.train_transforms = transforms.Compose(
            [
                transforms.ToTensor(),
                cifar10_normalization()
            ]
        )
        dm.val_transforms = transforms.Compose(
            [
                transforms.ToTensor(),
                cifar10_normalization()
            ]
        )
        dm.test_transforms = transforms.Compose(
            [
                transforms.ToTensor(),
                cifar10_normalization()
            ]
        )


def plot_BU_TD_TIME_ATT(BU_TD_TIME_ATT, size=100):

    BU_TD_TIME_ATT_cpu = []
    BU_TD_TIME_ATT_column = []

    names = ["Bottom-Up", "Top-Down", "Time", "Attention"]
    
    for i, el in enumerate(BU_TD_TIME_ATT):
        if(i>0 and BU_TD_TIME_ATT[i] == None):
            BU_TD_TIME_ATT[i] = torch.zeros(BU_TD_TIME_ATT[0].shape)

    for i, el in enumerate(BU_TD_TIME_ATT):
        BU_TD_TIME_ATT[i] = reduce(BU_TD_TIME_ATT[i],'n l d -> n l', 'mean')
        BU_TD_TIME_ATT[i] = rearrange(BU_TD_TIME_ATT[i],'(w h) l -> w h l', w = int(math.sqrt(BU_TD_TIME_ATT[i].detach().cpu().numpy().shape[0])))
        BU_TD_TIME_ATT_cpu.append(BU_TD_TIME_ATT[i].detach().cpu().numpy())
        BU_TD_TIME_ATT_column.append(BU_TD_TIME_ATT_cpu[i][:,:,0])

        for lev in range(BU_TD_TIME_ATT_cpu[i].shape[-1] - 1):
            BU_TD_TIME_ATT_column[i] = np.vstack((BU_TD_TIME_ATT_cpu[i][:,:,lev+1], BU_TD_TIME_ATT_column[i]))

        BU_TD_TIME_ATT_column[i] = ((BU_TD_TIME_ATT_column[i] / np.max(BU_TD_TIME_ATT_column[i])) * 255).astype('uint8')
        
        BU_TD_TIME_ATT_column[i] = (BU_TD_TIME_ATT_column[i]).astype(np.uint8)
        BU_TD_TIME_ATT_column[i] = cv2.applyColorMap(BU_TD_TIME_ATT_column[i], cv2.COLORMAP_INFERNO)
        BU_TD_TIME_ATT_column[i] = cv2.resize(BU_TD_TIME_ATT_column[i], (size, size*BU_TD_TIME_ATT_cpu[i].shape[-1]))

    plt.clf()

    BU_TD_TIME_ATT_columns = np.hstack((BU_TD_TIME_ATT_column[0], BU_TD_TIME_ATT_column[1], BU_TD_TIME_ATT_column[2], BU_TD_TIME_ATT_column[3]))

    cv2.imshow("BU_TD_TIME_ATT", BU_TD_TIME_ATT_columns)
    cv2.waitKey(1)


class IslandsPlotter():
    def __init__(self, FLAGS) -> None:
        self.columns = None
        self.reconstructions = None
        self.FLAGS=FLAGS
        self.to_plot = torch.zeros(((FLAGS.image_size // FLAGS.patch_size)**2, FLAGS.patch_size**2*FLAGS.n_channels), requires_grad=False).cuda()
    def add(self, column, reconstruction, first = False):
        if first:
            self.columns = column
            self.reconstructions = reconstruction
        else:
            self.columns = np.hstack((self.columns, column))
            self.reconstructions = np.hstack((self.reconstructions, reconstruction))
    def plot(self):
        cv2.imshow("Islands of agreement GPU "+str(torch.cuda.current_device()), self.columns)
        temp_rec = cv2.cvtColor(self.reconstructions, cv2.COLOR_RGB2BGR)
        cv2.imshow("Reconstructions GPU "+str(torch.cuda.current_device()), temp_rec)
        cv2.waitKey(1)

    def get_reconstruction(self, pred_pixel_values, unmasked_patches, masked_indices, unmasked_indices):
        index_masked = 0
        index_unmasked = 0
        all_indices = self.to_plot.shape[0]
        for i in torch.cat((unmasked_indices, masked_indices), dim=0):
            if torch.isin(i, masked_indices):
                self.to_plot[i] = pred_pixel_values[index_masked]
                index_masked += 1
            elif(i in unmasked_indices):
                self.to_plot[i] = unmasked_patches[index_unmasked]
                index_unmasked += 1
            all_indices -=1
        
        to_plot = rearrange(self.to_plot, '(h w) (p1 p2 c) -> (h p1) (w p2) c', p1 = self.FLAGS.patch_size, p2 = self.FLAGS.patch_size, w = (self.FLAGS.image_size // self.FLAGS.patch_size))
        return to_plot

    def plot_islands(self, levels, img, rec, unmasked_patches, masked_indices, unmasked_indices, iter=0, size=100):

        n, l, d = levels.shape
        threshold = 0.99

        cos_sim = nn.CosineEmbeddingLoss(reduction='none')

        levels = rearrange(levels,'n l d -> l n d')
        islands = None

        rec = self.get_reconstruction(rec, unmasked_patches[0], masked_indices[0], unmasked_indices[0])

        img_resized = cv2.resize(((img).permute(1,2,0).cpu().numpy()), (size,size))
        rec_resized = cv2.resize(((rec).cpu().numpy()), (size,size))
        for l, level in enumerate(levels):
            similarity = sim_matrix(levels[l], levels[l]).float()
            sim_image = similarity.cpu().numpy().astype(np.uint8)

            cluster_thresh = 0.1
            clusters = hcluster.fclusterdata(similarity.cpu(), cluster_thresh, criterion="distance")
            clusters = rearrange(clusters, "(h w) -> h w", h = (self.FLAGS.image_size // self.FLAGS.patch_size))
            clusters = clusters / similarity.shape[0] * 255
            uint_img = np.array(clusters).astype(np.uint8)#.astype('uint8')
            clusters = cv2.applyColorMap(uint_img, cv2.COLORMAP_INFERNO) #/ 255.
            

            island_img = cv2.resize(clusters, (size, size))
            
            if(l==0):
                reconstruction = np.vstack((rec_resized, img_resized))
                islands_img = island_img
            else:
                islands_img = np.vstack((island_img, islands_img))
            

        if iter == 0:
            self.add(islands_img, reconstruction, first=True)
        else:
            self.add(islands_img, reconstruction)

        if iter == 9:
            self.plot()

        return self.columns, self.reconstructions 


def sim_matrix(a, b, eps=1e-8):
    """
    added eps for numerical stability
    """
    a_n, b_n = a.norm(dim=1)[:, None], b.norm(dim=1)[:, None]
    a_norm = a / torch.max(a_n, eps * torch.ones_like(a_n))
    b_norm = b / torch.max(b_n, eps * torch.ones_like(b_n))
    sim_mt = torch.mm(a_norm, b_norm.transpose(0, 1))
    return sim_mt

def init_weights(module, nonlinearity="leaky_relu"):
    if isinstance(module, nn.Linear):
        nn.init.kaiming_uniform_(module.weight, a=math.sqrt(5.0), nonlinearity=nonlinearity)
        if module.bias is not None:
            nn.init.zeros_(module.bias)

    elif isinstance(module, (nn.Conv2d, nn.Conv1d)):
        nn.init.kaiming_uniform_(module.weight, a=math.sqrt(5.0), nonlinearity=nonlinearity)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
            
    elif isinstance(module, (nn.LayerNorm, nn.GroupNorm, nn.BatchNorm2d)):
        nn.init.zeros_(module.bias)
        nn.init.ones_(module.weight)

def init_weights_relu(module, nonlinearity="relu"):
    if isinstance(module, nn.Linear):
        nn.init.kaiming_uniform_(module.weight, a=math.sqrt(5.0), nonlinearity=nonlinearity)
        if module.bias is not None:
            nn.init.zeros_(module.bias)

    elif isinstance(module, (nn.Conv2d, nn.Conv1d)):
        nn.init.kaiming_uniform_(module.weight, a=math.sqrt(5.0), nonlinearity=nonlinearity)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
            
    elif isinstance(module, (nn.LayerNorm, nn.GroupNorm, nn.BatchNorm2d)):
        nn.init.zeros_(module.bias)
        nn.init.ones_(module.weight)


# From PyTorch internals
def _ntuple(n):
    def parse(x):
        if isinstance(x, container_abcs.Iterable):
            return x
        return tuple(repeat(x, n))

    return parse


to_1tuple = _ntuple(1)
to_2tuple = _ntuple(2)
to_3tuple = _ntuple(3)
to_4tuple = _ntuple(4)
to_ntuple = _ntuple

class ConvEmbed(nn.Module):
    """ Image to Conv Embedding
    """

    def __init__(self,
                 patch_size=7,
                 in_chans=3,
                 embed_dim=64,
                 stride=4,
                 padding=2,
                 norm_layer=None):
        super().__init__()
        patch_size = to_2tuple(patch_size)
        self.patch_size = patch_size

        self.proj = nn.Conv2d(
            in_chans, embed_dim,
            kernel_size=patch_size,
            stride=stride,
            padding=padding
        )
        self.norm = norm_layer(embed_dim) if norm_layer else None

    def forward(self, x):
        x = self.proj(x)
        
        B, C, H, W = x.shape
        x = rearrange(x, 'b c h w -> b (h w) c')
        if self.norm:
            x = self.norm(x)
        x = rearrange(x, 'b (h w) c -> b c h w', h=H, w=W)

        return x