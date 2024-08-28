from base64 import encode
import torch
from torch import nn
import torch.nn.functional as F
import pytorch_lightning as pl
from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR
from einops.layers.torch import Rearrange
from einops import rearrange, repeat
import numpy as np
import cv2
import scipy.spatial as sp
from main_sim import FLAGS
# from main_sim import FLAGS
from utils import init_weights, IslandsPlotter

from utils import plot_islands_agreement, accuracy, adjust_learning_rate
from modules import Transformer, ContrastiveLoss

from PIL import Image

import wandb

class SimMIM(pl.LightningModule):
    def __init__(
        self,
        *,
        encoder,
        FLAGS,
        masking_ratio = 0.5
    ):
        super().__init__()
        self.FLAGS = FLAGS
        assert masking_ratio > 0 and masking_ratio < 1, 'masking ratio must be kept between 0 and 1'
        self.masking_ratio = masking_ratio

        self.plot_epoch = -1
        self.to_plot = torch.zeros((self.FLAGS.image_size // self.FLAGS.patch_size)**2, self.FLAGS.patch_size**2*self.FLAGS.n_channels, requires_grad=False).cuda()
        self.dim_patch_noemb = (self.FLAGS.patch_size**2)* self.FLAGS.n_channels
        # extract some hyperparameters and functions from encoder (vision transformer to be trained)

        self.encoder = encoder
        num_patches, encoder_dim = encoder.pos_embedding.shape[-2:]
        self.to_patch_conv = encoder.to_patch_embedding_conv

        self.classification_head = nn.Sequential(
            nn.LayerNorm(self.FLAGS.patch_dim),
            nn.Linear(self.FLAGS.patch_dim, self.FLAGS.n_classes)
        )

        # simple linear head
        self.perIMG = nn.Linear(self.FLAGS.patch_dim,self.FLAGS.n_channels*(self.FLAGS.patch_size**2))

        self.mask_token = nn.Parameter(torch.randn(encoder_dim))
        self.mask_patches = nn.Parameter(torch.randn(self.dim_patch_noemb))
        self.to_pixels = nn.Linear(encoder_dim, self.dim_patch_noemb)

    def forward(self, img, label, mode):
        if self.FLAGS.supervise:
            loss, pred = self.classify(img, label, mode)
            self.batch_acc = accuracy(pred.data,label,topk=(1,))[0]
            self.log(f'{mode}/accuracy', self.batch_acc, prog_bar=True, sync_dist=True)
        else:
            loss, overall_loss = self.pretrain(img, mode)

        self.log(f'{mode}/loss', loss, sync_dist=True)

        if mode == 'Training':
            self.log(f'{mode}/LR', self.optimizer.param_groups[0]['lr'], prog_bar=True, sync_dist=True)


        if not self.FLAGS.supervise:
            return overall_loss
        return loss


    def classify(self, img, label, mode):
        device = img.device

        apply_convolutions = True  #CONVOLUTIONS

        epoch_offset = self.FLAGS.max_epochs - self.FLAGS.max_epochs_finetune -1

        adjust_learning_rate(self.optimizer, self.current_epoch - epoch_offset, self.FLAGS.max_epochs_finetune, self.FLAGS)


        # get patches
        if apply_convolutions:
            patches = self.to_patch_conv(img)
            tokens = patches
        else:
            patches = self.to_patch(img)
        
        batch, num_patches, *_ = patches.shape

        
        if apply_convolutions:
            img_patchata = rearrange(tokens,'b (h w) (p1 p2 c) -> b c (h p1) (w p2)', h = self.encoder.num_patches_side , w=self.encoder.num_patches_side ,p1 = self.FLAGS.patch_size, p2 = self.FLAGS.patch_size)
            tokens = self.encoder.image_to_tokens[0](img_patchata)
            tokens = self.encoder.image_to_tokens[1](tokens)
        
        bottom_level, levels = self.encoder.init_forward(tokens)
        for iter in range(self.encoder.iters):
            bottom_level, levels, bu_out, td_out, toplot  = self.encoder(bottom_level, levels, iter)
            encoded_tokens = levels[:,1:,-1]
        
        if(mode=="Validation" and self.FLAGS.plot_islands and self.plot_epoch != self.current_epoch):
            self.plot_epoch = self.current_epoch
            plot_islands_agreement(toplot, img[0,:,:,:])

        pred = encoded_tokens.mean(dim = 1) if not self.FLAGS.use_token else encoded_tokens[:, 0]

        pred = self.classification_head(pred)
        class_loss = F.cross_entropy(pred, label)
        
        return class_loss, pred
    

    def pretrain(self, img, mode):
        device = img.device

        apply_convolutions = True  #CONVOLUTIONS

        # get patches
        if apply_convolutions:
            patches = self.to_patch_conv(img)
        else:
            patches = self.to_patch(img)


        batch, num_patches, *_ = patches.shape

        # for indexing purposes

        batch_range = torch.arange(batch, device = device)[:, None] 


        # patch to encoder tokens and add positions


        if apply_convolutions:
            tokens = patches    # b n d

        # prepare mask tokens

        mask_tokens = repeat(self.mask_token, 'd -> b n d', b = batch, n = num_patches)

        if apply_convolutions:
            mask_patches = repeat(self.mask_patches, 'd -> b n d', b = batch, n = num_patches)

        # calculate of patches needed to be masked, and get positions (indices) to be masked

        num_masked = int(self.masking_ratio * num_patches)
        rand_indices = torch.rand(batch, num_patches, device = device).argsort(dim = -1)
        masked_indices, unmasked_indices = rand_indices[:, :num_masked], rand_indices[:, num_masked:]
        masked_bool_mask = torch.zeros((batch, num_patches), device = device).scatter_(-1, masked_indices, 1).bool()

        # mask tokens
        if apply_convolutions:
            tokens = torch.where(masked_bool_mask[..., None], mask_patches.float(), tokens.float())
        else:
            tokens = torch.where(masked_bool_mask[..., None], mask_tokens.float(), tokens.float())
        
        if apply_convolutions:
            img_patchata = rearrange(tokens,'b (h w) (p1 p2 c) -> b c (h p1) (w p2)', h = self.encoder.num_patches_side , w=self.encoder.num_patches_side ,p1 = self.FLAGS.patch_size, p2 = self.FLAGS.patch_size)
            

            tokens = self.encoder.image_to_tokens[0](img_patchata)
            tokens = self.encoder.image_to_tokens[1](tokens)
        # ----------------------------------------------------
        
        recon_loss = 0
        enable = False
        bu_contr_loss = torch.tensor([0.0], device="cuda")
        td_contr_loss = torch.tensor([0.0], device="cuda")
        bottom_level, levels = self.encoder.init_forward(tokens)
        for iter in range(self.encoder.iters):
            bottom_level, levels, bu_out, td_out, toplot  = self.encoder(bottom_level, levels, iter)

            encoded = levels[:,1:,0] 

            encoded_mask_tokens = encoded[batch_range, masked_indices] 

            pred_pixel_values = self.to_pixels(encoded_mask_tokens)

            masked_patches = patches[batch_range, masked_indices]
            unmasked_patches = patches[batch_range, unmasked_indices]

            importance = [1.0]
            for i, imp in enumerate(importance):
                temp_bu = rearrange(bu_out[:,:,-1,:].float(), "b n d -> (b n) d")
                temp_td = rearrange(td_out[:,:,-2,:].float(), "b n d -> (b n) d")
                temp_cons = rearrange(levels[:,:,-1,:].float(), "b n d -> (b n) d")
                temp_cons_2 = rearrange(levels[:,:,-2,:].float(), "b n d -> (b n) d")
                bu_contr_loss += 1 - torch.mean((self.sim_matrix(temp_bu.float(), temp_cons.float())).float()) * importance[i]
                td_contr_loss += 1 - torch.mean((self.sim_matrix(temp_td.float(), temp_cons_2.float())).float()) * importance[i]

            
            if mode == "Validation": 
                if iter == 0:
                    self.islands_plotter = IslandsPlotter(self.FLAGS)
                islands_fig, recons_fig = self.islands_plotter.plot_islands(levels[0, 1:,:,:].detach(), img[0].detach(), pred_pixel_values[0].detach(), unmasked_patches.detach(), masked_indices.detach(), unmasked_indices.detach(), iter)
                if self.val_batch_idx == 0 and iter == 9:
                    islands_fig = cv2.cvtColor(islands_fig, cv2.COLOR_BGR2RGB)
                    recons_images = cv2.cvtColor(recons_fig, cv2.COLOR_BGR2RGB)
                    islands_of_agreement = wandb.Image(islands_fig)
                    recons_images = wandb.Image(recons_fig)
                    self.logger.log_image(key=f"{mode}/Islands", images=[islands_of_agreement])
                    self.logger.log_image(key=f"{mode}/Reconstructions", images=[recons_images])

        recon_loss = F.mse_loss(pred_pixel_values, masked_patches) * iter / self.encoder.iters
        
        bu_contr_loss = bu_contr_loss.float()
        td_contr_loss = td_contr_loss.float()

        self.log(f'{mode}/contrastive_regBU', bu_contr_loss, sync_dist=True)
        self.log(f'{mode}/contrastive_regTD', td_contr_loss, sync_dist=True)


        # use only the second loss
        return recon_loss , recon_loss + bu_contr_loss + td_contr_loss

    def sim_matrix(self, a, b, eps=1e-8):
        """
        added eps for numerical stability
        """
        a_n, b_n = a.norm(dim=1)[:, None], b.norm(dim=1)[:, None]
        a_norm = a / torch.max(a_n, eps * torch.ones_like(a_n))
        b_norm = b / torch.max(b_n, eps * torch.ones_like(b_n))
        sim_mt = torch.mm(a_norm, b_norm.transpose(0, 1))
        return sim_mt

    def configure_optimizers(self):
        self.optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.FLAGS.learning_rate,
            weight_decay=self.FLAGS.weight_decay,
            betas=(self.FLAGS.beta1, self.FLAGS.beta2)
        )

        return {'optimizer': self.optimizer} 

        
    def training_step(self, train_batch, batch_idx):
        image, label = train_batch
        self.training_batch_idx = batch_idx


        loss = self.forward(image, label, 'Training')

        return loss

    def validation_step(self, val_batch, batch_idx):
        image, label = val_batch
        self.val_batch_idx = batch_idx

        loss = self.forward(image, label, 'Validation')
        
        return loss

    def test_step(self, test_batch, batch_idx):
        image, label = test_batch
        self.test_batch_idx = batch_idx

        loss = self.forward(image, label, 'Test')

        return loss

    def plot_sample(self, img, masked_indices, unmasked_indices, pred_pixel_values, unmasked_patches):
        index_masked = 0
        index_unmasked = 0
        all_indices = self.to_plot.shape[0]-1
        for i in torch.cat((unmasked_indices[0], masked_indices[0]), dim=0):
            if torch.isin(i, masked_indices[0]):
                self.to_plot[i] = pred_pixel_values[0, index_masked]
                index_masked += 1
            elif(i in unmasked_indices[0]):
                self.to_plot[i] = unmasked_patches[0, index_unmasked]
                index_unmasked += 1
            all_indices -=1
        
        to_plot = rearrange(self.to_plot, '(h w) (p1 p2 c) -> (h p1) (w p2) c', p1 = self.FLAGS.patch_size, p2 = self.FLAGS.patch_size, w = (self.FLAGS.image_size // self.FLAGS.patch_size))
        
        twoimg = np.hstack((
            cv2.cvtColor(img[0].permute(1, 2, 0).detach().cpu().numpy(), cv2.COLOR_RGB2BGR),
            cv2.cvtColor(to_plot.detach().cpu().numpy(), cv2.COLOR_RGB2BGR)
        ))

        twoimg = cv2.resize(twoimg, (400,200))
        cv2.imshow("pred_pixel_values", twoimg)
        cv2.waitKey(1)
