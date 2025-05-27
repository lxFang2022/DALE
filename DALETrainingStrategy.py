import os, argparse
import sys
sys.path.append("..")
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import torch.utils.data
from utils.loss import ce_loss,structure_loss, DiceLoss
from utils.utils import calculate_patch_entropy, split_fuzzy_nonfuzzy, split_fuzzy_nonfuzzy_soft, mask_patches_with_labels, Gass, process_image
import higher

def DALETraining(model, optimizer, masked_nonfuzzy_images, masked_nonfuzzy_labels, masked_fuzzy_images, masked_fuzzy_labels, image_mask_fuzzy, nonfuzzy_mask, label, parse_config):
    lateral, nonfea = model(masked_nonfuzzy_images)
    loss_nonfuzzy = structure_loss(lateral, masked_nonfuzzy_labels).mean()
    optimizer.zero_grad()
    loss_nonfuzzy.backward()
    optimizer.step()

    inner_optimiser = torch.optim.SGD(model.parameters(), lr=parse_config.lr_seg)
    para_w = torch.ones_like(label).float().cuda()
    para_w = torch.nn.Parameter(para_w, requires_grad=True)
    optimiser_para_w = torch.optim.Adam([para_w], lr=0.1)
    optimize_cnt = 1
    for optimize_id in range(optimize_cnt):
        inner_optimiser.zero_grad()
        optimiser_para_w.zero_grad()

        with higher.innerloop_ctx(model, inner_optimiser, copy_initial_weights=False) as (fcls, diffopt):

            virtual_outputs, _ = fcls(masked_fuzzy_images)
            seg_loss_fuzzy = structure_loss(virtual_outputs, masked_fuzzy_labels)
            ploss = seg_loss_fuzzy * para_w
            ploss_fuzzy = (ploss).mean()
            diffopt.step(ploss_fuzzy)

            # ------------------------ Calculate the optimal weights w --------------------------------------------------------------------------------------

            val_loss = 0
            for cnt in range(parse_config.val_cnt):
                val_pred, _ = fcls(masked_nonfuzzy_images)
                seg_loss_val = structure_loss(val_pred, masked_nonfuzzy_labels)
                loss_val = (seg_loss_val).mean()
                val_loss += loss_val
            (val_loss / parse_config.val_cnt).backward()
        optimiser_para_w.step()

    # ------------------------ Update the fuzzy set loss according to the optimal weights w ------------------------------------------------

    optimal_w = (para_w.squeeze() * image_mask_fuzzy).clone().detach()
    noisy_mask = (para_w.grad.squeeze() * image_mask_fuzzy) < 0
    clean_mask = (para_w.grad.squeeze() * image_mask_fuzzy) > 0

    optimizer.zero_grad()
    fuzzy_outputs, fuzfea = model(masked_fuzzy_images)

    if torch.sum(noisy_mask) > 0:
        gass_loss = Gass(clean_mask, nonfuzzy_mask, masked_fuzzy_labels, masked_nonfuzzy_labels,
                         nonfea.detach(), fuzfea.detach(), parse_config.num_classes)
    else:
        gass_loss = 0

    seg_loss_fuzzy = structure_loss(fuzzy_outputs, masked_fuzzy_labels)

    loss_fuzzy = ((seg_loss_fuzzy + parse_config.beta * gass_loss) * optimal_w).mean()
    loss_fuzzy.backward()
    optimizer.step()
