import torch
from utils.metrics import *
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from torch_scatter import scatter_mean
from utils.permutation_loss import compute_permutation_loss
import os
from utils.get_sym_rmsd import get_symmetry_rmsd
from utils.utils import write_mol, read_mol, gumbel_softmax_no_random

def train_one_epoch(epoch, accelerator, args, logger, train_loader, model, optimizer, scheduler, com_coord_criterion, distmap_criterion, pocket_cls_criterion, pocket_coord_criterion, pocket_radius_criterion, device):
    y_list = []
    y_pred_list = []
    com_coord_list = []
    com_coord_pred_list = []
    rmsd_list = []
    rmsd_2A_list = []
    rmsd_5A_list = []
    centroid_dis_list = []
    centroid_dis_2A_list = []
    centroid_dis_5A_list = []
    pocket_coord_list = []
    pocket_coord_pred_list = []
    # pocket_coord_pred_for_update_list = []
    pocket_cls_list = []
    pocket_cls_pred_list = []
    pocket_cls_pred_round_list = []
    protein_len_list = []
    # pdb_list = []
    count = 0
    skip_count = 0
    batch_loss = 0.0
    batch_by_pred_loss = 0.0
    batch_distill_loss = 0.0
    com_coord_batch_loss = 0.0
    pocket_cls_batch_loss = 0.0
    pocket_coord_batch_loss = 0.0
    pocket_radius_pred_batch_loss = 0.0
    keepNode_less_5_count = 0

    n_steps_per_epoch = len(train_loader)
    if args.disable_tqdm:
        data_iter = train_loader
    else:
        data_iter = tqdm(train_loader, mininterval=args.tqdm_interval, disable=not accelerator.is_main_process)
    for batch_id, data in enumerate(data_iter, start=1):
        # try:
        # Denote num_atom as N, num_amino_acid_of_pocket as M, num_amino_acid_of_protein as L
        # com_coord_pred: [B x N, 3]
        # y_pred, y_pred_by_coord: [B, N x M]
        # pocket_cls_pred, protein_out_mask_whole: [B, L]
        # p_coords_batched_whole: [B, L, 3]
        # pred_pocket_center: [B, 3]
        com_coord_pred, compound_batch, y_pred, y_pred_by_coord, pocket_cls_pred, pocket_cls, protein_out_mask_whole, p_coords_batched_whole, pred_pocket_center, dis_map, keepNode_less_5, pocket_radius_pred, _ = model(data, train=True)
        # y = data.y
        if y_pred.isnan().any() or com_coord_pred.isnan().any() or pocket_cls_pred.isnan().any() or pred_pocket_center.isnan().any() or y_pred_by_coord.isnan().any():
            print(f"nan occurs in epoch {epoch}")
            continue
        com_coord = data.coords
        pocket_cls_loss = args.pocket_cls_loss_weight * pocket_cls_criterion(pocket_cls_pred, pocket_cls.float()) * (protein_out_mask_whole.numel() / protein_out_mask_whole.sum())
        pocket_coord_loss = args.pocket_distance_loss_weight * pocket_coord_criterion(pred_pocket_center, data.coords_center)
        contact_by_pred_loss = args.pair_distance_loss_weight * distmap_criterion(y_pred_by_coord, dis_map) if len(dis_map) > 0 else torch.tensor([0])
        if args.dismap_choice == 'wodm':
            contact_loss = torch.tensor([0], device=accelerator.device)
            contact_distill_loss = torch.tensor([0], device=accelerator.device)
        else:
            contact_loss = args.pair_distance_loss_weight * distmap_criterion(y_pred, dis_map) if len(dis_map) > 0 else torch.tensor([0])
            contact_distill_loss = args.pair_distance_distill_loss_weight * distmap_criterion(y_pred_by_coord, y_pred) if len(y_pred) > 0 else torch.tensor([0])
        
        if not args.force_fix_radius:
            pocket_radius_pred_loss = args.pocket_radius_loss_weight * pocket_radius_criterion(pocket_radius_pred.squeeze(1), data.ligand_radius.to(pocket_radius_pred.dtype))
        else:
            pocket_radius_pred_loss = torch.zeros_like(contact_distill_loss)
        
        if args.permutation_invariant:
            com_coord_loss = args.coord_loss_weight * compute_permutation_loss(com_coord_pred, com_coord, data, com_coord_criterion).mean()
        else:
            com_coord_loss = args.coord_loss_weight * com_coord_criterion(com_coord_pred, com_coord) if len(com_coord) > 0 else torch.tensor([0])

        sd = ((com_coord_pred.detach() - com_coord) ** 2).sum(dim=-1)
        rmsd = scatter_mean(sd, index=compound_batch, dim=0).sqrt().detach()

        centroid_pred = scatter_mean(src=com_coord_pred, index=compound_batch, dim=0)
        centroid_true = scatter_mean(src=com_coord, index=compound_batch, dim=0)
        centroid_dis = (centroid_pred - centroid_true).norm(dim=-1)
        
        if args.dismap_choice == 'wodm':
            loss = com_coord_loss + \
                contact_by_pred_loss + \
                pocket_cls_loss + pocket_radius_pred_loss + \
                pocket_coord_loss
        else:
            loss = com_coord_loss + \
                contact_loss + contact_by_pred_loss + contact_distill_loss + \
                pocket_cls_loss + pocket_radius_pred_loss + \
                pocket_coord_loss
        
        accelerator.backward(loss)
        if args.clip_grad:
            # clip_grad_norm_(model.parameters(), max_norm=1.0, error_if_nonfinite=True)
            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        # except Exception as e:
        #     logger.log_message(f"RuntimeError: {e}")
        #     continue

        if (batch_id + 1) % args.gradient_accumulate_step == 0:
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()
            

        batch_loss += len(y_pred)*contact_loss.item()
        batch_by_pred_loss += len(y_pred_by_coord)*contact_by_pred_loss.item()
        batch_distill_loss += len(y_pred_by_coord)*contact_distill_loss.item()
        com_coord_batch_loss += len(com_coord_pred)*com_coord_loss.item()

        pocket_cls_batch_loss += len(pocket_cls_pred)*pocket_cls_loss.item()
        pocket_coord_batch_loss += len(pred_pocket_center)*pocket_coord_loss.item()
        pocket_radius_pred_batch_loss += len(pocket_radius_pred)*pocket_radius_pred_loss.item()

        keepNode_less_5_count += keepNode_less_5

        y_list.append(dis_map.detach())
        y_pred_list.append(y_pred.detach())
        com_coord_list.append(com_coord)
        com_coord_pred_list.append(com_coord_pred.detach())
        rmsd_list.append(rmsd.detach())
        rmsd_2A_list.append((rmsd.detach() < 2).float())
        rmsd_5A_list.append((rmsd.detach() < 5).float())
        centroid_dis_list.append(centroid_dis.detach())
        centroid_dis_2A_list.append((centroid_dis.detach() < 2).float())
        centroid_dis_5A_list.append((centroid_dis.detach() < 5).float())

        
        batch_len = protein_out_mask_whole.sum(dim=1).detach()
        protein_len_list.append(batch_len)
        pocket_coord_pred_list.append(pred_pocket_center.detach())
        pocket_coord_list.append(data.coords_center)
        # use hard to calculate acc and skip samples
        for i, j in enumerate(batch_len):
            count += 1
            pocket_cls_list.append(pocket_cls.detach()[i][:j])
            pocket_cls_pred_list.append(pocket_cls_pred.detach()[i][:j].sigmoid())
            pocket_cls_pred_round_list.append(pocket_cls_pred.detach()[i][:j].sigmoid().round().int())
            pred_index_bool = (pocket_cls_pred.detach()[i][:j].sigmoid().round().int() == 1)
            if pred_index_bool.sum() == 0: # all the prediction is False, skip
                skip_count += 1

        if batch_id % args.log_interval == 0:
            stats_dict = {}
            stats_dict['step'] = batch_id
            stats_dict['lr'] = optimizer.param_groups[0]['lr']
            stats_dict['contact_loss'] = contact_loss.item()
            stats_dict['contact_by_pred_loss'] = contact_by_pred_loss.item()
            stats_dict['contact_distill_loss'] = contact_distill_loss.item()
            stats_dict['com_coord_loss'] = com_coord_loss.item()
            stats_dict['pocket_cls_loss'] = pocket_cls_loss.item()
            stats_dict['pocket_coord_loss'] = pocket_coord_loss.item()
            logger.log_stats(stats_dict, epoch, args, prefix='train')

    y = torch.cat(y_list)
    y_pred = torch.cat(y_pred_list)
    # y, y_pred = accelerator.gather((y, y_pred))

    com_coord = torch.cat(com_coord_list)
    com_coord_pred = torch.cat(com_coord_pred_list)
    # com_coord, com_coord_pred = accelerator.gather((com_coord, com_coord_pred))

    rmsd = torch.cat(rmsd_list)
    rmsd_2A = torch.cat(rmsd_2A_list)
    rmsd_5A = torch.cat(rmsd_5A_list)
    # rmsd, rmsd_2A, rmsd_5A = accelerator.gather((rmsd, rmsd_2A, rmsd_5A))
    rmsd_25 = torch.quantile(rmsd, 0.25)
    rmsd_50 = torch.quantile(rmsd, 0.50)
    rmsd_75 = torch.quantile(rmsd, 0.75)

    centroid_dis = torch.cat(centroid_dis_list)
    centroid_dis_2A = torch.cat(centroid_dis_2A_list)
    centroid_dis_5A = torch.cat(centroid_dis_5A_list)
    # centroid_dis, centroid_dis_2A, centroid_dis_5A = accelerator.gather((centroid_dis, centroid_dis_2A, centroid_dis_5A))
    centroid_dis_25 = torch.quantile(centroid_dis, 0.25)
    centroid_dis_50 = torch.quantile(centroid_dis, 0.50)
    centroid_dis_75 = torch.quantile(centroid_dis, 0.75)

    pocket_cls = torch.cat(pocket_cls_list)
    pocket_cls_pred = torch.cat(pocket_cls_pred_list)
    pocket_cls_pred_round = torch.cat(pocket_cls_pred_round_list)
    pocket_coord_pred = torch.cat(pocket_coord_pred_list)
    pocket_coord = torch.cat(pocket_coord_list)
    protein_len = torch.cat(protein_len_list)

    pocket_cls_accuracy = (pocket_cls_pred_round == pocket_cls).sum().item() / len(pocket_cls_pred_round)
    

    metrics = {"samples": count, "skip_samples": skip_count, "keepNode < 5": keepNode_less_5_count}
    metrics.update({"lr": optimizer.param_groups[0]['lr']})
    metrics.update({"contact_loss":batch_loss/len(y_pred), "contact_by_pred_loss":batch_by_pred_loss/len(y_pred), "contact_distill_loss": batch_distill_loss/len(y_pred)})
    metrics.update({"com_coord_huber_loss": com_coord_batch_loss/len(com_coord_pred)})
    metrics.update({"rmsd": rmsd.mean().item(), "rmsd < 2A": rmsd_2A.mean().item(), "rmsd < 5A": rmsd_5A.mean().item()})
    metrics.update({"rmsd 25%": rmsd_25.item(), "rmsd 50%": rmsd_50.item(), "rmsd 75%": rmsd_75.item()})
    metrics.update({"centroid_dis": centroid_dis.mean().item(), "centroid_dis < 2A": centroid_dis_2A.mean().item(), "centroid_dis < 5A": centroid_dis_5A.mean().item()})
    metrics.update({"centroid_dis 25%": centroid_dis_25.item(), "centroid_dis 50%": centroid_dis_50.item(), "centroid_dis 75%": centroid_dis_75.item()})

    metrics.update({"pocket_cls_bce_loss": pocket_cls_batch_loss / len(pocket_coord_pred)})
    metrics.update({"pocket_coord_mse_loss": pocket_coord_batch_loss / len(pocket_coord_pred)})
    metrics.update({"pocket_cls_accuracy": pocket_cls_accuracy})
    metrics.update({"pocket_radius_pred_loss": pocket_radius_pred_batch_loss / len(pocket_coord_pred)})
    metrics.update(pocket_metrics(pocket_coord_pred, pocket_coord))
    
    return metrics


@torch.no_grad()
def validate(accelerator, args, data_loader, model, com_coord_criterion, distmap_criterion, pocket_cls_criterion, pocket_coord_criterion, pocket_radius_criterion, device, stage=1):
    y_list = []
    y_pred_list = []
    com_coord_list = []
    com_coord_pred_list = []
    com_coord_per_sample_list = []
    com_coord_pred_per_sample_list = []
    com_coord_offset_per_sample_list = []
    com_coord_offset_pocket_center_per_sample_list = []
    # contain the ground truth for classiifcation(may not all)
    pocket_coord_list = []
    pocket_coord_pred_list = []
    # contain the ground truth for regression(all)
    pocket_coord_direct_list = []
    pocket_coord_pred_direct_list = []
    pocket_cls_list = []
    pocket_cls_pred_list = []
    # protein_len_list = []
    # real_y_mask_list = []

    rmsd_list = []
    rmsd_2A_list = []
    rmsd_5A_list = []
    centroid_dis_list = []
    centroid_dis_2A_list = []
    centroid_dis_5A_list = []
    pdb_list = []

    skip_count = 0
    count = 0
    batch_loss = 0.0
    batch_by_pred_loss = 0.0
    batch_distill_loss = 0.0
    com_coord_batch_loss = 0.0
    pocket_cls_batch_loss = 0.0
    pocket_coord_direct_batch_loss = 0.0
    pocket_radius_pred_batch_loss = 0.0
    keepNode_less_5_count = 0
    n_steps_per_epoch = len(data_loader)
    if args.disable_tqdm:
        data_iter = data_loader
    else:
        data_iter = tqdm(data_loader, mininterval=args.tqdm_interval, disable=not accelerator.is_main_process)
    for batch_id, data in enumerate(data_iter, start=1):
        data = data.to(device)
        com_coord_pred, compound_batch, y_pred, y_pred_by_coord, pocket_cls_pred, pocket_cls, protein_out_mask_whole, p_coords_batched_whole, pocket_coord_pred_direct, dis_map, keepNode_less_5, pocket_radius_pred, pocket_center_bias  = model(data, stage=stage, train=False)       
        # y = data.y
        com_coord = data.coords
        
        for i in range(compound_batch.max().item()+1):
            i_mask = (compound_batch == i)
            com_coord_pred_i = com_coord_pred[i_mask]
            com_coord_i = com_coord[i_mask]
            com_coord_pred_per_sample_list.append(com_coord_pred_i)
            com_coord_per_sample_list.append(com_coord_i)
                
        if args.symmetric_rmsd:
            rmsd = torch.zeros(len(data.pdb), device=device)
            for batch_item_id in range(len(data.pdb)):
                try:
                    mol = read_mol(f'{args.symmetric_rmsd}/{data.pdb[batch_item_id]}.sdf', None)
                    gt_lig_pos = np.array(com_coord[compound_batch==batch_item_id].detach().cpu())
                    pred_lig_pos = np.array(com_coord_pred[compound_batch==batch_item_id].detach().cpu())
                    rmsd_item = get_symmetry_rmsd(mol, gt_lig_pos, pred_lig_pos, mol2=mol)
                    rmsd[batch_item_id] = rmsd_item
                except Exception as e:
                    print("Using non corrected RMSD because of the error:", e)
                    sd = ((com_coord_pred[compound_batch==batch_item_id] - com_coord[compound_batch==batch_item_id]) ** 2).sum(dim=-1)
                    rmsd_item = sd.mean().sqrt()
                    rmsd[batch_item_id] = rmsd_item.item()
        else:
            sd = ((com_coord_pred - com_coord) ** 2).sum(dim=-1)
            rmsd = scatter_mean(src=sd, index=compound_batch, dim=0).sqrt()
        
        centroid_pred = scatter_mean(src=com_coord_pred, index=compound_batch, dim=0)
        centroid_true = scatter_mean(src=com_coord, index=compound_batch, dim=0)
        centroid_dis = (centroid_pred - centroid_true).norm(dim=-1)

        contact_by_pred_loss = args.pair_distance_loss_weight * distmap_criterion(y_pred_by_coord, dis_map) if len(dis_map) > 0 else torch.tensor([0])
        if args.dismap_choice == 'wodm':
            contact_loss = torch.tensor([0], device=device)
            contact_distill_loss = torch.tensor([0], device=device)
        else:
            contact_loss = args.pair_distance_loss_weight * distmap_criterion(y_pred, dis_map) if len(dis_map) > 0 else torch.tensor([0])
            contact_distill_loss = args.pair_distance_distill_loss_weight * distmap_criterion(y_pred_by_coord, y_pred) if len(y_pred) > 0 else torch.tensor([0])

        pocket_cls_loss = args.pocket_cls_loss_weight * pocket_cls_criterion(pocket_cls_pred, pocket_cls.float())
        pocket_coord_direct_loss = args.pocket_distance_loss_weight * pocket_coord_criterion(pocket_coord_pred_direct, data.coords_center)

        if args.permutation_invariant:
            com_coord_loss = args.coord_loss_weight * compute_permutation_loss(com_coord_pred, com_coord, data, com_coord_criterion).mean()
        else:
            com_coord_loss = args.coord_loss_weight * com_coord_criterion(com_coord_pred, com_coord)
        pocket_radius_pred_loss = pocket_radius_criterion(pocket_radius_pred.squeeze(1), data.ligand_radius.to(pocket_radius_pred.dtype))

        batch_loss += len(y_pred)*contact_loss.item()
        batch_by_pred_loss += len(y_pred_by_coord)*contact_by_pred_loss.item()
        batch_distill_loss += len(y_pred_by_coord)*contact_distill_loss.item()
        com_coord_batch_loss += len(com_coord_pred)*com_coord_loss.item()
        pocket_cls_batch_loss += len(pocket_cls_pred)*pocket_cls_loss.item()
        pocket_coord_direct_batch_loss += len(pocket_coord_pred_direct)*pocket_coord_direct_loss.item()
        pocket_radius_pred_batch_loss += len(pocket_radius_pred)*pocket_radius_pred_loss.item()
        keepNode_less_5_count += keepNode_less_5

        y_list.append(dis_map)
        y_pred_list.append(y_pred.detach())
        com_coord_list.append(com_coord)
        com_coord_pred_list.append(com_coord_pred.detach())

        rmsd_list.append(rmsd.detach())
        rmsd_2A_list.append((rmsd.detach() < 2).float())
        rmsd_5A_list.append((rmsd.detach() < 5).float())
        centroid_dis_list.append(centroid_dis.detach())
        centroid_dis_2A_list.append((centroid_dis.detach() < 2).float())
        centroid_dis_5A_list.append((centroid_dis.detach() < 5).float())

        batch_len = protein_out_mask_whole.sum(dim=1).detach()
        # protein_len_list.append(batch_len)
        pocket_coord_pred_direct_list.append(pocket_coord_pred_direct.detach())
        pocket_coord_direct_list.append(data.coords_center)
        for i, j in enumerate(batch_len):
            count += 1
            pdb_list.append(data.pdb[i])
            com_coord_offset_per_sample_list.append(data.coord_offset[i])
            com_coord_offset_pocket_center_per_sample_list.append(pocket_center_bias[i])
            pocket_cls_list.append(pocket_cls.detach()[i][:j])
            pocket_cls_pred_list.append(pocket_cls_pred.detach()[i][:j].sigmoid().round().int())
            pred_index_bool = (pocket_cls_pred.detach()[i][:j].sigmoid().round().int() == 1)
            if pred_index_bool.sum() != 0:
                pred_pocket_center = p_coords_batched_whole.detach()[i][:j][pred_index_bool].mean(dim=0).unsqueeze(0)
                pocket_coord_pred_list.append(pred_pocket_center)
                pocket_coord_list.append(data.coords_center[i].unsqueeze(0))
            else: # all the prediction is False, skip
                skip_count += 1
                pred_index_true = pocket_cls_pred[i][:j].sigmoid().unsqueeze(-1)
                pred_index_false = 1. - pred_index_true
                pred_index_prob = torch.cat([pred_index_false, pred_index_true], dim=-1)
                pred_index_log_prob = torch.log(pred_index_prob)
                pred_index_one_hot = gumbel_softmax_no_random(pred_index_log_prob, tau=args.gs_tau, hard=False)
                pred_index_one_hot_true = pred_index_one_hot[:, 1].unsqueeze(-1)
                pred_pocket_center_gumbel = pred_index_one_hot_true * p_coords_batched_whole[i][:j]
                pred_pocket_center_gumbel_mean = pred_pocket_center_gumbel.sum(dim=0) / pred_index_one_hot_true.sum(dim=0) 
                pocket_coord_pred_list.append(pred_pocket_center_gumbel_mean.unsqueeze(0).detach())
                pocket_coord_list.append(data.coords_center[i].unsqueeze(0))


        # real_y_mask_list.append(data.real_y_mask)
    y = torch.cat(y_list)
    y_pred = torch.cat(y_pred_list)
    
    com_coord = torch.cat(com_coord_list)
    com_coord_pred = torch.cat(com_coord_pred_list)

    rmsd = torch.cat(rmsd_list)
    rmsd_2A = torch.cat(rmsd_2A_list)
    rmsd_5A = torch.cat(rmsd_5A_list)
    rmsd_25 = torch.quantile(rmsd, 0.25)
    rmsd_50 = torch.quantile(rmsd, 0.50)
    rmsd_75 = torch.quantile(rmsd, 0.75)
    centroid_dis = torch.cat(centroid_dis_list)
    centroid_dis_2A = torch.cat(centroid_dis_2A_list)
    centroid_dis_5A = torch.cat(centroid_dis_5A_list)
    centroid_dis_25 = torch.quantile(centroid_dis, 0.25)
    centroid_dis_50 = torch.quantile(centroid_dis, 0.50)
    centroid_dis_75 = torch.quantile(centroid_dis, 0.75)

    pocket_cls = torch.cat(pocket_cls_list)
    pocket_cls_pred = torch.cat(pocket_cls_pred_list)

    if len(pocket_coord_pred_list) > 0:
        pocket_coord_pred = torch.cat(pocket_coord_pred_list)
        pocket_coord = torch.cat(pocket_coord_list)
    pocket_coord_pred_direct = torch.cat(pocket_coord_pred_direct_list)
    pocket_coord_direct = torch.cat(pocket_coord_direct_list)

    pocket_cls_accuracy = (pocket_cls_pred == pocket_cls).sum().item() / len(pocket_cls_pred)
    
    if args.save_rmsd_dir is not None:
        if not os.path.exists(args.save_rmsd_dir):
            os.system(f'mkdir -p {args.save_rmsd_dir}')
        for i in range(len(pdb_list)):
            pdb = pdb_list[i]
            rmsd_i = rmsd[i]
            with open(os.path.join(args.save_rmsd_dir, f'seed_{args.seed}.txt'), 'a') as f:
                f.write(f'{pdb} {rmsd_i.item()}\n')
    
    metrics = {"samples": count, "skip_samples": skip_count, "keepNode < 5": keepNode_less_5_count}
    metrics.update({"contact_loss":batch_loss/len(y_pred), "contact_by_pred_loss":batch_by_pred_loss/len(y_pred)})
    metrics.update({"com_coord_huber_loss": com_coord_batch_loss/len(com_coord_pred)})
    # Final evaluation metrics
    metrics.update({"rmsd": rmsd.mean().item(), "rmsd < 2A": rmsd_2A.mean().item(), "rmsd < 5A": rmsd_5A.mean().item()})
    metrics.update({"rmsd 25%": rmsd_25.item(), "rmsd 50%": rmsd_50.item(), "rmsd 75%": rmsd_75.item()})
    metrics.update({"centroid_dis": centroid_dis.mean().item(), "centroid_dis < 2A": centroid_dis_2A.mean().item(), "centroid_dis < 5A": centroid_dis_5A.mean().item()})
    metrics.update({"centroid_dis 25%": centroid_dis_25.item(), "centroid_dis 50%": centroid_dis_50.item(), "centroid_dis 75%": centroid_dis_75.item()})
    
    metrics.update({"pocket_cls_bce_loss": pocket_cls_batch_loss / len(pocket_cls_pred_list)})
    metrics.update({"pocket_coord_mse_loss": pocket_coord_direct_batch_loss / len(pocket_coord_pred_direct)})
    metrics.update({"pocket_cls_accuracy": pocket_cls_accuracy})
    metrics.update({"pocket_radius_pred_loss": pocket_radius_pred_batch_loss / len(pocket_coord_pred_direct)})

    if len(pocket_coord_pred_list) > 0:
        metrics.update(pocket_metrics(pocket_coord_pred, pocket_coord))

    if args.write_mol_to_file is not None:
        os.system('mkdir -p {}'.format(args.write_mol_to_file))
        
        compound_folder = os.path.join(args.data_path, 'renumber_atom_index_same_as_smiles')
        df_pdb_rmsd = pd.DataFrame({'pdb': pdb_list, 'rmsd': rmsd.detach().cpu()})
        df_pdb_rmsd.to_pickle(os.path.join(args.write_mol_to_file, 'info.pkl'))
        for i in range(len(pdb_list)):
            pdb = pdb_list[i]
            com_coord_pred_i = com_coord_pred_per_sample_list[i]
            # com_coord_i = com_coord_per_sample_list[i]
            coord_offset_i = com_coord_offset_per_sample_list[i]
            pocket_center_offset_i = com_coord_offset_pocket_center_per_sample_list[i]
            com_coord_pred_i += coord_offset_i + pocket_center_offset_i
            if os.path.exists(os.path.join(compound_folder,  f"{pdb}.sdf")):
                ligand_original_file_sdf = os.path.join(compound_folder,  f"{pdb}.sdf")
                output_file = os.path.join(args.write_mol_to_file, f"{pdb}.sdf")
            else:
                raise ValueError(f"Cannot find {pdb}.sdf in {compound_folder}")
            mol = write_mol(reference_file=ligand_original_file_sdf, coords=com_coord_pred_i, output_file=output_file)
    
    return metrics

