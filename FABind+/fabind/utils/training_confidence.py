import torch
from utils.metrics import *
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
import torchmetrics
from torch_scatter import scatter_mean
from utils.permutation_loss import compute_permutation_loss, compute_permutation_rmsd
import os
from utils.get_sym_rmsd import get_symmetry_rmsd
from utils.utils import write_mol, read_mol, gumbel_softmax_no_random

def train_one_epoch(epoch, accelerator, args, logger, train_loader, model, optimizer, scheduler, compound_confidence_criterion, device):
    # basic log
    rmsd_list = []
    rmsd_2A_list = []
    rmsd_5A_list = []
    centroid_dis_list = []
    centroid_dis_2A_list = []
    centroid_dis_5A_list = []
    protein_len_list = []
    count = 0
    skip_count = 0
    keepNode_less_5_count = 0    
    # sampling-relevant
    ranking_accuracy_list = []
    confidence_batch_loss = 0.0
    ranking_batch_loss = 0.0
    confidence_ce_batch_loss = 0.0
    confidence_correct_count = 0.0
    hit_count = 0.0

    n_steps_per_epoch = len(train_loader)
    if args.disable_tqdm:
        data_iter = train_loader
    else:
        data_iter = tqdm(train_loader, mininterval=args.tqdm_interval, disable=not accelerator.is_main_process)
    for batch_id, data in enumerate(data_iter, start=1):
        com_coord_pred, compound_batch, pocket_cls_pred, protein_out_mask_whole, keepNode_less_5, confidence_score_pred, _ = model(data, train=True)
        com_coord = data.coords        
        sd = ((com_coord_pred.detach() - com_coord) ** 2).sum(dim=-1)
        rmsd = scatter_mean(sd, index=compound_batch, dim=0).sqrt().detach()

        centroid_pred = scatter_mean(src=com_coord_pred, index=compound_batch, dim=0)
        centroid_true = scatter_mean(src=com_coord, index=compound_batch, dim=0)
        centroid_dis = (centroid_pred - centroid_true).norm(dim=-1)
        
        # sort confidence_score_pred according to rmsd
        confidence_score_pred = confidence_score_pred.squeeze()
        sorted_ranking_score_pred = confidence_score_pred[rmsd.argsort()]
        sorted_rmsd = rmsd[rmsd.argsort()]
        
        ranking_loss = 0.
        rmsd_2A_score = (rmsd < 2).float()
        
        for i in range(len(sorted_ranking_score_pred)):
            for j in range(i):
                if args.ranking_loss == 'dynamic_hinge':
                    # dynamic margin
                    gt_rmsd_diff = sorted_rmsd[i] - sorted_rmsd[j] # j is better than i, has lower rmsd
                    ranking_loss += F.relu(gt_rmsd_diff - (sorted_ranking_score_pred[j] - sorted_ranking_score_pred[i]))
                elif args.ranking_loss == 'logsigmoid':
                    ranking_loss += - F.logsigmoid((sorted_ranking_score_pred[j] - sorted_ranking_score_pred[i]))  # j is better than i, has lower rmsd => score[j] > score[i]

                ranking_accuracy_list.append(float(sorted_ranking_score_pred[j] > sorted_ranking_score_pred[i]))
                
        ranking_loss = ranking_loss / (len(sorted_ranking_score_pred) * (len(sorted_ranking_score_pred) - 1) / 2)
        if args.keep_cls_2A:
            ce_loss = compound_confidence_criterion(confidence_score_pred, rmsd_2A_score)
            loss = ranking_loss + ce_loss
        else:
            ce_loss = torch.tensor([0], device=accelerator.device)
            loss = ranking_loss
            
        # log hit count when sorted_ranking_score_pred[0] is the largest
        hit_count += sorted_ranking_score_pred[0] > sorted_ranking_score_pred[1:].max()
        confidence_correct_count += ((confidence_score_pred[0] > 0).float() == rmsd_2A_score).sum().item()

        
        accelerator.backward(loss)
        if args.clip_grad:
            # clip_grad_norm_(model.parameters(), max_norm=1.0, error_if_nonfinite=True)
            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        if (batch_id + 1) % args.gradient_accumulate_step == 0:
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()
            
        confidence_batch_loss += len(confidence_score_pred) * loss.item()
        ranking_batch_loss += len(confidence_score_pred) * ranking_loss.item()
        confidence_ce_batch_loss += len(confidence_score_pred) * ce_loss.item()
        
        keepNode_less_5_count += keepNode_less_5

        rmsd_list.append(rmsd.detach())
        rmsd_2A_list.append((rmsd.detach() < 2).float())
        rmsd_5A_list.append((rmsd.detach() < 5).float())
        centroid_dis_list.append(centroid_dis.detach())
        centroid_dis_2A_list.append((centroid_dis.detach() < 2).float())
        centroid_dis_5A_list.append((centroid_dis.detach() < 5).float())
        
        batch_len = protein_out_mask_whole.sum(dim=1).detach()
        protein_len_list.append(batch_len)
        # use hard to calculate acc and skip samples
        for i, j in enumerate(batch_len):
            count += 1
            pred_index_bool = (pocket_cls_pred.detach()[i][:j].sigmoid().round().int() == 1)
            if pred_index_bool.sum() == 0: # all the prediction is False, skip
                skip_count += 1

        if batch_id % args.log_interval == 0:
            stats_dict = {}
            stats_dict['step'] = batch_id
            stats_dict['lr'] = optimizer.param_groups[0]['lr']
            stats_dict['ranking_loss'] = ranking_loss.item()
            stats_dict['ce_loss'] = ce_loss.item()
            logger.log_stats(stats_dict, epoch, args, prefix='train')


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

    metrics = {"samples": count, "skip_samples": skip_count, "keepNode < 5": keepNode_less_5_count}
    metrics.update({"lr": optimizer.param_groups[0]['lr']})
    metrics.update({"rmsd": rmsd.mean().item(), "rmsd < 2A": rmsd_2A.mean().item(), "rmsd < 5A": rmsd_5A.mean().item()})
    metrics.update({"rmsd 25%": rmsd_25.item(), "rmsd 50%": rmsd_50.item(), "rmsd 75%": rmsd_75.item()})
    metrics.update({"centroid_dis": centroid_dis.mean().item(), "centroid_dis < 2A": centroid_dis_2A.mean().item(), "centroid_dis < 5A": centroid_dis_5A.mean().item()})
    metrics.update({"centroid_dis 25%": centroid_dis_25.item(), "centroid_dis 50%": centroid_dis_50.item(), "centroid_dis 75%": centroid_dis_75.item()})
    
    metrics.update({"confidence_loss": confidence_batch_loss / len(rmsd)})
    metrics.update({"ranking_loss": ranking_batch_loss / len(rmsd)})
    metrics.update({"confidence_ce_loss": confidence_ce_batch_loss / len(rmsd)})
    metrics.update({"confidence_accuracy": confidence_correct_count / len(rmsd)})
    ranking_accuracy = sum(ranking_accuracy_list) / len(ranking_accuracy_list) if len(ranking_accuracy_list) > 0 else 0.
    metrics.update({"ranking_accuracy": ranking_accuracy})   
    metrics.update({"hit_rate": hit_count / len(rmsd)})
    
    
    return metrics


@torch.no_grad()
def validate(accelerator, args, data_loader, model, compound_confidence_criterion, device, epoch=0, stage=1):
    # basic log
    rmsd_list = []
    rmsd_2A_list = []
    rmsd_5A_list = []
    centroid_dis_list = []
    centroid_dis_2A_list = []
    centroid_dis_5A_list = []
    pdb_list = []
    skip_count = 0
    count = 0
    # for saving mol    
    com_coord_per_sample_list = []
    com_coord_pred_per_sample_list = []
    com_coord_offset_per_sample_list = []
    com_coord_offset_pocket_center_per_sample_list = []
    # sampling-relevant
    ranking_accuracy_list = []
    confidence_pred_score_list = []
    hit_count = 0.0
    confidence_batch_loss = 0.0
    ranking_batch_loss = 0.0
    confidence_ce_batch_loss = 0.0
    confidence_correct_count = 0.0
    keepNode_less_5_count = 0    
    
    n_steps_per_epoch = len(data_loader)
    if args.disable_tqdm:
        data_iter = data_loader
    else:
        data_iter = tqdm(data_loader, mininterval=args.tqdm_interval, disable=not accelerator.is_main_process)
    for batch_id, data in enumerate(data_iter, start=1):
        data = data.to(device)
        com_coord_pred, compound_batch, pocket_cls_pred, protein_out_mask_whole, keepNode_less_5, confidence_score_pred, pocket_center_bias = model(data, stage=stage, train=False)       
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
        
        if not args.confidence_inference:
            # sort confidence_score_pred according to rmsd
            confidence_score_pred = confidence_score_pred.squeeze()
            sorted_ranking_score_pred = confidence_score_pred[rmsd.argsort()]
            sorted_rmsd = rmsd[rmsd.argsort()]
            
            ranking_loss = 0.
            rmsd_2A_score = (rmsd < 2).float()
            
            for i in range(len(sorted_ranking_score_pred)):
                for j in range(i):
                    if args.ranking_loss == 'dynamic_hinge':
                        # dynamic margin
                        gt_rmsd_diff = sorted_rmsd[i] - sorted_rmsd[j] # j is better than i, has lower rmsd
                        ranking_loss += F.relu(gt_rmsd_diff - (sorted_ranking_score_pred[j] - sorted_ranking_score_pred[i]))
                    elif args.ranking_loss == 'logsigmoid':
                        ranking_loss += - F.logsigmoid((sorted_ranking_score_pred[j] - sorted_ranking_score_pred[i]))  # j is better than i, has lower rmsd => score[j] > score[i]

                    ranking_accuracy_list.append(float(sorted_ranking_score_pred[j] > sorted_ranking_score_pred[i]))
                
            ranking_loss = ranking_loss / (len(sorted_ranking_score_pred) * (len(sorted_ranking_score_pred) - 1) / 2)
            if args.keep_cls_2A:
                ce_loss = compound_confidence_criterion(confidence_score_pred, rmsd_2A_score)
                loss = ranking_loss + ce_loss
            else:
                ce_loss = torch.tensor([0], device=accelerator.device)
                loss = ranking_loss
                
            # log hit count when sorted_ranking_score_pred[0] is the largest
            hit_count += sorted_ranking_score_pred[0] > sorted_ranking_score_pred[1:].max()
            confidence_correct_count += ((confidence_score_pred[0] > 0).float() == rmsd_2A_score).sum().item()
        else:
            loss = torch.tensor([0], device=accelerator.device)
            ce_loss = torch.tensor([0], device=accelerator.device)
            ranking_loss = torch.tensor([0], device=accelerator.device)

        confidence_batch_loss += len(confidence_score_pred) * loss.item()
        ranking_batch_loss += len(confidence_score_pred) * ranking_loss.item()
        confidence_ce_batch_loss += len(confidence_score_pred) * ce_loss.item()

        keepNode_less_5_count += keepNode_less_5
        rmsd_list.append(rmsd.detach())
        rmsd_2A_list.append((rmsd.detach() < 2).float())
        rmsd_5A_list.append((rmsd.detach() < 5).float())
        centroid_dis_list.append(centroid_dis.detach())
        centroid_dis_2A_list.append((centroid_dis.detach() < 2).float())
        centroid_dis_5A_list.append((centroid_dis.detach() < 5).float())

        batch_len = protein_out_mask_whole.sum(dim=1).detach()
        # protein_len_list.append(batch_len)
        for i, j in enumerate(batch_len):
            count += 1
            pdb_list.append(data.pdb[i])
            com_coord_offset_per_sample_list.append(data.coord_offset[i])
            com_coord_offset_pocket_center_per_sample_list.append(pocket_center_bias[i])
            pred_index_bool = (pocket_cls_pred.detach()[i][:j].sigmoid().round().int() == 1)
            if pred_index_bool.sum() == 0:
                skip_count += 1
                
        if args.confidence_inference:
            confidence_pred_score_list.append(confidence_score_pred.detach())
        else:
            confidence_pred_score_list.append(torch.tensor([0], device=accelerator.device))
        

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
    confidence_pred_score = torch.cat(confidence_pred_score_list)

    
    if args.save_rmsd_dir is not None and args.confidence_inference:
        if not os.path.exists(args.save_rmsd_dir):
            os.system(f'mkdir -p {args.save_rmsd_dir}')
        for i in range(len(pdb_list)):
            pdb = pdb_list[i]
            rmsd_i = rmsd[i]
            centroid_dis_i = centroid_dis[i]
            confidence_pred_score_i = confidence_pred_score[i]
            with open(os.path.join(args.save_rmsd_dir, f'epoch{epoch}_confidence.txt'), 'a') as f:
                f.write(f'{pdb} {rmsd_i.item()} {centroid_dis_i.item()} {confidence_pred_score_i.item()}\n')
    
    metrics = {"samples": count, "skip_samples": skip_count, "keepNode < 5": keepNode_less_5_count}
    # Final evaluation metrics
    metrics.update({"rmsd": rmsd.mean().item(), "rmsd < 2A": rmsd_2A.mean().item(), "rmsd < 5A": rmsd_5A.mean().item()})
    metrics.update({"rmsd 25%": rmsd_25.item(), "rmsd 50%": rmsd_50.item(), "rmsd 75%": rmsd_75.item()})
    metrics.update({"centroid_dis": centroid_dis.mean().item(), "centroid_dis < 2A": centroid_dis_2A.mean().item(), "centroid_dis < 5A": centroid_dis_5A.mean().item()})
    metrics.update({"centroid_dis 25%": centroid_dis_25.item(), "centroid_dis 50%": centroid_dis_50.item(), "centroid_dis 75%": centroid_dis_75.item()})
    
    metrics.update({"confidence_loss": confidence_batch_loss / len(rmsd)})
    metrics.update({"ranking_loss": ranking_batch_loss / len(rmsd)})
    metrics.update({"confidence_ce_loss": confidence_ce_batch_loss / len(rmsd)})
    metrics.update({"confidence_accuracy": confidence_correct_count / len(rmsd)})
    ranking_accuracy = sum(ranking_accuracy_list) / len(ranking_accuracy_list) if len(ranking_accuracy_list) > 0 else 0.
    metrics.update({"ranking_accuracy": ranking_accuracy})   
    metrics.update({"hit_rate": hit_count / len(rmsd)})


    if args.write_mol_to_file is not None:
        os.system('mkdir -p {}'.format(os.path.join(args.write_mol_to_file, str(epoch))))
        
        compound_folder = os.path.join(args.data_path, 'renumber_atom_index_same_as_smiles')
        df_pdb_rmsd = pd.DataFrame({'pdb': pdb_list, 'rmsd': rmsd.detach().cpu()})
        df_pdb_rmsd.to_pickle(os.path.join(args.write_mol_to_file, str(epoch), 'info.pkl'))
        for i in range(len(pdb_list)):
            pdb = pdb_list[i]
            com_coord_pred_i = com_coord_pred_per_sample_list[i]
            # com_coord_i = com_coord_per_sample_list[i]
            coord_offset_i = com_coord_offset_per_sample_list[i]
            pocket_center_offset_i = com_coord_offset_pocket_center_per_sample_list[i]
            com_coord_pred_i += coord_offset_i + pocket_center_offset_i
            if os.path.exists(os.path.join(compound_folder,  f"{pdb}.sdf")):
                ligand_original_file_sdf = os.path.join(compound_folder,  f"{pdb}.sdf")
                output_file = os.path.join(args.write_mol_to_file, str(epoch), f"{pdb}.sdf")
            else:
                raise ValueError(f"Cannot find {pdb}.sdf in {compound_folder}")
            mol = write_mol(reference_file=ligand_original_file_sdf, coords=com_coord_pred_i, output_file=output_file)
    
    return metrics

