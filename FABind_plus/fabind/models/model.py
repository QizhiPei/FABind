import torch
from torch_geometric.utils import to_dense_batch
from torch import nn
import torch.nn as nn
from models.att_model import EfficientMCAttModel
import torch.nn.functional as F
from utils.utils import get_keepNode_tensor, gumbel_softmax_no_random
import random
from models.model_utils import MLP, MLPwithLastAct, MLP4Confidence
from sklearn.cluster import DBSCAN
from torch_scatter import scatter_max, scatter_add

class FABindPlus(torch.nn.Module):
    def __init__(self, args, embedding_channels=128, pocket_pred_embedding_channels=128):
        super().__init__()
        self.args = args
        self.coordinate_scale = args.coordinate_scale
        self.normalize_coord = lambda x: x / self.coordinate_scale
        self.unnormalize_coord = lambda x: x * self.coordinate_scale

        # global nodes for protein / compound
        self.glb_c = nn.Parameter(torch.ones(1, embedding_channels))
        self.glb_p = nn.Parameter(torch.ones(1, embedding_channels))
        protein_hidden = 1280 # hard-coded for ESM2 feature
        compound_hidden = 56  # hard-coded for hand-crafted feature

        self.protein_linear_whole_protein = nn.Linear(protein_hidden, embedding_channels)
        self.compound_linear_whole_protein = nn.Linear(compound_hidden, embedding_channels)
        self.embedding_shrink = nn.Linear(embedding_channels, pocket_pred_embedding_channels)
        self.embedding_enlarge = nn.Linear(pocket_pred_embedding_channels, embedding_channels)
        
        n_channel = 1 # ligand node has only one coordinate dimension.
        self.pocket_pred_model = EfficientMCAttModel(
            args, pocket_pred_embedding_channels, pocket_pred_embedding_channels, n_channel, n_edge_feats=0, n_layers=args.pocket_pred_layers, n_iter=args.pocket_pred_n_iter,
            inter_cutoff=args.inter_cutoff, intra_cutoff=args.intra_cutoff, normalize_coord=self.normalize_coord, unnormalize_coord=self.unnormalize_coord,
            )
        self.pocket_radius_head = MLP(args, embedding_channels=embedding_channels, n=self.args.mlp_hidden_scale, out_channels=1)
        self.protein_to_pocket = MLP(args, embedding_channels=embedding_channels, n=self.args.mlp_hidden_scale, out_channels=1)

        self.complex_model = EfficientMCAttModel(
            args, embedding_channels, embedding_channels, n_channel, n_edge_feats=0, n_layers=args.mean_layers, n_iter=args.n_iter,
            inter_cutoff=args.inter_cutoff, intra_cutoff=args.intra_cutoff, normalize_coord=self.normalize_coord, unnormalize_coord=self.unnormalize_coord,
        )
        self.distmap_mlp = MLP(args, embedding_channels=embedding_channels, n=self.args.mlp_hidden_scale, out_channels=1)
        
        
        torch.nn.init.xavier_uniform_(self.protein_linear_whole_protein.weight, gain=0.001)
        torch.nn.init.xavier_uniform_(self.compound_linear_whole_protein.weight, gain=0.001)
        torch.nn.init.xavier_uniform_(self.embedding_shrink.weight, gain=0.001)
        torch.nn.init.xavier_uniform_(self.embedding_enlarge.weight, gain=0.001)
        
        self.confidence_training = getattr(args, 'confidence_training', False)
        if self.confidence_training:
            if self.args.stack_mlp:
                self.ranking_mlp_pre = MLP4Confidence(args, embedding_channels=embedding_channels, n=args.confidence_mlp_hidden_scale, out_channels=embedding_channels)
            self.ranking_score_mlp = MLP4Confidence(args, embedding_channels=embedding_channels, n=args.confidence_mlp_hidden_scale, out_channels=1)
            
            if self.args.use_clustering:
                import mlflow
                mlflow.sklearn.autolog(disable=True)
                self.dbscan_module = DBSCAN(eps=self.args.dbscan_eps, min_samples=self.args.dbscan_min_samples)

    def forward(self, data, stage=2, train=False):
        keepNode_less_5 = 0
        compound_batch = data['compound'].batch
        pocket_batch = data['pocket'].batch
        complex_batch = data['complex'].batch
        protein_batch_whole = data['protein_whole'].batch
        complex_batch_whole_protein = data['complex_whole_protein'].batch

        # Pocket Prediction
        batched_complex_coord_whole_protein = self.normalize_coord(data['complex_whole_protein'].node_coords.unsqueeze(-2))
        batched_complex_coord_LAS_whole_protein = self.normalize_coord(data['complex_whole_protein'].node_coords_LAS.unsqueeze(-2))
        batched_compound_emb_whole_protein = self.compound_linear_whole_protein(data['compound'].node_feats)
        batched_protein_emb_whole_protein = self.protein_linear_whole_protein(data['protein_whole'].node_feats)

        for i in range(complex_batch_whole_protein.max()+1):
            if i == 0:
                new_samples_whole_protein = torch.cat((
                    self.glb_c, batched_compound_emb_whole_protein[compound_batch==i], 
                    self.glb_p, batched_protein_emb_whole_protein[protein_batch_whole==i]
                    ), dim=0)
            else:
                new_sample_whole_protein = torch.cat((
                    self.glb_c, batched_compound_emb_whole_protein[compound_batch==i], 
                    self.glb_p, batched_protein_emb_whole_protein[protein_batch_whole==i]
                    ), dim=0)
                new_samples_whole_protein = torch.cat((new_samples_whole_protein, new_sample_whole_protein), dim=0)

        new_samples_whole_protein = self.embedding_shrink(new_samples_whole_protein)

        _, complex_out_whole_protein, _ = self.pocket_pred_model(
            batched_complex_coord_whole_protein,
            new_samples_whole_protein, 
            batch_id=complex_batch_whole_protein, 
            segment_id=data['complex_whole_protein'].segment,
            mask=data['complex_whole_protein'].mask, 
            is_global=data['complex_whole_protein'].is_global,
            compound_edge_index=data['complex_whole_protein', 'c2c', 'complex_whole_protein'].edge_index,
            LAS_edge_index=data['complex_whole_protein', 'LAS', 'complex_whole_protein'].edge_index,
            batched_complex_coord_LAS=batched_complex_coord_LAS_whole_protein,
            LAS_mask=None
        )

        complex_out_whole_protein = self.embedding_enlarge(complex_out_whole_protein)

        compound_flag_whole_protein = torch.logical_and(data['complex_whole_protein'].segment == 0, ~data['complex_whole_protein'].is_global)
        compound_out_whole_protein = complex_out_whole_protein[compound_flag_whole_protein]
                
        # pocket radius predict
        if self.args.use_for_radius_pred == 'ligand':
            compound_in_complex_whole_batch = complex_batch_whole_protein[compound_flag_whole_protein]
            compound_emb_batch, compound_emb_mask = to_dense_batch(compound_out_whole_protein, compound_in_complex_whole_batch)
            pocket_radius_pred = self.pocket_radius_head(compound_emb_batch.sum(dim=1)).relu() # [B, 1]
        elif self.args.use_for_radius_pred == 'both':
            complex_out_whole_protein_batched, _ = to_dense_batch(complex_out_whole_protein, complex_batch_whole_protein)
            pocket_radius_pred = self.pocket_radius_head(complex_out_whole_protein_batched.sum(dim=1)).relu() # [B, 1]
        elif self.args.use_for_radius_pred == 'global':
            glc_flag_whole_protein = torch.logical_and(data['complex_whole_protein'].segment == 0, data['complex_whole_protein'].is_global)
            glc_out_whole_protein = complex_out_whole_protein[glc_flag_whole_protein]
            pocket_radius_pred = self.pocket_radius_head(glc_out_whole_protein).relu()

        protein_flag_whole_protein = torch.logical_and(data['complex_whole_protein'].segment == 1, ~data['complex_whole_protein'].is_global)
        protein_out_whole_protein = complex_out_whole_protein[protein_flag_whole_protein]
        protein_out_batched_whole, protein_out_mask_whole = to_dense_batch(protein_out_whole_protein, protein_batch_whole)
        pocket_cls_pred = self.protein_to_pocket(protein_out_batched_whole)
        pocket_cls_pred = pocket_cls_pred.squeeze(-1) * protein_out_mask_whole
        pocket_cls, _ = to_dense_batch(data.pocket_idx, protein_batch_whole)

        pocket_coords_batched, _ = to_dense_batch(self.normalize_coord(data.node_xyz), pocket_batch)
        protein_coords_batched_whole, protein_coords_mask_whole = to_dense_batch(data.node_xyz_whole, protein_batch_whole)
  
        pred_index_true = pocket_cls_pred.sigmoid().unsqueeze(-1)
        pred_index_false = 1. - pred_index_true
        pred_index_prob = torch.cat([pred_index_false, pred_index_true], dim=-1)
        # For training stability
        pred_index_prob = torch.clamp(pred_index_prob, min=1e-6, max=1-1e-6)
        pred_index_log_prob = torch.log(pred_index_prob)
        if self.pocket_pred_model.training:
            pred_index_one_hot = F.gumbel_softmax(pred_index_log_prob, tau=self.args.gs_tau, hard=self.args.gs_hard)
        else:
            pred_index_one_hot = gumbel_softmax_no_random(pred_index_log_prob, tau=self.args.gs_tau, hard=self.args.gs_hard)
        pred_index_one_hot_true = (pred_index_one_hot[:, :, 1] * protein_out_mask_whole).unsqueeze(-1)
        pred_pocket_center_gumbel = pred_index_one_hot_true * protein_coords_batched_whole
        pred_pocket_center = pred_pocket_center_gumbel.sum(dim=1) / pred_index_one_hot_true.sum(dim=1)

        if self.args.use_clustering:
            centers = torch.zeros_like(pred_pocket_center)
            for i in range(pred_pocket_center.shape[0]):            
                positive_prob = pred_index_true[i].squeeze()
                positive_residue_coords = protein_coords_batched_whole[i][positive_prob > 0.5]
                if positive_residue_coords.shape[0] < 50:
                    top_50_indices = torch.argsort(positive_prob)[-50:]
                    bool_tensor = torch.full(positive_prob.shape, False)
                    bool_tensor[top_50_indices] = True
                    positive_residue_coords = protein_coords_batched_whole[i][bool_tensor]
                positive_residue_coords = positive_residue_coords.detach().cpu().numpy()
                clustering = self.dbscan_module.fit(positive_residue_coords)
                min_cluster_mae = 100.
                # sample a integer from 0 to clustering.labels_.max() in the next line
                random_cluster_id = random.randint(0, clustering.labels_.max())
                if random.random() < self.args.choose_cluster_prob:
                    center_array = positive_residue_coords[clustering.labels_==random_cluster_id].mean(axis=0)
                    centers[i] = torch.tensor(center_array, device=compound_batch.device)
                else:
                    centers[i] = pred_pocket_center[i]
            pred_pocket_center = centers


        if train or stage==1: # teacher forcing when training, while use predicted pocket for docking when testing
            pocket_center_bias = torch.zeros_like(pred_pocket_center, device=compound_batch.device)
            
            batched_compound_emb = compound_out_whole_protein
            batched_pocket_emb = protein_out_whole_protein[data['pocket'].keepNode]
            
            for i in range(complex_batch.max()+1):
                num_compound_atoms = data['compound'].node_feats[compound_batch==i].shape[0]
                temp_coords = data['complex'].node_coords[complex_batch==i]
                temp_coords[1:num_compound_atoms+1] = temp_coords[1:num_compound_atoms+1] - temp_coords[1:num_compound_atoms+1].mean(dim=0)
                temp_coords[num_compound_atoms+2:] = temp_coords[num_compound_atoms+2:] - data.pocket_residue_center[i].unsqueeze(0) # move pocket to origin
                data['complex'].node_coords[complex_batch==i] = temp_coords
                
                data.coords[compound_batch==i] = data.coords[compound_batch==i] - data.pocket_residue_center[i].unsqueeze(0)
                if i == 0:
                    new_samples = torch.cat((
                        self.glb_c, batched_compound_emb[compound_batch==i], 
                        self.glb_p, batched_pocket_emb[pocket_batch==i]
                        ), dim=0)
                else:
                    new_sample = torch.cat((
                        self.glb_c, batched_compound_emb[compound_batch==i], 
                        self.glb_p, batched_pocket_emb[pocket_batch==i]
                        ), dim=0)
                    new_samples = torch.cat((new_samples, new_sample), dim=0)
            dis_map = data.dis_map

            batched_complex_coord = self.normalize_coord(data['complex'].node_coords.unsqueeze(-2))
            batched_complex_coord_LAS = self.normalize_coord(data['complex'].node_coords_LAS.unsqueeze(-2))

        else:
            # Replace raw feature with pocket prediction output
            # batched_compound_emb = self.compound_linear(data['compound'].node_feats)
            batched_compound_emb = compound_out_whole_protein
            # keepNode_batch = torch.tensor([], device=compound_batch.device)
            data['complex'].node_coords = torch.tensor([], device=compound_batch.device)
            data['complex'].node_coords_LAS = torch.tensor([], device=compound_batch.device)
            data['complex'].segment = torch.tensor([], device=compound_batch.device)
            data['complex'].mask = torch.tensor([], device=compound_batch.device)
            data['complex'].is_global = torch.tensor([], device=compound_batch.device)
            complex_batch = torch.tensor([], device=compound_batch.device)
            pocket_batch = torch.tensor([], device=compound_batch.device)
            data['complex', 'c2c', 'complex'].edge_index = torch.tensor([], device=compound_batch.device)
            data['complex', 'LAS', 'complex'].edge_index = torch.tensor([], device=compound_batch.device)
            pocket_coords_concats = torch.tensor([], device=compound_batch.device)
            dis_map = torch.tensor([], device=compound_batch.device)
            pocket_center_bias = torch.zeros_like(pred_pocket_center, device=compound_batch.device)

            if self.args.train_pred_pocket_noise and train:
                pred_pocket_center += self.args.train_pred_pocket_noise * (2 * torch.rand_like(pred_pocket_center) - 1)

            for i in range(pred_pocket_center.shape[0]):
                protein_i = data.node_xyz_whole[protein_batch_whole==i].detach()
                if self.args.pocket_radius_buffer <= 2.0:
                    pocket_radius = (pocket_radius_pred[i] * self.args.pocket_radius_buffer).item()
                else:
                    pocket_radius = (pocket_radius_pred[i] + self.args.pocket_radius_buffer).item()
                if pocket_radius < self.args.min_pocket_radius:
                    pocket_radius = self.args.min_pocket_radius
                if self.args.force_fix_radius:
                    pocket_radius = self.args.pocket_radius
                keepNode = get_keepNode_tensor(protein_i, pocket_radius, None, pred_pocket_center[i].detach())
                # TODO Check the case
                if keepNode.sum() < 5:
                    # if only include less than 5 residues, simply add first 100 residues.
                    keepNode[:100] = True
                    keepNode_less_5 += 1
                pocket_emb = protein_out_batched_whole[i][protein_out_mask_whole[i]][keepNode]
                # node emb
                if i == 0:
                    new_samples = torch.cat((
                        self.glb_c, batched_compound_emb[compound_batch==i], 
                        self.glb_p, pocket_emb
                        ), dim=0)
                else:
                    new_sample = torch.cat((
                        self.glb_c, batched_compound_emb[compound_batch==i], 
                        self.glb_p, pocket_emb
                        ), dim=0)
                    new_samples = torch.cat((new_samples, new_sample), dim=0)

                # Node coords. 
                # Ligand coords are initialized at pocket center with rdkit random conformation.
                # Pocket coords are from origin protein coords.
                pocket_coords = protein_coords_batched_whole[i][protein_coords_mask_whole[i]][keepNode]
                pocket_coords_center = pocket_coords.mean(dim=0).reshape(1, 3)
                pocket_coords = pocket_coords - pocket_coords_center
                data.coords[compound_batch==i] = data.coords[compound_batch==i] - pocket_coords_center
                pocket_center_bias[i] = pocket_coords_center.squeeze()
                pocket_coords_concats = torch.cat((pocket_coords_concats, pocket_coords), dim=0)
                
                data['complex'].node_coords = torch.cat( # [glb_c || compound || glb_p || protein]
                    (
                        data['complex'].node_coords,
                        torch.zeros((1, 3), device=compound_batch.device),
                        data['compound'].node_coords[compound_batch==i] - data['compound'].node_coords[compound_batch==i].mean(dim=0).reshape(1, 3) + pocket_coords.mean(dim=0).reshape(1, 3),
                        torch.zeros((1, 3), device=compound_batch.device), 
                        pocket_coords,
                    ), dim=0
                ).float()


                data['complex'].node_coords_LAS = torch.cat( # [glb_c || compound || glb_p || protein]
                    (
                        data['complex'].node_coords_LAS,
                        torch.zeros((1, 3), device=compound_batch.device),
                        data['compound'].rdkit_coords[compound_batch==i],
                        torch.zeros((1, 3), device=compound_batch.device), 
                        torch.zeros_like(pocket_coords)
                    ), dim=0
                ).float()

                # masks
                n_protein = pocket_emb.shape[0]
                n_compound = batched_compound_emb[compound_batch==i].shape[0]
                segment = torch.zeros((n_protein + n_compound + 2), device=complex_batch.device)
                segment[n_compound+1:] = 1 # compound: 0, protein: 1
                data['complex'].segment = torch.cat((data['complex'].segment, segment), dim=0) # protein or ligand
                mask = torch.zeros((n_protein + n_compound + 2), device=complex_batch.device)
                mask[:n_compound+2] = 1 # glb_p can be updated
                data['complex'].mask = torch.cat((data['complex'].mask, mask.bool()), dim=0)
                is_global = torch.zeros((n_protein + n_compound + 2), device=complex_batch.device)
                is_global[0] = 1
                is_global[n_compound+1] = 1
                data['complex'].is_global = torch.cat((data['complex'].is_global, is_global.bool()), dim=0)

                # edge_index
                data['complex', 'c2c', 'complex'].edge_index = torch.cat(
                    (
                        data['complex', 'c2c', 'complex'].edge_index, 
                        data['compound_atom_edge_list'].x[data['compound_atom_edge_list'].batch==i].t() + complex_batch.shape[0]
                    ), dim=1)
                data['complex', 'LAS', 'complex'].edge_index = torch.cat(
                    (
                        data['complex', 'LAS', 'complex'].edge_index, 
                        data['LAS_edge_list'].x[data['LAS_edge_list'].batch==i].t() + complex_batch.shape[0]
                    ), dim=1)
                
                # batch_id
                complex_batch = torch.cat((complex_batch, torch.ones((n_compound + n_protein + 2), device=compound_batch.device)*i), dim=0)
                pocket_batch = torch.cat((pocket_batch, torch.ones((n_protein), device=compound_batch.device)*i), dim=0)

                # distance map
                dis_map_i = torch.cdist(pocket_coords, data['compound'].node_coords[compound_batch==i].to(torch.float32)-pocket_coords_center).flatten()
                dis_map_i[dis_map_i>self.args.dis_map_thres] = self.args.dis_map_thres
                dis_map = torch.cat((dis_map, dis_map_i), dim=0)

            # construct inputs
            batched_complex_coord = self.normalize_coord(data['complex'].node_coords.unsqueeze(-2))
            batched_complex_coord_LAS = self.normalize_coord(data['complex'].node_coords_LAS.unsqueeze(-2))
            complex_batch = complex_batch.to(torch.int64)
            pocket_batch = pocket_batch.to(torch.int64)
            pocket_coords_batched, _ = to_dense_batch(self.normalize_coord(pocket_coords_concats), pocket_batch)
            data['complex', 'c2c', 'complex'].edge_index = data['complex', 'c2c', 'complex'].edge_index.to(torch.int64)
            data['complex', 'LAS', 'complex'].edge_index = data['complex', 'LAS', 'complex'].edge_index.to(torch.int64)
            data['complex'].segment = data['complex'].segment.to(torch.bool)
            data['complex'].mask = data['complex'].mask.to(torch.bool)
            data['complex'].is_global = data['complex'].is_global.to(torch.bool)

        

        complex_coords, complex_out, pair_embed_batched = self.complex_model(
            batched_complex_coord,
            new_samples, 
            batch_id=complex_batch, 
            segment_id=data['complex'].segment,
            mask=data['complex'].mask, 
            is_global=data['complex'].is_global,
            compound_edge_index=data['complex', 'c2c', 'complex'].edge_index,
            LAS_edge_index=data['complex', 'LAS', 'complex'].edge_index,
            batched_complex_coord_LAS=batched_complex_coord_LAS,
            LAS_mask=None
        )
        
        if self.args.only_last_LAS:
            LAS_edge_list = data['complex', 'LAS', 'complex'].edge_index
            complex_coords.squeeze_(1)
            batched_complex_coord_LAS.squeeze_(1)
            for step in range(self.args.geom_reg_steps):
                LAS_cur_squared = torch.sum(
                    (complex_coords[LAS_edge_list[0]] - complex_coords[LAS_edge_list[1]]) ** 2, dim=1)
                LAS_true_squared = torch.sum(
                    (batched_complex_coord_LAS[LAS_edge_list[0]] - batched_complex_coord_LAS[LAS_edge_list[1]]) ** 2,
                    dim=1)
                grad_squared = 2 * (complex_coords[LAS_edge_list[0]] - complex_coords[LAS_edge_list[1]])
                LAS_force = 2 * (LAS_cur_squared - LAS_true_squared)[:, None] * grad_squared
                LAS_delta_coord = scatter_add(src=LAS_force, index=LAS_edge_list[1], dim=0,
                                            dim_size=complex_coords.shape[0])

                complex_coords = complex_coords + (LAS_delta_coord * self.args.geometry_reg_step_size) \
                    .clamp(min=self.normalize_coord(-2), max=self.normalize_coord(2))
            complex_coords.unsqueeze_(1)


        compound_flag = torch.logical_and(data['complex'].segment == 0, ~data['complex'].is_global)
        protein_flag  = torch.logical_and(data['complex'].segment == 1, ~data['complex'].is_global)
        pocket_out  = complex_out[protein_flag]
        compound_out = complex_out[compound_flag]
        compound_coords_out = complex_coords[compound_flag].squeeze(-2)

        # pocket_batch version could further process b matrix. better than for loop.
        # pocket_out_batched of shape (b, n, c). to_dense_batch is torch geometric function.
        _, pocket_out_mask = to_dense_batch(pocket_out, pocket_batch)
        _, compound_out_mask = to_dense_batch(compound_out, compound_batch)
        compound_coords_out_batched, _ = to_dense_batch(compound_coords_out, compound_batch)

        # get the pair distance of protein and compound
        pocket_com_dis_map = torch.cdist(pocket_coords_batched, compound_coords_out_batched)
        
        z = pair_embed_batched[:, 1:, 1:, ...]
        z_mask = torch.einsum("bi,bj->bij", pocket_out_mask, compound_out_mask)
        
        b = self.distmap_mlp(z).squeeze(-1)
        y_pred = b[z_mask]
        y_pred = y_pred.sigmoid() * self.args.dis_map_thres   # normalize to 0 to 10.

        y_pred_by_coords = pocket_com_dis_map[z_mask]
        y_pred_by_coords = self.unnormalize_coord(y_pred_by_coords)
        y_pred_by_coords = torch.clamp(y_pred_by_coords, 0, self.args.dis_map_thres)
                
        compound_coords_out = self.unnormalize_coord(compound_coords_out)
        
        
        if self.confidence_training:
            if self.args.stack_mlp:
                confidence_hidden = self.ranking_mlp_pre(scatter_add(src=complex_out, index=complex_batch, dim=0)).relu()
            else:
                confidence_hidden = scatter_add(src=complex_out, index=complex_batch, dim=0)
            confidence_score_pred = self.ranking_score_mlp(confidence_hidden).squeeze(-1)
            return compound_coords_out, compound_batch, pocket_cls_pred, protein_out_mask_whole, keepNode_less_5, confidence_score_pred, pocket_center_bias
        else:
            return compound_coords_out, compound_batch, y_pred, y_pred_by_coords, pocket_cls_pred, pocket_cls, protein_out_mask_whole, protein_coords_batched_whole, pred_pocket_center, dis_map, keepNode_less_5, pocket_radius_pred, pocket_center_bias
        

    def inference(self, data):
        compound_batch = data['compound'].batch
        protein_batch_whole = data['protein_whole'].batch
        complex_batch_whole_protein = data['complex_whole_protein'].batch

        # Pocket Prediction
        batched_complex_coord_whole_protein = self.normalize_coord(data['complex_whole_protein'].node_coords.unsqueeze(-2))
        batched_complex_coord_LAS_whole_protein = self.normalize_coord(data['complex_whole_protein'].node_coords_LAS.unsqueeze(-2))
        batched_compound_emb_whole_protein = self.compound_linear_whole_protein(data['compound'].node_feats)
        batched_protein_emb_whole_protein = self.protein_linear_whole_protein(data['protein_whole'].node_feats)

        # TODO self.glb_c and self.glb_p shared?
        for i in range(complex_batch_whole_protein.max()+1):
            if i == 0:
                new_samples_whole_protein = torch.cat((
                    self.glb_c, batched_compound_emb_whole_protein[compound_batch==i], 
                    self.glb_p, batched_protein_emb_whole_protein[protein_batch_whole==i]
                    ), dim=0)
            else:
                new_sample_whole_protein = torch.cat((
                    self.glb_c, batched_compound_emb_whole_protein[compound_batch==i], 
                    self.glb_p, batched_protein_emb_whole_protein[protein_batch_whole==i]
                    ), dim=0)
                new_samples_whole_protein = torch.cat((new_samples_whole_protein, new_sample_whole_protein), dim=0)
        
        new_samples_whole_protein = self.embedding_shrink(new_samples_whole_protein)

        _, complex_out_whole_protein, _ = self.pocket_pred_model(
            batched_complex_coord_whole_protein,
            new_samples_whole_protein, 
            batch_id=complex_batch_whole_protein, 
            segment_id=data['complex_whole_protein'].segment,
            mask=data['complex_whole_protein'].mask, 
            is_global=data['complex_whole_protein'].is_global,
            compound_edge_index=data['complex_whole_protein', 'c2c', 'complex_whole_protein'].edge_index,
            LAS_edge_index=data['complex_whole_protein', 'LAS', 'complex_whole_protein'].edge_index,
            batched_complex_coord_LAS=batched_complex_coord_LAS_whole_protein,
            LAS_mask=None
        )

        complex_out_whole_protein = self.embedding_enlarge(complex_out_whole_protein)


        compound_flag_whole_protein = torch.logical_and(data['complex_whole_protein'].segment == 0, ~data['complex_whole_protein'].is_global)
        compound_out_whole_protein = complex_out_whole_protein[compound_flag_whole_protein]
                
        # pocket radius predict
        if self.args.use_for_radius_pred == 'ligand':
            compound_in_complex_whole_batch = complex_batch_whole_protein[compound_flag_whole_protein]
            compound_emb_batch, compound_emb_mask = to_dense_batch(compound_out_whole_protein, compound_in_complex_whole_batch)
            pocket_radius_pred = self.pocket_radius_head(compound_emb_batch.sum(dim=1)).relu() # [B, 1]
        elif self.args.use_for_radius_pred == 'both':
            complex_out_whole_protein_batched, _ = to_dense_batch(complex_out_whole_protein, complex_batch_whole_protein)
            pocket_radius_pred = self.pocket_radius_head(complex_out_whole_protein_batched.sum(dim=1)).relu() # [B, 1]
        elif self.args.use_for_radius_pred == 'global':
            glc_flag_whole_protein = torch.logical_and(data['complex_whole_protein'].segment == 0, data['complex_whole_protein'].is_global)
            glc_out_whole_protein = complex_out_whole_protein[glc_flag_whole_protein]
            pocket_radius_pred = self.pocket_radius_head(glc_out_whole_protein).relu()

        protein_flag_whole_protein = torch.logical_and(data['complex_whole_protein'].segment == 1, ~data['complex_whole_protein'].is_global)
        protein_out_whole_protein = complex_out_whole_protein[protein_flag_whole_protein]
        protein_out_batched_whole, protein_out_mask_whole = to_dense_batch(protein_out_whole_protein, protein_batch_whole)
        pocket_cls_pred = self.protein_to_pocket(protein_out_batched_whole)
        pocket_cls_pred = pocket_cls_pred.squeeze(-1) * protein_out_mask_whole

        protein_coords_batched_whole, protein_coords_mask_whole = to_dense_batch(data.node_xyz_whole, protein_batch_whole)
  
        pred_index_true = pocket_cls_pred.sigmoid().unsqueeze(-1)
        pred_index_false = 1. - pred_index_true
        pred_index_prob = torch.cat([pred_index_false, pred_index_true], dim=-1)
        # For training stability
        pred_index_prob = torch.clamp(pred_index_prob, min=1e-6, max=1-1e-6)
        pred_index_log_prob = torch.log(pred_index_prob)
        pred_index_one_hot = gumbel_softmax_no_random(pred_index_log_prob, tau=self.args.gs_tau, hard=self.args.gs_hard)
        pred_index_one_hot_true = (pred_index_one_hot[:, :, 1] * protein_out_mask_whole).unsqueeze(-1)
        pred_pocket_center_gumbel = pred_index_one_hot_true * protein_coords_batched_whole
        pred_pocket_center = pred_pocket_center_gumbel.sum(dim=1) / pred_index_one_hot_true.sum(dim=1)

        if self.args.use_clustering:
            centers = torch.zeros_like(pred_pocket_center)
            for i in range(pred_pocket_center.shape[0]):            
                positive_prob = pred_index_true[i].squeeze()
                positive_residue_coords = protein_coords_batched_whole[i][positive_prob > 0.5]
                if positive_residue_coords.shape[0] < 50:
                    top_50_indices = torch.argsort(positive_prob)[-50:]
                    bool_tensor = torch.full(positive_prob.shape, False)
                    bool_tensor[top_50_indices] = True
                    positive_residue_coords = protein_coords_batched_whole[i][bool_tensor]
                positive_residue_coords = positive_residue_coords.detach().cpu().numpy()
                clustering = self.dbscan_module.fit(positive_residue_coords)
                min_cluster_mae = 100.
                # sample a integer from 0 to clustering.labels_.max() in the next line
                random_cluster_id = random.randint(0, clustering.labels_.max())
                if random.random() < self.args.choose_cluster_prob:
                    center_array = positive_residue_coords[clustering.labels_==random_cluster_id].mean(axis=0)
                    centers[i] = torch.tensor(center_array, device=compound_batch.device)
                else:
                    centers[i] = pred_pocket_center[i]
            pred_pocket_center = centers

        # Replace raw feature with pocket prediction output
        # batched_compound_emb = self.compound_linear(data['compound'].node_feats)
        batched_compound_emb = compound_out_whole_protein
        # keepNode_batch = torch.tensor([], device=compound_batch.device)
        data['complex'].node_coords = torch.tensor([], device=compound_batch.device)
        data['complex'].node_coords_LAS = torch.tensor([], device=compound_batch.device)
        data['complex'].segment = torch.tensor([], device=compound_batch.device)
        data['complex'].mask = torch.tensor([], device=compound_batch.device)
        data['complex'].is_global = torch.tensor([], device=compound_batch.device)
        complex_batch = torch.tensor([], device=compound_batch.device)
        pocket_batch = torch.tensor([], device=compound_batch.device)
        data['complex', 'c2c', 'complex'].edge_index = torch.tensor([], device=compound_batch.device)
        data['complex', 'LAS', 'complex'].edge_index = torch.tensor([], device=compound_batch.device)
        pocket_coords_concats = torch.tensor([], device=compound_batch.device)
        pocket_center_bias = torch.zeros_like(pred_pocket_center)

        for i in range(pred_pocket_center.shape[0]):
            protein_i = data.node_xyz_whole[protein_batch_whole==i].detach()
            if self.args.pocket_radius_buffer <= 2.0:
                pocket_radius = (pocket_radius_pred[i] * self.args.pocket_radius_buffer).item()
            else:
                pocket_radius = (pocket_radius_pred[i] + self.args.pocket_radius_buffer).item()
            if pocket_radius < self.args.min_pocket_radius:
                pocket_radius = self.args.min_pocket_radius

            if self.args.force_fix_radius:
                pocket_radius = self.args.pocket_radius
        
            keepNode = get_keepNode_tensor(protein_i, pocket_radius, None, pred_pocket_center[i].detach())
            # TODO Check the case
            if keepNode.sum() < 5:
                # if only include less than 5 residues, simply add first 100 residues.
                keepNode[:100] = True
            pocket_emb = protein_out_batched_whole[i][protein_out_mask_whole[i]][keepNode]
            # node emb
            if i == 0:
                new_samples = torch.cat((
                    self.glb_c, batched_compound_emb[compound_batch==i], 
                    self.glb_p, pocket_emb
                    ), dim=0)
            else:
                new_sample = torch.cat((
                    self.glb_c, batched_compound_emb[compound_batch==i], 
                    self.glb_p, pocket_emb
                    ), dim=0)
                new_samples = torch.cat((new_samples, new_sample), dim=0)

            # Node coords. 
            # Ligand coords are initialized at pocket center with rdkit random conformation.
            # Pocket coords are from origin protein coords.
            pocket_coords = protein_coords_batched_whole[i][protein_coords_mask_whole[i]][keepNode]
            pocket_coords_center = pocket_coords.mean(dim=0).reshape(1, 3)
            pocket_coords = pocket_coords - pocket_coords_center
            pocket_center_bias[i] = pocket_coords_center.squeeze()
            # data.coords[compound_batch==i] = data.coords[compound_batch==i] - pocket_coords_center    
            pocket_coords_concats = torch.cat((pocket_coords_concats, pocket_coords), dim=0)
            
            data['complex'].node_coords = torch.cat( # [glb_c || compound || glb_p || protein]
                (
                    data['complex'].node_coords,
                    torch.zeros((1, 3), device=compound_batch.device),
                    data['compound'].node_coords[compound_batch==i] - data['compound'].node_coords[compound_batch==i].mean(dim=0).reshape(1, 3) + pocket_coords.mean(dim=0).reshape(1, 3),
                    torch.zeros((1, 3), device=compound_batch.device), 
                    pocket_coords,
                ), dim=0
            ).float()

            data['complex'].node_coords_LAS = torch.cat( # [glb_c || compound || glb_p || protein]
                (
                    data['complex'].node_coords_LAS,
                    torch.zeros((1, 3), device=compound_batch.device),
                    data['compound'].rdkit_coords[compound_batch==i],
                    torch.zeros((1, 3), device=compound_batch.device), 
                    torch.zeros_like(pocket_coords)
                ), dim=0
            ).float()

            # masks
            n_protein = pocket_emb.shape[0]
            n_compound = batched_compound_emb[compound_batch==i].shape[0]
            segment = torch.zeros((n_protein + n_compound + 2), device=complex_batch.device)
            segment[n_compound+1:] = 1 # compound: 0, protein: 1
            data['complex'].segment = torch.cat((data['complex'].segment, segment), dim=0) # protein or ligand
            mask = torch.zeros((n_protein + n_compound + 2), device=complex_batch.device)
            mask[:n_compound+2] = 1 # glb_p can be updated
            data['complex'].mask = torch.cat((data['complex'].mask, mask.bool()), dim=0)
            is_global = torch.zeros((n_protein + n_compound + 2), device=complex_batch.device)
            is_global[0] = 1
            is_global[n_compound+1] = 1
            data['complex'].is_global = torch.cat((data['complex'].is_global, is_global.bool()), dim=0)

            # edge_index
            data['complex', 'c2c', 'complex'].edge_index = torch.cat(
                (
                    data['complex', 'c2c', 'complex'].edge_index, 
                    data['compound_atom_edge_list'].x[data['compound_atom_edge_list'].batch==i].t() + complex_batch.shape[0]
                ), dim=1)
            data['complex', 'LAS', 'complex'].edge_index = torch.cat(
                (
                    data['complex', 'LAS', 'complex'].edge_index, 
                    data['LAS_edge_list'].x[data['LAS_edge_list'].batch==i].t() + complex_batch.shape[0]
                ), dim=1)
            
            # batch_id
            complex_batch = torch.cat((complex_batch, torch.ones((n_compound + n_protein + 2), device=compound_batch.device)*i), dim=0)
            pocket_batch = torch.cat((pocket_batch, torch.ones((n_protein), device=compound_batch.device)*i), dim=0)

        # construct inputs
        batched_complex_coord = self.normalize_coord(data['complex'].node_coords.unsqueeze(-2))
        batched_complex_coord_LAS = self.normalize_coord(data['complex'].node_coords_LAS.unsqueeze(-2))
        complex_batch = complex_batch.to(torch.int64)
        pocket_batch = pocket_batch.to(torch.int64)
        pocket_coords_batched, _ = to_dense_batch(self.normalize_coord(pocket_coords_concats), pocket_batch)
        data['complex', 'c2c', 'complex'].edge_index = data['complex', 'c2c', 'complex'].edge_index.to(torch.int64)
        data['complex', 'LAS', 'complex'].edge_index = data['complex', 'LAS', 'complex'].edge_index.to(torch.int64)
        data['complex'].segment = data['complex'].segment.to(torch.bool)
        data['complex'].mask = data['complex'].mask.to(torch.bool)
        data['complex'].is_global = data['complex'].is_global.to(torch.bool)
        data['complex'].batch = complex_batch


        complex_coords, complex_out, pair_embed_batched = self.complex_model(
            batched_complex_coord,
            new_samples, 
            batch_id=complex_batch, 
            segment_id=data['complex'].segment,
            mask=data['complex'].mask, 
            is_global=data['complex'].is_global,
            compound_edge_index=data['complex', 'c2c', 'complex'].edge_index,
            LAS_edge_index=data['complex', 'LAS', 'complex'].edge_index,
            batched_complex_coord_LAS=batched_complex_coord_LAS,
            LAS_mask=None
        )

        if self.args.only_last_LAS:
            LAS_edge_list = data['complex', 'LAS', 'complex'].edge_index
            complex_coords.squeeze_(1)
            batched_complex_coord_LAS.squeeze_(1)
            for step in range(self.args.geom_reg_steps):
                LAS_cur_squared = torch.sum(
                    (complex_coords[LAS_edge_list[0]] - complex_coords[LAS_edge_list[1]]) ** 2, dim=1)
                LAS_true_squared = torch.sum(
                    (batched_complex_coord_LAS[LAS_edge_list[0]] - batched_complex_coord_LAS[LAS_edge_list[1]]) ** 2,
                    dim=1)
                grad_squared = 2 * (complex_coords[LAS_edge_list[0]] - complex_coords[LAS_edge_list[1]])
                LAS_force = 2 * (LAS_cur_squared - LAS_true_squared)[:, None] * grad_squared
                LAS_delta_coord = scatter_add(src=LAS_force, index=LAS_edge_list[1], dim=0,
                                            dim_size=complex_coords.shape[0])

                complex_coords = complex_coords + (LAS_delta_coord * self.args.geometry_reg_step_size) \
                    .clamp(min=self.normalize_coord(-2), max=self.normalize_coord(2))
            complex_coords.unsqueeze_(1)

        compound_flag = torch.logical_and(data['complex'].segment == 0, ~data['complex'].is_global)
        compound_coords_out = complex_coords[compound_flag].squeeze(-2)
        compound_coords_out = self.unnormalize_coord(compound_coords_out) + pocket_center_bias[compound_batch] # move back to whole protein coordinate
        
        if self.confidence_training:
            if self.args.stack_mlp:
                confidence_hidden = self.ranking_mlp_pre(scatter_add(src=complex_out, index=complex_batch, dim=0)).relu()
            else:
                confidence_hidden = scatter_add(src=complex_out, index=complex_batch, dim=0)
            confidence_score_pred = self.ranking_score_mlp(confidence_hidden).squeeze(-1)
            return compound_coords_out, compound_batch, confidence_score_pred
        else:
            return compound_coords_out, compound_batch


def get_model(args, logger):
    logger.log_message("FABind plus")
    model = FABindPlus(args, args.hidden_size, args.pocket_pred_hidden_size)
    return model
