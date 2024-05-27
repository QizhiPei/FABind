import torch


def update_best_isomorphism_idx(pos_x, pos_y, data, docking_coord_criterion):
    with torch.no_grad():
        pre_nodes = 0
        new_idx_x = []
        for i in range(len(data)):
            current_num_nodes = data.num_atoms[i]
            current_isomorphisms = [
                torch.LongTensor(iso).to(pos_x.device) for iso in data.isomorphisms[i] # [[j for j in range(current_num_nodes)], [j for j in range(current_num_nodes)], [j for j in range(current_num_nodes)]]
            ]

            if len(current_isomorphisms) == 1:
                new_idx_x.append(current_isomorphisms[0] + pre_nodes)
            else:
                pos_y_i = pos_y[pre_nodes : pre_nodes + current_num_nodes]
                pos_x_i = pos_x[pre_nodes : pre_nodes + current_num_nodes]
                pos_x_list = []

                for iso in current_isomorphisms:
                    pos_x_list.append(torch.index_select(pos_x_i, 0, iso))
                total_iso = len(pos_x_list)
                pos_y_i = pos_y_i.repeat(total_iso, 1, 1)
                pos_x_i = torch.stack(pos_x_list, dim=0)
                losses = docking_coord_criterion(pos_x_i, pos_y_i).reshape(len(current_isomorphisms), -1).mean(dim=-1)
                argmin_index = torch.argmin(losses)
                new_idx_x.append(current_isomorphisms[argmin_index] + pre_nodes)

            pre_nodes += current_num_nodes

    new_idx = torch.cat(new_idx_x, dim=0)
    return new_idx

def compute_permutation_loss(pos_x, pos_y, data, docking_coord_criterion):
    new_idx = update_best_isomorphism_idx(pos_x, pos_y, data, docking_coord_criterion)
    loss = docking_coord_criterion(pos_x[new_idx], pos_y)
    return loss


def update_best_isomorphism_idx_rmsd(pos_x, pos_y, data):
    with torch.no_grad():
        pre_nodes = 0
        new_idx_x = []
        for i in range(len(data)):
            current_num_nodes = data.num_atoms[i]
            current_isomorphisms = [
                torch.LongTensor(iso).to(pos_x.device) for iso in data.isomorphisms[i] # [[j for j in range(current_num_nodes)], [j for j in range(current_num_nodes)], [j for j in range(current_num_nodes)]]
            ]

            if len(current_isomorphisms) == 1:
                new_idx_x.append(current_isomorphisms[0] + pre_nodes)
            else:
                pos_y_i = pos_y[pre_nodes : pre_nodes + current_num_nodes]
                pos_x_i = pos_x[pre_nodes : pre_nodes + current_num_nodes]
                pos_x_list = []

                for iso in current_isomorphisms:
                    pos_x_list.append(torch.index_select(pos_x_i, 0, iso))
                total_iso = len(pos_x_list)
                pos_y_i = pos_y_i.repeat(total_iso, 1, 1)
                pos_x_i = torch.stack(pos_x_list, dim=0)
                losses = ((pos_x_i - pos_y_i) ** 2).sum(dim=-1).mean(dim=-1).sqrt()
                argmin_index = torch.argmin(losses)
                new_idx_x.append(current_isomorphisms[argmin_index] + pre_nodes)

            pre_nodes += current_num_nodes

    new_idx = torch.cat(new_idx_x, dim=0)
    return new_idx

def compute_permutation_rmsd(pos_x, pos_y, data):
    new_idx = update_best_isomorphism_idx_rmsd(pos_x, pos_y, data)
    loss = ((pos_x[new_idx] - pos_y) ** 2).sum(dim=-1).mean().sqrt()
    return loss