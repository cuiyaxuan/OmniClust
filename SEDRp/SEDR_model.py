import time
import numpy as np
import torch
import torch.nn.modules.loss
import torch.nn.functional as F
from sklearn.cluster import KMeans
from .SEDR_module import SEDR_module, SEDR_impute_module
from tqdm import tqdm

def target_distribution(batch):
    weight = (batch ** 2) / torch.sum(batch, 0)
    return (weight.t() / torch.sum(weight, 1)).t()

def reconstruction_loss(decoded, x):
    loss_func = torch.nn.MSELoss()
    loss_rcn = loss_func(decoded, x)
    return loss_rcn

def contrastive_loss(original, corrupted):
    """
    Contrastive loss between original (positive samples) and corrupted (negative samples).
    """
    loss_func = torch.nn.MSELoss()
    loss_contrast = loss_func(original, corrupted)
    return loss_contrast

def gcn_loss(preds, labels, mu, logvar, n_nodes, norm):
    cost = norm * F.binary_cross_entropy_with_logits(preds, labels)
    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 / n_nodes * torch.mean(torch.sum(
        1 + 2 * logvar - mu.pow(2) - logvar.exp().pow(2), 1))
    return cost + KLD

class Sedr:
    def __init__(
            self,
            X,
            graph_dict,
            rec_w=10,
            gcn_w=0.1,
            self_w=1,
            dec_kl_w=1,
            contrast_w=0.1,
            mode='clustering',
            device='cuda:0',
    ):

        self.rec_w = rec_w
        self.gcn_w = gcn_w
        self.self_w = self_w
        self.dec_kl_w = dec_kl_w
        self.contrast_w = contrast_w
        self.device = device
        self.mode = mode

        if 'mask' in graph_dict:
            self.mask = True
            self.adj_mask = graph_dict['mask'].to(self.device)
        else:
            self.mask = False

        self.cell_num = len(X)

        self.X = torch.FloatTensor(X.copy()).to(self.device)
        self.input_dim = self.X.shape[1]

        self.adj_norm = graph_dict["adj_norm"].to(self.device)
        self.adj_label = graph_dict["adj_label"].to(self.device)

        self.norm_value = graph_dict["norm_value"]

        if self.mode == 'clustering':
            self.model = SEDR_module(self.input_dim).to(self.device)
        elif self.mode == 'imputation':
            self.model = SEDR_impute_module(self.input_dim).to(self.device)
        else:
            raise ValueError(f'{self.mode} is not currently supported!')

    def mask_generator(self, N=1):
        idx = self.adj_label.indices()

        list_non_neighbor = []
        for i in range(0, self.cell_num):
            neighbor = idx[1, torch.where(idx[0, :] == i)[0]]
            n_selected = len(neighbor) * N

            # non neighbors
            total_idx = torch.arange(0, self.cell_num, dtype=torch.long, device=self.device)
            non_neighbor = total_idx[~torch.isin(total_idx, neighbor)]
            indices = torch.randperm(len(non_neighbor), device=self.device)
            random_non_neighbor = non_neighbor[indices[:n_selected]]
            list_non_neighbor.append(random_non_neighbor)

        x = torch.repeat_interleave(idx[0], N)
        y = torch.cat(list_non_neighbor)

        indices = torch.stack([x, y])
        indices = torch.cat([idx, indices], dim=1)

        value = torch.cat([self.adj_label.values(), torch.zeros(len(x), device=self.device)])
        adj_mask = torch.sparse_coo_tensor(indices, value, size=self.adj_label.shape, device=self.device)

        return adj_mask

    def corrupt_data(self, data, corruption_ratio=0.1):
        """
        Randomly set 10% of the values in the data to zero to create corrupted samples.
        """
        corrupted = data.clone()
        num_elements = data.numel()
        num_corrupt = int(num_elements * corruption_ratio)
        indices = torch.randperm(num_elements, device=self.device)[:num_corrupt]
        corrupted.view(-1)[indices] = 0
        return corrupted

    def train_without_dec(
            self,
            epochs=200,
            lr=0.01,
            decay=0.01,
            N=1,
    ):
        self.optimizer = torch.optim.Adam(
            params=list(self.model.parameters()),
            lr=lr,
            weight_decay=decay)

        self.model.train()

        for _ in tqdm(range(epochs)):
            self.model.train()
            self.optimizer.zero_grad()
            latent_z, mu, logvar, de_feat, _, feat_x, _, loss_self = self.model(self.X, self.adj_norm)

            if self.mask:
                pass
            else:
                if self.mode == 'imputation':
                    adj_mask = self.mask_generator(N=0)
                else:
                    adj_mask = self.mask_generator(N=1)
                self.adj_mask = adj_mask
                self.mask = True

            loss_gcn = gcn_loss(
                preds=self.model.dc(latent_z, self.adj_mask),
                labels=self.adj_mask.coalesce().values(),
                mu=mu,
                logvar=logvar,
                n_nodes=self.cell_num,
                norm=self.norm_value,
            )

            loss_rec = reconstruction_loss(de_feat, self.X)

            corrupted_X = self.corrupt_data(self.X)
            loss_contrast = contrastive_loss(self.X, corrupted_X)

            loss = self.rec_w * loss_rec + self.gcn_w * loss_gcn + self.self_w * loss_self + self.contrast_w * loss_contrast
            loss.backward()
            self.optimizer.step()

    def save_model(self, save_model_file):
        torch.save({'state_dict': self.model.state_dict()}, save_model_file)
        print('Saving model to %s' % save_model_file)

    def load_model(self, save_model_file):
        saved_state_dict = torch.load(save_model_file)
        self.model.load_state_dict(saved_state_dict['state_dict'])
        print('Loading model from %s' % save_model_file)

    def process(self):
        self.model.eval()
        latent_z, _, _, _, q, feat_x, gnn_z, _ = self.model(self.X, self.adj_norm)

        latent_z = latent_z.data.cpu().numpy()
        q = q.data.cpu().numpy()
        feat_x = feat_x.data.cpu().numpy()
        gnn_z = gnn_z.data.cpu().numpy()

        return latent_z, q, feat_x, gnn_z

    def recon(self):
        self.model.eval()
        latent_z, _, _, de_feat, q, feat_x, gnn_z, _ = self.model(self.X, self.adj_norm)
        de_feat = de_feat.data.cpu().numpy()

        # revise std and mean
        from sklearn.preprocessing import StandardScaler
        out = StandardScaler().fit_transform(de_feat)

        return out

    def train_with_dec(
            self,
            epochs=200,
            dec_interval=20,
            dec_tol=0.00,
            N=1,
    ):
        self.train_without_dec()

        kmeans = KMeans(n_clusters=self.model.dec_cluster_n, n_init=self.model.dec_cluster_n * 2, random_state=42)
        test_z, _, _, _ = self.process()
        y_pred_last = np.copy(kmeans.fit_predict(test_z))

        self.model.cluster_layer.data = torch.tensor(kmeans.cluster_centers_, device=self.device)
        self.model.train()

        for epoch_id in tqdm(range(epochs)):
            if epoch_id % dec_interval == 0:
                _, tmp_q, _, _ = self.process()
                tmp_p = target_distribution(torch.Tensor(tmp_q).to(self.device))
                y_pred = tmp_p.cpu().numpy().argmax(1)
                delta_label = np.sum(y_pred != y_pred_last).astype(np.float32) / y_pred.shape[0]
                y_pred_last = np.copy(y_pred)
                self.model.train()
                if epoch_id > 0 and delta_label < dec_tol:
                    print('delta_label {:.4}'.format(delta_label), '< tol', dec_tol)
                    print('Reached tolerance threshold. Stopping training.')
                    break

            torch.set_grad_enabled(True)
            self.optimizer.zero_grad()
            latent_z, mu, logvar, de_feat, out_q, _, _, _ = self.model(self.X, self.adj_norm)

            loss_gcn = gcn_loss(
                preds=self.model.dc(latent_z, self.adj_mask),
                labels=self.adj_mask.coalesce().values(),
                mu=mu,
                logvar=logvar,
                n_nodes=self.cell_num,
                norm=self.norm_value,
            )

            loss_rec = reconstruction_loss(de_feat, self.X)
            loss_kl = F.kl_div(out_q.log(), tmp_p)

            corrupted_X = self.corrupt_data(self.X)
            loss_contrast = contrastive_loss(self.X, corrupted_X)

            loss = self.gcn_w * loss_gcn + self.dec_kl_w * loss_kl + self.rec_w * loss_rec + self.contrast_w * loss_contrast
            loss.backward()
            self.optimizer.step()
