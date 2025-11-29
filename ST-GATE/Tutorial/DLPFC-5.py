import warnings

warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np
import scanpy as sc
import matplotlib.pyplot as plt
import os
import sys
from sklearn import metrics
from sklearn.metrics.cluster import adjusted_rand_score
from ST_GATE import Mix_ST_GATE
from Train import train_ST_GATE
from Mix_adj import Transfer_pytorch_Data, Mix_adj, mclust_R

learning_rate = 0.0005
n_epochs = 1300
alpha = 12
beta = 1
gamma = 1
lane = 1
tau = 0.5
num_clusters = 7
batch_size = 200
k_cutoff = 6
rad_cutoff = 150

spatial_hidden_dim = 64
spatial_dropout = 0.1
spatial_batch_size = 128

use_scheduler = False
scheduler_patience = 30
scheduler_factor = 0.7
lr_min = 1e-6

section_id = '151675'
print('Current slice %s' % (section_id))
input_dir = os.path.join('/DATA/DLPFC', section_id)
adata = sc.read_visium(path=input_dir, count_file='filtered_feature_bc_matrix.h5')
adata.var_names_make_unique()

sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=3000)
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)

Ann_df = pd.read_csv(os.path.join('/DATA/DLPFC', section_id, section_id + '_truth.txt'), sep='\t', header=None,
                     index_col=0)
Ann_df.columns = ['ground_truth']
adata.obs['ground_truth'] = Ann_df.loc[adata.obs_names, 'ground_truth']

adata = adata[~pd.isnull(adata.obs['ground_truth'])]

plt.rcParams["figure.figsize"] = (4, 4)
sc.pl.spatial(adata, img_key="hires", color=["ground_truth"], show=False)
plt.savefig('Ground_truth_spatial.png', dpi=300, bbox_inches='tight')
plt.close()

Mix_adj(adata, k_cutoff=k_cutoff, rad_cutoff=rad_cutoff)

print(f"Training with learning rate: {learning_rate}, alpha: {alpha}, beta: {beta}, gamma: {gamma}, lane: {lane}")
print(
    f"Spatial Attention params - hidden_dim: {spatial_hidden_dim}, dropout: {spatial_dropout}, batch_size: {spatial_batch_size}")
print(f"Using learning rate scheduler: {use_scheduler}")

adata = train_ST_GATE(
    adata,
    n_epochs=n_epochs,
    lr=learning_rate,
    dim_input=3000,
    dim_output=64,
    alpha=alpha,
    beta=beta,
    gamma=gamma,
    lane=lane,
    tau=tau,
    num_clusters=num_clusters,
    batch_size=batch_size,
    spatial_hidden_dim=spatial_hidden_dim,
    spatial_dropout=spatial_dropout,
    spatial_batch_size=spatial_batch_size,
    use_scheduler=use_scheduler,
    scheduler_patience=scheduler_patience,
    scheduler_factor=scheduler_factor,
    lr_min=lr_min,
    save_loss=True,
    verbose=True
)
print("Completion of training")

sc.pp.neighbors(adata, use_rep='ST_GATE')
sc.tl.umap(adata)
print("Start clustering")
plt.rcParams["figure.figsize"] = (4, 4)

adata = mclust_R(adata, num_cluster=num_clusters, used_obsm='ST_GATE')

cluster_label_map = {
    '1': 'Layer_1',
    '2': 'Layer_2',
    '3': 'Layer_3',
    '4': 'Layer_4',
    '5': 'Layer_5',
    '6': 'Layer_6',
    '7': 'WM'
}

adata.obs['mclust'] = adata.obs['mclust'].astype(str)
adata.obs['mclust_renamed'] = adata.obs['mclust'].map(cluster_label_map)

ARI = metrics.adjusted_rand_score(adata.obs['mclust'], adata.obs['ground_truth'])
NMI = metrics.normalized_mutual_info_score(adata.obs['mclust'], adata.obs['ground_truth'])
AMI = metrics.adjusted_mutual_info_score(adata.obs['mclust'], adata.obs['ground_truth'])
FM = metrics.fowlkes_mallows_score(adata.obs['mclust'], adata.obs['ground_truth'])

adata.uns['ARI'] = ARI
adata.uns['NMI'] = NMI
adata.uns['AMI'] = AMI
adata.uns['FM'] = FM

adata.uns['training_params'] = {
    'learning_rate': learning_rate,
    'n_epochs': n_epochs,
    'alpha': alpha,
    'beta': beta,
    'gamma': gamma,
    'lane': lane,
    'tau': tau,
    'num_clusters': num_clusters,
    'k_cutoff': k_cutoff,
    'rad_cutoff': rad_cutoff,
    'spatial_hidden_dim': spatial_hidden_dim,
    'spatial_dropout': spatial_dropout,
    'spatial_batch_size': spatial_batch_size,
    'use_scheduler': use_scheduler
}

print('Dataset:', section_id)
print(f'Training parameters - LR: {learning_rate}, Epochs: {n_epochs}, α: {alpha}, β: {beta}, γ: {gamma}, λ: {lane}')
print(f'Spatial Attention params - hidden_dim: {spatial_hidden_dim}, dropout: {spatial_dropout}')
print('ARI:', ARI)
print('NMI:', NMI)
print('AMI:', AMI)
print('FM:', FM)

adata.obs['mclust_renamed'] = adata.obs['mclust_renamed'].astype('category')

metrics_str = f'ARI={ARI:.2f}'
params_str = f'LR={learning_rate}, α={alpha}, β={beta}, γ={gamma}, λ={lane}'
spatial_str = f'Space Attn: dim={spatial_hidden_dim}, drop={spatial_dropout}'

sc.pl.umap(adata, color="mclust_renamed",
           title=None,
           legend_loc=None,
           show=False)
plt.title(f'ST_GATE (ARI={ARI:.2f})')
plt.savefig(f'{section_id}_ST_GATE_umap_ARI{ARI:.2f}.png', dpi=300, bbox_inches='tight')
plt.close()

sc.pl.umap(adata, color="ground_truth",
           title="Ground Truth",
           show=False)
fig = plt.gcf()
plt.savefig(f'{section_id}_Ground_truth_umap.png', dpi=300, bbox_inches='tight')
plt.close()

sc.pl.spatial(adata, color="mclust_renamed",
              title=None,
              legend_loc=None,
              show=False)
plt.title(f'ST_GATE (ARI={ARI:.2f})')
plt.savefig(f'{section_id}_ST_GATE_spatial_ARI{ARI:.2f}.png', dpi=300, bbox_inches='tight')
plt.close()

sc.pl.spatial(adata, color="ground_truth",
              title="Ground Truth",
              show=False)
plt.savefig(f'{section_id}_Ground_truth_spatial.png', dpi=300, bbox_inches='tight')
plt.close()

import matplotlib.pyplot as plt
import scanpy as sc

sc.settings.set_figure_params(dpi=300, dpi_save=300)

plt.rcParams["figure.figsize"] = (4, 4)
plt.rcParams["axes.labelsize"] = 10

sc.tl.embedding_density(adata, basis='umap')

sc.pl.embedding_density(adata,
                        basis='umap',
                        fg_dotsize=40,
                        save=section_id + '-ST_GATE-umap_density.png')

plt.rcParams["figure.figsize"] = (4, 4)
sc.tl.embedding_density(adata, basis='umap', groupby='ground_truth')
sc.pl.embedding_density(adata,
                        basis='umap',
                        groupby='ground_truth',
                        save=section_id + '-ST_GATE-umap_density_by_layer.png')

adata = adata[adata.obs['ground_truth'].notna(), :]
sc.tl.paga(adata, groups='ground_truth')

plt.rcParams["figure.figsize"] = (4, 4)
fig = sc.pl.paga_compare(adata, legend_fontsize=10, frameon=True, size=50,
                         legend_fontoutline=2, show=False)

current_fig = plt.gcf()
axes = current_fig.get_axes()

if len(axes) >= 2:
    current_fig.suptitle('')

    axes[0].set_title('ST_GATE', fontsize=12)
    axes[0].set_xlabel('')
    axes[0].set_ylabel('')

plt.tight_layout()

if len(axes) >= 2:
    extent1 = axes[0].get_window_extent().transformed(current_fig.dpi_scale_trans.inverted())
    current_fig.savefig(f'{section_id}_ST_GATE_umap.png', bbox_inches=extent1.expanded(1.2, 1.2),
                        dpi=300)

    extent2 = axes[1].get_window_extent().transformed(current_fig.dpi_scale_trans.inverted())
    current_fig.savefig(f'{section_id}_ST_GATE_paga.png', bbox_inches=extent2.expanded(1.2, 1.2), dpi=300)

plt.close()