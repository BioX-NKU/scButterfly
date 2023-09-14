import scanpy as sc
import anndata as ad
from scipy import sparse
import pandas as pd


def draw_tsne(data, title, color):
    """
    Visualization of prediction using tSNE Algorithm.
    
    Parameters
    ----------
    data: Anndata
        Anndata for visualization.
        
    title: str
        title of picture.

    color: list
        color of picture

    Returns
    ----------
    fig
        figure with specific color and title.
    """

    sc.settings.set_figure_params(dpi=120, facecolor='white')
    sc.pp.pca(data)
    sc.pp.neighbors(data)
    sc.tl.tsne(data)
    fig = sc.pl.tsne(data, color=color, title=title, return_fig=True)
    return fig


def draw_reg_plot(eval_adata,
                  cell_type,
                  reg_type='mean',
                  axis_keys={"x": "pred", "y": "stimulated"},
                  condition_key='condition',
                  gene_draw=None,
                  top_gene_list=None,
                  save_path=None,
                  title=None,
                  show=True,
                  fontsize=14
                 ):
    import pandas as pd
    import seaborn as sns
    import numpy as np
    import scanpy as sc
    import matplotlib.pyplot as plt
    from adjustText import adjust_text
    
    df_case = eval_adata[(eval_adata.obs[condition_key]==axis_keys["y"])].to_df()
    df_pred = eval_adata[(eval_adata.obs[condition_key]==axis_keys["x"])].to_df()
    if reg_type=='mean':
        mean_case = df_case.mean().values.reshape(-1, 1)
        mean_pred = df_pred.mean().values.reshape(-1, 1)
    elif reg_type=='var':
        mean_case = df_case.var().values.reshape(-1, 1)
        mean_pred = df_pred.var().values.reshape(-1, 1)
    data = np.hstack((mean_case, mean_pred))
    data_df = pd.DataFrame(data, columns=['case', 'predict'], index=df_case.columns)
    
    sns.set(color_codes=True)
    
    fig = plt.figure()
    ax = fig.add_axes([0,0,1,1])
    
    sns.regplot(x='case', y='predict', data=data_df, ax=ax)
    if gene_draw is not None:
        texts = []
        x = mean_case
        y = mean_pred
        for i in gene_draw:
            j = eval_adata.var_names.tolist().index(i)
            x_bar = x[j]
            y_bar = y[j]
            texts.append(plt.text(x_bar, y_bar, i, fontsize=11, color="black"))
            ax.plot(x_bar, y_bar, "o", color="red", markersize=5)
        adjust_text(
                texts,
                x=x,
                y=y,
                arrowprops=dict(arrowstyle="->", color="grey", lw=0.5),
                force_points=(0.0, 0.0),
        )
    if top_gene_list is not None:
        data_deg = data_df.loc[top_gene_list, :]
        r_top = round(data_deg['case'].corr(data_deg['predict'], method='pearson'), 3)
        xt = 0.1 * np.max(data_df['case'])
        yt = 0.85 * np.max(data_df['predict'])
        ax.text(xt, yt, s='$R^2_{top 100 genes}$=' + str(round(r_top*r_top,3)), fontsize=fontsize, color='black')
    r = round(data_df['case'].corr(data_df['predict'], method='pearson'), 3)
    xt = 0.1 * np.max(data_df['case'])
    yt = 0.75 * np.max(data_df['predict'])
    ax.text(xt, yt, s='$R^2_{all genes}$=' + str(round(r*r,3)), fontsize=fontsize, color='black')
    if title:
        ax.set_title(title)
    else:
        ax.set_title('The Linear Regression of True and Predict Expression '+ reg_type +' of ' + cell_type)

    return round(r*r,3),round(r_top*r_top,3), fig