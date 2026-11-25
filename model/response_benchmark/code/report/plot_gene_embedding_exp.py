"""Plot frozen gene-feature comparisons using saved matched-scope statistics."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

METHODS=['esm2_650m','esm2_graph__esm2_650m','qwen','qwen_graph__qwen','state_15b','state_graph__state_15b']
LABELS=['ESM graph / learned ID','ESM graph / ESM features','Qwen graph / learned ID','Qwen graph / Qwen features','STATE graph / learned ID','STATE graph / 15B features']
TASK_ORDER=[('GSE149383_PC9_DTEP_sample_grouped','sample_set_macro'),
            ('GSE165897_pre_EOC','patient_grouped_OOF_patients'),
            ('GSE223003_carboplatin_pre','replicate_macro'),
            ('GSE223003_nk_pre','replicate_macro'),
            ('GSE223003_olaparib_pre','replicate_macro'),
            ('GSE140440','frozen_cell_line_DU145'),('GSE140440','frozen_cell_line_PC3'),
            ('GSE163836','frozen_transfer_all_labeled_or_primary_pair'),
            ('GSE223779_TS485T','frozen_TS485T_same_drug_crizotinib'),
            ('GSE223779_TS485T','frozen_TS485T_cross_drug_alectinib'),
            ('GSE154600_EPI_external','frozen_ovarian_external_patients')]


def task_label(dataset,scope):
    if dataset=='GSE163836': return 'GSE163836 | FCIBC02'
    if dataset=='GSE223779_TS485T':
        return 'GSE223779 | TS485T '+('Crizotinib' if 'crizotinib' in scope else 'Alectinib')
    if dataset=='GSE154600_EPI_external': return 'GSE154600 | external patients'
    if dataset=='GSE165897_pre_EOC': return 'GSE165897 | patient OOF'
    if dataset=='GSE149383_PC9_DTEP_sample_grouped': return 'GSE149383 | PC9 DTEP'
    if dataset.startswith('GSE223003_'):
        endpoint=dataset.split('_')[1]
        return 'GSE223003 | '+('NK survival' if endpoint=='nk' else endpoint.capitalize())
    if scope.startswith('frozen_cell_line_'): return dataset+' | '+scope.removeprefix('frozen_cell_line_')
    if scope=='within_cell_split_test': return dataset+' | within test'
    return dataset+' | '+scope


def heatmaps(table,methods,labels,metrics,titles,path,suptitle,difference=False):
    tasks=table[['dataset','scope']].drop_duplicates()
    keys=list(zip(tasks.dataset,tasks.scope))
    original_rank={key:index for index,key in enumerate(keys)}
    keys.sort(key=lambda key:TASK_ORDER.index(key) if key in TASK_ORDER else len(TASK_ORDER)+original_rank[key])
    width=7.2 if len(methods)<=2 else 14
    height=max(3.1,len(keys)*.27+1.9)
    fig,axes=plt.subplots(1,len(metrics),figsize=(width,height),layout='constrained',squeeze=False)
    cmap=matplotlib.colors.LinearSegmentedColormap.from_list('matched',['#CA5596','#FFFFFF','#67A8AF'])
    cmap.set_bad('#DDDDDD')
    for ax,metric,title in zip(axes.ravel(),metrics,titles):
        values=table.pivot(index=['dataset','scope'],columns='model',values=metric).reindex(index=pd.MultiIndex.from_tuples(keys),columns=methods).to_numpy(float)
        image=ax.imshow(np.ma.masked_invalid(values),cmap=cmap,vmin=-.4 if difference else 0,vmax=.4 if difference else 1,aspect='auto')
        for i in range(len(keys)):
            for j in range(len(methods)):
                value=values[i,j]
                ax.text(j,i,'NA' if not np.isfinite(value) else (f'{value:+.3f}' if difference else f'{value:.3f}'),ha='center',va='center',fontsize=6.5)
        ax.set_xticks(range(len(methods)),labels,rotation=55,ha='right')
        ax.set_yticks(range(len(keys)),[task_label(d,s) for d,s in keys] if ax is axes[0,0] else [])
        ax.set_title(title,fontsize=7.3);ax.tick_params(length=0,labelsize=6.8)
        if keys==TASK_ORDER: ax.axhline(4.5,color='#333333',linewidth=.8)
    fig.colorbar(image,ax=axes.ravel().tolist(),label='Change from same-graph learned identity' if difference else 'Metric value',shrink=.65)
    fig.suptitle(suptitle,fontsize=8.5)
    fig.savefig(path);plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--open_path',required=True);parser.add_argument('--save_path',required=True)
    parser.add_argument('--variant',nargs='+',choices=['state_graph__state_15b','esm2_graph__esm2_650m','qwen_graph__qwen'],default=['state_graph__state_15b'])
    args=parser.parse_args();source=Path(args.open_path);out=Path(args.save_path);out.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({'font.family':'Arial','pdf.fonttype':42,'font.size':7,'axes.linewidth':.8})
    combined=pd.read_csv(source/'combined_primary_metrics.tsv',sep='\t')
    pairs={'state_graph__state_15b':('state_15b','Learned ID (16D)','ESM2-15B (5120D)'),
           'esm2_graph__esm2_650m':('esm2_650m','ESM graph / learned ID','ESM graph / ESM features'),
           'qwen_graph__qwen':('qwen','Qwen graph / learned ID','Qwen graph / Qwen features')}
    methods=[];labels=[]
    for variant in args.variant:
        baseline,old_label,new_label=pairs[variant]
        methods.extend([baseline,variant]);labels.extend([old_label,new_label])
    if combined[combined.model.isin(methods)].shape[0]!=22*len(args.variant): raise ValueError('Selected matched comparison is incomplete')
    heatmaps(combined[combined.model.isin(methods)],methods,labels,
        ['auroc','balanced_accuracy','auprc_resistant'],['AUROC','Balanced accuracy\n(threshold 0.5)','Positive-class AP'],
        out/'18_pretrained_gene_features_matched_topologies.pdf','STATE graph: frozen gene features versus learned identity' if len(args.variant)==1 else 'Corresponding graph / feature comparisons')
    changes=pd.read_csv(source/'matched_feature_differences.tsv',sep='\t')
    heatmaps(changes[changes.model.isin(args.variant)],args.variant,[pairs[v][2] for v in args.variant],
        ['delta_auroc','delta_balanced_accuracy'],['Change in AUROC','Change in balanced accuracy'],
        out/'19_pretrained_gene_features_matched_differences.pdf','Frozen gene features minus learned identity (same graph)',difference=True)
    within=pd.read_csv(source/'within_cell_split_test_metrics.tsv',sep='\t')
    heatmaps(within[within.model.isin(methods)],methods,labels,
        ['auroc','balanced_accuracy','auprc_resistant'],['AUROC','Balanced accuracy\n(threshold 0.5)','Positive-class AP'],
        out/'20_pretrained_gene_features_within_cell_split.pdf','Within-cohort random cell splits (exploratory)')


if __name__=='__main__':main()
