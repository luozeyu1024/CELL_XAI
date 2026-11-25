"""Plot grouped clinical validation alongside the existing frozen-transfer results."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

MODELS=['state_15b','esm2_650m','qwen','ppi','no_edges','random_signature','native_state_embedding']
LABELS=['STATE-15B','ESM2-650M','Qwen','PPI','No edges','Random signature','SE + MLP']
COLORS=['#67A8AF','#DB9735','#CA5596','#8DA0CB','#777777','#B3DE8E','#4D4D4D']


def clinical_patient_plot(summary_path,out):
    root=Path(summary_path).parent.parent.parent
    source='GSE165897_pre_EOC';target='GSE154600_EPI_external';parts=[]
    for model in MODELS:
        kind='state_mlp' if model=='native_state_embedding' else 'gat_gine'
        path=root/'output/transfer/ovarian_clinical'/model/(source+'_to_'+target)/kind/'ensemble_patient_predictions.tsv'
        if not path.exists(): return
        frame=pd.read_csv(path,sep='\t');frame['model']=model;parts.append(frame)
    data=pd.concat(parts,ignore_index=True)
    if data.duplicated(['patient_id','model']).any(): raise ValueError('Repeated clinical patient/model')
    annotation=data.groupby('patient_id').agg(y=('y','first'),n_cells=('n_cells','first'))
    if data.groupby('patient_id').y.nunique().gt(1).any() or data.groupby('patient_id').n_cells.nunique().gt(1).any():
        raise ValueError('Clinical patient annotation differs between models')
    annotation=annotation.assign(order=annotation.y.map({0:0,1:1,-1:2})).sort_values(['order','patient_id'])
    matrix=data.pivot(index='patient_id',columns='model',values='probability_resistant').reindex(index=annotation.index,columns=MODELS)
    if matrix.isna().any().any(): raise ValueError('Incomplete clinical patient predictions')
    data.to_csv(out/'data/clinical_external_patient_probabilities.tsv',sep='\t',index=False,float_format='%.17g')
    fig,ax=plt.subplots(figsize=(8.7,3.7),layout='constrained')
    cmap=matplotlib.colors.LinearSegmentedColormap.from_list('clinical_probability',['#CA5596','#FFFFFF','#67A8AF'])
    image=ax.imshow(matrix.to_numpy(),vmin=0,vmax=1,cmap=cmap,aspect='auto')
    for i in range(len(matrix)):
        for j in range(len(MODELS)): ax.text(j,i,f'{matrix.iloc[i,j]:.3f}',ha='center',va='center',fontsize=8)
    labels=[f'{pid} | '+{0:'Sensitive',1:'Resistant',-1:'Unknown (refractory)'}[int(row.y)]+f' | n={int(row.n_cells):,}'
            for pid,row in annotation.iterrows()]
    ax.set_yticks(np.arange(len(matrix)),labels,fontsize=8)
    ax.set_xticks(np.arange(len(MODELS)),LABELS,rotation=35,ha='right',fontsize=8)
    ax.set_title('GSE154600: frozen four-source-fold ensemble; 4 labeled patients + 1 unknown')
    fig.colorbar(image,ax=ax,label='Mean predicted resistant probability (uncalibrated)',ticks=[0,.25,.5,.75,1],shrink=.85)
    fig.savefig(out/'figures/13_frozen_clinical_patient_probabilities.pdf');plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--open_path',required=True)
    parser.add_argument('--save_path',required=True)
    args=parser.parse_args()
    table=pd.read_csv(args.open_path,sep='\t')
    out=Path(args.save_path);(out/'figures').mkdir(parents=True,exist_ok=True)
    (out/'data').mkdir(parents=True,exist_ok=True)
    table.to_csv(out/'data/mechanism_combined_metrics.tsv',sep='\t',index=False)
    plt.rcParams.update({'font.family':'Arial','pdf.fonttype':42,'ps.fonttype':42,
        'font.size':9,'axes.labelsize':10,'axes.titlesize':10,'axes.linewidth':.8})
    clinical_patient_plot(args.open_path,out)
    patient=table[table.scope.eq('patient_grouped_OOF_patients')]
    for task,frame in patient.groupby('dataset',sort=True):
        frame=frame.set_index('model').reindex(MODELS)
        fig,axes=plt.subplots(1,3,figsize=(10.5,3.5),layout='constrained')
        for ax,column,title in zip(axes,['auroc','balanced_accuracy','auprc_resistant'],['AUROC','Balanced accuracy','Positive-class AP']):
            values=frame[column].to_numpy(dtype=float)
            ax.bar(np.arange(len(MODELS)),values,color=COLORS,edgecolor='#333333',linewidth=.35,width=.72)
            for i,value in enumerate(values):
                if np.isfinite(value): ax.text(i,value+.018,f'{value:.3f}',ha='center',fontsize=7)
            reference=float(frame.resistant_prevalence.dropna().iloc[0]) if column=='auprc_resistant' else .5
            ax.axhline(reference,color='#777777',linestyle='--',linewidth=.7)
            ax.set_ylim(0,1.08);ax.set_yticks([0,.25,.5,.75,1]);ax.set_title(title)
            ax.set_xticks(np.arange(len(MODELS)),LABELS,rotation=55,ha='right',fontsize=7)
        fig.suptitle(f'{task}: held-out patient outcomes (4-fold OOF)',fontsize=11)
        fig.savefig(out/'figures'/f'10_grouped_patient_{task}.pdf');plt.close(fig)
    replicate=table[table.scope.str.startswith('held_out_replicate_')].copy()
    for task,frame in replicate.groupby('dataset',sort=True):
        if frame.model.nunique()!=len(MODELS): continue
        fig,axes=plt.subplots(1,2,figsize=(9,3.5),layout='constrained')
        scopes=sorted(frame.scope.unique())
        for ax,column,title in zip(axes,['auroc','auprc_resistant'],['AUROC','Positive-class AP']):
            for j,scope in enumerate(scopes):
                part=frame[frame.scope.eq(scope)].set_index('model').reindex(MODELS)
                ax.scatter(np.arange(len(MODELS))+(j-.5)*.18,part[column],c=COLORS,
                           marker=['o','^'][j],s=35,edgecolor='#333333',linewidth=.4,zorder=3)
                if column=='auprc_resistant':
                    baseline=float(part.resistant_prevalence.dropna().iloc[0])
                    ax.axhline(baseline,color='#999999',linestyle=['--',':'][j],linewidth=.7)
            if column=='auroc': ax.axhline(.5,color='#999999',linestyle='--',linewidth=.7)
            if column=='auroc':
                ax.set_ylim(0,1.05);ax.set_yticks([0,.25,.5,.75,1])
            else:
                from matplotlib.ticker import MaxNLocator
                upper=max(.3,min(1.05,float(np.ceil((frame[column].max()+.03)*10)/10)))
                ax.set_ylim(0,upper);ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
            ax.set_title(title)
            ax.set_xticks(np.arange(len(MODELS)),LABELS,rotation=45,ha='right',fontsize=7)
        from matplotlib.lines import Line2D
        axes[1].legend(handles=[Line2D([],[],marker=marker,linestyle='none',color='#666666',
            label=scope.replace('held_out_','').replace('_',' ')) for marker,scope in zip(['o','^'],scopes)],fontsize=7,frameon=False)
        fig.suptitle(task+': independent held-out experiments',fontsize=11)
        fig.savefig(out/'figures'/f'12_replicate_validation_{task}.pdf');plt.close(fig)
    # Show one explicit evaluation unit per task, with replicate detail above.
    display=table[~table.scope.isin(['patient_grouped_OOF_cells','frozen_ovarian_external_epithelial_cells','replicate_grouped_OOF_cells'])
                  & ~table.scope.str.startswith('held_out_replicate_')].copy()
    scope_labels={'patient_grouped_OOF_patients':'patient OOF','replicate_grouped_OOF_cells':'replicate OOF cells',
                  'frozen_transfer_all_labeled':'frozen cell transfer','frozen_transfer_all_labeled_or_primary_pair':'frozen cell transfer',
                  'frozen_TS485T_same_drug_crizotinib':'TS485T Crizotinib frozen cells',
                  'frozen_TS485T_cross_drug_alectinib':'TS485T Alectinib frozen cells'}
    scope_labels['frozen_ovarian_external_patients']='frozen clinical patients (n=4)'
    scope_labels['replicate_macro']='mean of two held-out replicate AUROCs'
    display['row']=display.dataset+' | '+display.scope.map(lambda s:scope_labels.get(s,s.replace('_',' ')))
    matrix=display.pivot(index='row',columns='model',values='auroc').reindex(columns=MODELS)
    fig,ax=plt.subplots(figsize=(9.5,max(3.6,.48*len(matrix)+1.3)),layout='constrained')
    cmap=matplotlib.colors.LinearSegmentedColormap.from_list('transfer',['#CA5596','#FFFFFF','#67A8AF'])
    cmap.set_bad('#D9D9D9')
    image=ax.imshow(matrix.to_numpy(dtype=float),vmin=0,vmax=1,cmap=cmap,aspect='auto')
    for row in range(len(matrix)):
        for col in range(len(MODELS)):
            value=matrix.iloc[row,col]
            ax.text(col,row,'NA' if pd.isna(value) else f'{value:.3f}',ha='center',va='center',fontsize=8)
    ax.set_xticks(np.arange(len(MODELS)),LABELS,rotation=35,ha='right')
    ax.set_yticks(np.arange(len(matrix)),matrix.index,fontsize=8)
    ax.set_title('AUROC by task and evaluation unit; no pooled performance')
    fig.colorbar(image,ax=ax,label='AUROC',ticks=[0,.25,.5,.75,1],shrink=.8)
    fig.savefig(out/'figures/11_mechanism_and_existing_transfer_AUROC.pdf');plt.close(fig)
    print('PLOT_ROWS',len(matrix),'PATIENT_TASKS',patient.dataset.nunique())


if __name__=='__main__': main()
