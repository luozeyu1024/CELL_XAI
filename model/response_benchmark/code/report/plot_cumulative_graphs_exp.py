"""Plot graph-size/connectivity tradeoffs and paired classifier results from saved tables."""
import argparse,json
from pathlib import Path
import numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.backends.backend_pdf import PdfPages
COLORS={'state_15b':'#CA5596','ppi':'#D99A32','qwen':'#67A8AF','esm2_650m':'#90D9E5'}
LABELS={'state_15b':'ESM2-15B','ppi':'PPI','qwen':'Qwen','esm2_650m':'ESM2-650M'}
plt.rcParams.update({'font.family':'Arial','font.size':7,'axes.titlesize':8,'pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False,'legend.frameon':False})
def short(ds):return ds.replace('_PC9_DTEP_sample_grouped',' / PC9 DTEP').replace('_pre_EOC',' / pre-EOC').replace('_carboplatin_pre',' / Carboplatin').replace('_nk_pre',' / NK').replace('_olaparib_pre',' / Olaparib').replace('_EPI_external',' / EPI external').replace('_TS485T',' / TS485T')

def plot_training_designs(root,out):
 project=next(p for p in root.parents if (p/'model/response_benchmark').is_dir())
 bench=project/'model/response_benchmark'
 source=pd.read_csv(root/'source_data/training_split_metrics.tsv',sep='\t')
 source=source[source.split.eq('test')].copy()
 designs=list(source[['dataset','fold']].drop_duplicates().itertuples(index=False,name=None))
 assert len(designs)==14 and len(source)==56
 rows=[]
 for ds,fold in designs:
  for rel in COLORS:
   row=source[source.dataset.eq(ds)&source.fold.eq(fold)&source.relation.eq(rel)].iloc[0]
   simple=fold=='single'
   new=bench/'output/cumulative_signature'/rel
   if not simple:new=new/'grouped'/ds/fold
   new=new/'within'/ds/'gat_gine'
   old=(bench/'output' if rel=='state_15b' else bench/'output/transfer'/rel) if simple else bench/'output/grouped_adaptive'/ds/fold/rel
   old=old/'within'/ds/'gat_gine'
   record=dict(dataset=ds,fold=fold,relation=rel,n_test_cells=int(row.n_cells),evaluation_unit='held-out cells within this design',split_design='exploratory random-cell split' if simple else 'biological-group-held-out test cells',mass_threshold=row.mass_threshold,cumulative_auroc=row.auroc,cumulative_balanced_accuracy=row.balanced_accuracy,cumulative_predictions=str(new/'predictions.tsv'),fixed_k_metrics=str(old/'metrics.json'),fixed_k_auroc=np.nan,fixed_k_balanced_accuracy=np.nan,auroc_difference=np.nan,comparison_status='missing_fixed_k_result')
   if (old/'metrics.json').exists():
    m=json.loads((old/'metrics.json').read_text(encoding='utf-8'))
    a=pd.read_csv(new/'predictions.tsv',sep='\t',float_precision='round_trip');b=pd.read_csv(old/'predictions.tsv',sep='\t',float_precision='round_trip')
    assert a.cell_id.tolist()==b.cell_id.tolist() and a.y.tolist()==b.y.tolist() and a.split.tolist()==b.split.tolist(),(ds,fold,rel)
    for frame,saved in [(a,{'auroc':row.auroc,'balanced_accuracy':row.balanced_accuracy}),(b,m['test'])]:
     test=frame[frame.split.eq('test')];assert len(test)==int(row.n_cells)
     y=test.y.to_numpy();ranks=test.margin_resistant_minus_sensitive.rank(method='average').to_numpy();n1=int((y==1).sum());n0=int((y==0).sum());auc=(ranks[y==1].sum()-n1*(n1+1)/2)/(n1*n0)
     pred=test.probability_resistant.to_numpy()>=.5;ba=((pred[y==1]).mean()+(~pred[y==0]).mean())/2
     assert np.isclose(auc,saved['auroc'],atol=1e-10) and np.isclose(ba,saved['balanced_accuracy'],atol=1e-10)
    record.update(fixed_k_auroc=m['test']['auroc'],fixed_k_balanced_accuracy=m['test']['balanced_accuracy'],auroc_difference=row.auroc-m['test']['auroc'],comparison_status='same_cells_labels_and_splits_verified')
   rows.append(record)
 table=pd.DataFrame(rows);table.to_csv(root/'source_data/training_design_comparison.tsv',sep='\t',index=False)
 labels=[short(ds)+' / '+fold+(' [random cells]' if fold=='single' else '') for ds,fold in designs]
 index=pd.MultiIndex.from_tuples(designs,names=['dataset','fold'])
 aucmap=matplotlib.colors.LinearSegmentedColormap.from_list('design_auc',['white','#67A8AF','#24486B'])
 deltamap=matplotlib.colors.LinearSegmentedColormap.from_list('design_change',['#CA5596','white','#67A8AF'])
 for filename,fields,titles in [
  ('07_all_14_training_designs.pdf',['cumulative_auroc','cumulative_balanced_accuracy'],['Cumulative IG: test AUROC','Cumulative IG: test balanced accuracy']),
  ('08_all_14_designs_fixedK_comparison.pdf',['fixed_k_auroc','cumulative_auroc','auroc_difference'],['Existing fixed K','Cumulative IG policy','AUROC change'])]:
  fig,axs=plt.subplots(1,len(fields),sharey=True,figsize=(9.5 if len(fields)==3 else 7.2,6.3))
  for ax,field,title in zip(axs,fields,titles):
   values=table.pivot(index=['dataset','fold'],columns='relation',values=field).reindex(index=index,columns=list(COLORS)).to_numpy()
   delta=field=='auroc_difference';cmap=(deltamap if delta else aucmap).copy();cmap.set_bad('#eeeeee')
   im=ax.imshow(np.ma.masked_invalid(values),cmap=cmap,vmin=-.5 if delta else 0,vmax=.5 if delta else 1,aspect='auto')
   for i in range(14):
    for j in range(4):
     v=values[i,j];text='NA' if np.isnan(v) else (f'{v:+.3f}' if delta else f'{v:.3f}')
     ax.text(j,i,text,ha='center',va='center',fontsize=6.3,color='white' if not delta and v>.65 else 'black')
   ax.set_title(title);ax.set_xticks(range(4),[LABELS[x] for x in COLORS],rotation=45,ha='right');ax.set_yticks(range(14),labels);ax.axhline(1.5,color='#777777',lw=.8)
   fig.colorbar(im,ax=ax,orientation='horizontal',fraction=.025,pad=.16,label='AUROC difference' if delta else ('Balanced accuracy' if 'balanced' in field else 'AUROC'))
  fig.subplots_adjust(left=.36 if len(fields)==2 else .29,right=.985,top=.94,bottom=.10,wspace=.15)
  fig.savefig(out/filename);plt.close(fig)
 print('14 designs x 4 graphs:',len(table),'current results;',int(table.fixed_k_auroc.notna().sum()),'matched fixed-K results')
def main():
 p=argparse.ArgumentParser();p.add_argument('--open_path',required=True);p.add_argument('--save_path',required=True);a=p.parse_args();root=Path(a.open_path).resolve();out=Path(a.save_path);out.mkdir(parents=True,exist_ok=True)
 s=pd.read_csv(root/'source_data/graph_summary.tsv',sep='\t');s=s[s.scope.eq('all_cells')];dslist=list(s.dataset.unique());handles=[Line2D([0],[0],color=COLORS[r],marker='o',ms=3,label=LABELS[r]) for r in COLORS]
 for field,label,name in [('n_edges','Undirected edges','01_edge_counts'),('isolated_fraction','Isolated nodes (%)','02_isolation'),('largest_component_fraction','Largest component / nodes (%)','03_largest_component')]:
  fig,axs=plt.subplots(int(np.ceil(len(dslist)/3)),3,figsize=(7.2,1.8*int(np.ceil(len(dslist)/3))),squeeze=False)
  for ax,ds in zip(axs.flat,dslist):
   f=s[s.dataset.eq(ds)];factor=1 if field=='n_edges' else 100
   for r in COLORS:
    g=f[f.relation.eq(r)].sort_values('threshold');x=g.threshold.to_numpy()*100;ax.plot(x,g[field+'_median']*factor,color=COLORS[r],marker='o',ms=2.5,lw=1);ax.fill_between(x,g[field+'_p25']*factor,g[field+'_p75']*factor,color=COLORS[r],alpha=.1)
   ax.set_title(short(ds),loc='left');ax.set_xticks([50,60,70,80]);ax.set_xlabel('Target full IG mass (%)');ax.set_ylabel(label)
   if field!='n_edges':ax.set_ylim(0,100)
  for ax in list(axs.flat)[len(dslist):]:ax.set_visible(False)
  fig.legend(handles=handles,loc='upper center',ncol=4,bbox_to_anchor=(.5,1));fig.tight_layout(rect=(0,0,1,.965));fig.savefig(out/(name+'.pdf'));plt.close(fig)
 if (root/'source_data/gpu_cost.tsv').exists():
  cost=pd.read_csv(root/'source_data/gpu_cost.tsv',sep='\t');cost['trial']=cost.dataset+' / '+cost.fold;trials=list(cost.trial.unique())
  with PdfPages(out/'04_measured_gpu_cost.pdf') as pdf:
   for start in range(0,len(trials),8):
    page_trials=trials[start:start+8];fig,axs=plt.subplots(int(np.ceil(len(page_trials)/2)),2,figsize=(7.2,2*int(np.ceil(len(page_trials)/2))),squeeze=False)
    for ax,trial in zip(axs.flat,page_trials):
     for r in COLORS:
      f=cost[cost.trial.eq(trial)&cost.relation.eq(r)].groupby('threshold').peak_allocated_GiB.max();ax.plot(f.index.to_numpy()*100,f.values,color=COLORS[r],marker='o',ms=3)
     ax.set_title(short(trial),loc='left');ax.set_xlabel('Target full IG mass (%)');ax.set_ylabel('Peak allocated GPU (GiB)');ax.set_xticks([50,60,70,80])
    for ax in list(axs.flat)[len(page_trials):]:ax.set_visible(False)
    fig.legend(handles=handles,loc='upper center',ncol=4);fig.tight_layout(rect=(0,0,1,.92 if len(page_trials)<3 else .97));pdf.savefig(fig);plt.close(fig)
 metrics=root/'source_data/matched_fixedK_differences.tsv'
 if metrics.exists() and metrics.stat().st_size>5:
  f=pd.read_csv(metrics,sep='\t')
  if len(f):
   f=f[f.metric.eq('auroc')].copy();f['evaluation']=f.dataset.map(short)+' / '+f.scope.str.replace('frozen_','',regex=False).str.replace('patient_grouped_OOF_patients','patient OOF',regex=False).str.replace('transfer_all_labeled_or_primary_pair','FCIBC02 pair',regex=False).str.replace('TS485T_same_drug_crizotinib','crizotinib',regex=False).str.replace('TS485T_cross_drug_alectinib','alectinib',regex=False).str.replace('ovarian_external_patients','patients',regex=False);grid=f.pivot(index='evaluation',columns='relation',values='difference').reindex(columns=list(COLORS));fig,ax=plt.subplots(figsize=(7.2,max(3,len(grid)*.28+1.2)));v=grid.to_numpy();im=ax.imshow(v,cmap=matplotlib.colors.LinearSegmentedColormap.from_list('change',['#CA5596','white','#67A8AF']),vmin=-.5,vmax=.5,aspect='auto')
   for i in range(len(grid)):
    for j in range(4):ax.text(j,i,'NA' if np.isnan(v[i,j]) else f'{v[i,j]:+.3f}',ha='center',va='center',fontsize=7)
   ax.set_xticks(range(4),[LABELS[r] for r in COLORS]);ax.set_yticks(range(len(grid)),grid.index);fig.colorbar(im,ax=ax,label='AUROC change: cumulative minus fixed K',shrink=.7);fig.tight_layout();fig.savefig(out/'05_paired_AUROC_changes.pdf');plt.close(fig)
   fig,axs=plt.subplots(1,2,sharey=True,figsize=(7.2,max(3,len(grid)*.28+1.3)))
   for ax,field,title in zip(axs,['fixed_k','cumulative'],['Existing fixed K','Cumulative IG policy']):
    panel=f.pivot(index='evaluation',columns='relation',values=field).reindex(index=grid.index,columns=list(COLORS));values=panel.to_numpy();im=ax.imshow(values,cmap=matplotlib.colors.LinearSegmentedColormap.from_list('auc',['white','#67A8AF','#24486B']),vmin=0,vmax=1,aspect='auto');ax.set_title(title)
    for i in range(len(panel)):
     for j in range(4):ax.text(j,i,'NA' if np.isnan(values[i,j]) else f'{values[i,j]:.3f}',ha='center',va='center',fontsize=6.5,color='white' if values[i,j]>.65 else 'black')
    ax.set_xticks(range(4),[LABELS[r] for r in COLORS],rotation=45,ha='right');ax.set_yticks(range(len(panel)),panel.index)
   fig.subplots_adjust(left=.39,right=.90,bottom=.18,top=.9,wspace=.15);cb=fig.add_axes([.92,.25,.015,.45]);fig.colorbar(im,cax=cb,label='AUROC');fig.savefig(out/'06_absolute_AUROC_comparison.pdf');plt.close(fig)
 plot_training_designs(root,out)
 print('Saved plots from current verified tables')
if __name__=='__main__':main()


