"""Compare gene attributions from three models on identical cells."""
import argparse, csv, itertools, json
from pathlib import Path
import numpy as np
from scipy.sparse import load_npz
from scipy.stats import spearmanr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib_venn import venn3

MODELS=['scimilarity','scvi_czi','scvi_local']
LABELS=['SCimilarity','scVI-CZI','scVI-local']
PAIRS=list(itertools.combinations(MODELS,2))

def write(path,rows):
    if not rows:return
    with path.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]),delimiter='\t');w.writeheader();w.writerows(rows)

def save(fig,path):
    fig.savefig(str(path)+'.pdf',bbox_inches='tight',metadata={'CreationDate':None})

def corr(x,y):
    return float(spearmanr(x,y).statistic) if len(x)>2 and np.ptp(x)>0 and np.ptp(y)>0 else float('nan')

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--open_path',type=Path,required=True);p.add_argument('--save_path',type=Path,required=True)
    p.add_argument('--top_k',nargs='+',type=int,default=[10,50,100])
    args=p.parse_args()
    out=args.save_path;out.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({'font.family':'Arial','font.size':8,'pdf.fonttype':42,'svg.fonttype':'none'})
    stats=[];overlaps=[];members=[];qc=[];cellstats=[];datasets=[]
    for ds in sorted(args.open_path.glob('GSE*')):
        sf=ds/'dataset_status.json'
        if not sf.exists() or not json.loads(sf.read_text(encoding='utf-8')).get('eligible',False):
            print('Skipping',ds.name,'counts not validated',flush=True);continue
        if any(not (ds/'attributions'/f'{m}_ig.npz').exists() for m in MODELS):
            print('Skipping',ds.name,'incomplete three-model results',flush=True);continue
        scores={};arrays={};positions={};manifests={}
        for m in MODELS:
            a=load_npz(ds/'attributions'/f'{m}_ig.npz').tocsr()
            names=json.loads((ds/'attributions'/f'{m}_ig_var_names.json').read_text(encoding='utf-8'))
            mf=json.loads((ds/'manifests'/f'{m}_ig_manifest.json').read_text(encoding='utf-8'))
            assert len(names)==a.shape[1] and len(set(names))==len(names)
            assert np.isfinite(a.data).all() and (a.data>=0).all()
            if m=='scvi_czi':assert mf['model']['trained_on_input'] is False
            if m=='scvi_local':assert mf['model']['hvg_flavor']=='seurat_v3'
            arrays[m]=a;positions[m]={g:i for i,g in enumerate(names)};manifests[m]=mf
            scores[m]=dict(zip(names,np.asarray(a.mean(axis=0)).ravel()))
            sums=np.asarray(a.sum(axis=1)).ravel()
            qc.append(dict(dataset=ds.name,model=m,cells=a.shape[0],genes=a.shape[1],zero_rows=int((sums==0).sum()),row_sum_min=float(sums.min()),row_sum_max=float(sums.max())))
        ref=manifests[MODELS[0]]['input']
        for m in MODELS[1:]:
            assert manifests[m]['input']['cell_ids_used']==ref['cell_ids_used']
            assert manifests[m]['input']['cell_indices_used']==ref['cell_indices_used']
        datasets.append(ds.name)
        for m1,m2 in PAIRS:
            common=sorted(scores[m1].keys()&scores[m2].keys())
            x,y=[np.array([scores[m][g] for g in common]) for m in [m1,m2]];active=(x>0)|(y>0)
            stats.append(dict(dataset=ds.name,model1=m1,model2=m2,n_common=len(common),spearman_rho=corr(x,y),n_active=int(active.sum()),active_spearman_rho=corr(x[active],y[active])))
            ix=[positions[m1][g] for g in common];iy=[positions[m2][g] for g in common]
            for start in range(0,arrays[m1].shape[0],128):
                aa=arrays[m1][start:start+128,:][:,ix].toarray();bb=arrays[m2][start:start+128,:][:,iy].toarray()
                for j,(a,b) in enumerate(zip(aa,bb)):
                    cellstats.append(dict(dataset=ds.name,model1=m1,model2=m2,source_cell_index=ref['cell_indices_used'][start+j],cell_id=ref['cell_ids_used'][start+j],spearman_rho=corr(a,b),n_common=len(common)))
        folder=out/ds.name;folder.mkdir(exist_ok=True)
        union=sorted(set.union(*(set(scores[m]) for m in MODELS)))
        write(folder/'gene_mean_importance.tsv',[dict(gene=g,**{m:scores[m].get(g,'') for m in MODELS}) for g in union])
        shared=sorted(set.intersection(*(set(scores[m]) for m in MODELS)))
        for scope in ['common','native']:
            fig,axes=plt.subplots(1,len(args.top_k),figsize=(3.5*len(args.top_k),3.8),squeeze=False)
            for ax,k in zip(axes[0],args.top_k):
                sets=[];ties=[]
                for m in MODELS:
                    pool=shared if scope=='common' else list(scores[m]);order=sorted((g for g in pool if scores[m][g]>0),key=lambda g:(-scores[m][g],g))
                    selected=order[:k];sets.append(set(selected));ties.append(sum(scores[m][g]==scores[m][selected[-1]] for g in order) if selected else 0)
                    for rank,g in enumerate(selected,1):members.append(dict(dataset=ds.name,scope=scope,top_k=k,model=m,rank=rank,gene=g,mean_importance=scores[m][g]))
                allshared=set.intersection(*sets);allunion=set.union(*sets)
                row=dict(dataset=ds.name,scope=scope,top_k=k,n_common_universe=len(shared),intersection_all3=len(allshared),union_all3=len(allunion),jaccard_all3=len(allshared)/len(allunion) if allunion else '')
                row.update({f'{m}_n':len(s) for m,s in zip(MODELS,sets)});row.update({f'{m}_boundary_ties':n for m,n in zip(MODELS,ties)})
                for bits in itertools.product([0,1],repeat=3):
                    if not any(bits):continue
                    region=set(allunion)
                    for b,s in zip(bits,sets):region=region&s if b else region-s
                    row['region_'+''.join(map(str,bits))]=len(region)
                overlaps.append(row)
                v=venn3(sets,set_labels=LABELS,set_colors=['#CA5596','#67A8AF','#DB9735'],alpha=.65,ax=ax)
                for t in list(v.set_labels or [])+list(v.subset_labels or []):
                    if t:t.set_fontsize(8)
                ax.set_title(f'Top {k} | shared by all: {len(allshared)}',fontsize=9)
            fig.suptitle(f'{ds.name} | {scope} gene universe',fontsize=10);fig.tight_layout();save(fig,folder/f'top_gene_overlap_{scope}');plt.close(fig)
        print(ds.name,'cells',arrays[MODELS[0]].shape[0],'three-model common genes',len(shared),flush=True)
    for name,rows in [('spearman_dataset',stats),('top_gene_overlap',overlaps),('top_gene_members',members),('matrix_qc',qc),('spearman_per_cell',cellstats)]:write(out/(name+'.tsv'),rows)
    if not datasets:
        raise SystemExit('No cohort has three complete IG matrices and validated counts.')
    cmap=LinearSegmentedColormap.from_list('rho',['#CA5596','#FFFFFF','#67A8AF'])
    for key,filename,title in [('spearman_rho','spearman_heatmap','All pairwise-common genes'),('active_spearman_rho','spearman_heatmap_active','Active pairwise-common genes')]:
        values=np.array([[next(r[key] for r in stats if r['dataset']==d and (r['model1'],r['model2'])==pair) for pair in PAIRS] for d in datasets])
        fig,ax=plt.subplots(figsize=(6.8,4.6));im=ax.imshow(values,vmin=-1,vmax=1,cmap=cmap,aspect='auto')
        ax.set_xticks(range(3),['SCimilarity\nvs scVI-CZI','SCimilarity\nvs scVI-local','scVI-CZI\nvs scVI-local']);ax.set_yticks(range(len(datasets)),datasets)
        for i in range(len(datasets)):
            for j in range(3):ax.text(j,i,f'{values[i,j]:.3f}',ha='center',va='center')
        ax.set_title('Mean gene importance | '+title,fontsize=10);fig.colorbar(im,ax=ax,label='Spearman rho');fig.tight_layout();save(fig,out/filename);plt.close(fig)

if __name__=='__main__':main()
