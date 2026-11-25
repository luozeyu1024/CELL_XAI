"""Train paired within-GSE classifiers and transfer the largest-GSE checkpoints unchanged."""

# Keep established imports and experiment-relative paths available.
import sys as _sys
from pathlib import Path as _Path
_code_root = _Path(__file__).resolve().parents[1]
if str(_code_root) not in _sys.path:
    _sys.path.insert(0, str(_code_root))
from benchmark_paths import CODE_ROOT, BENCH_ROOT, PROJECT_ROOT, script_path

import argparse
import json
from pathlib import Path
import random
import time
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader as TensorLoader
from torch_geometric.loader import DataLoader as GraphLoader
from sklearn.metrics import roc_auc_score, average_precision_score, balanced_accuracy_score
from sklearn.metrics import matthews_corrcoef, brier_score_loss, confusion_matrix, f1_score
from cell_graph_dataset import CellGraphDataset
from graph_model import GraphClassifier
from split_registry import load_cells, apply_splits, fold_from_path, split_root

SOURCE_DATASETS = ['GSE140440','GSE223779']
DATASETS = SOURCE_DATASETS + ['GSE163836','GSE202234','GSE223779_TS485T']


def write_json(path, value):
    Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n',encoding='utf-8')


def seed_all(seed):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False


class StateMLP(nn.Module):
    def __init__(self, input_dim=2048, hidden=256, bottleneck=64, dropout=.2):
        super().__init__()
        self.config=dict(input_dim=input_dim,hidden=hidden,bottleneck=bottleneck,dropout=dropout)
        self.register_buffer('feature_mean',torch.zeros(input_dim))
        self.register_buffer('feature_std',torch.ones(input_dim))
        self.net=nn.Sequential(nn.Linear(input_dim,hidden),nn.ReLU(),nn.Dropout(dropout),
            nn.Linear(hidden,bottleneck),nn.ReLU(),nn.Dropout(dropout),nn.Linear(bottleneck,2))

    def forward(self,x):
        return self.net((x-self.feature_mean)/self.feature_std.clamp_min(1e-6))


def metrics(frame):
    f=frame.loc[frame.y.ge(0)];y=f.y.to_numpy();prob=f.probability_resistant.to_numpy(dtype=np.float64);pred=prob>=.5
    score=f.margin_resistant_minus_sensitive.to_numpy(dtype=np.float64)
    if not len(f):return {'n_cells':0}
    both=len(np.unique(y))==2
    return dict(n_cells=len(y),n_sensitive=int((y==0).sum()),n_resistant=int((y==1).sum()),
        auroc=float(roc_auc_score(y,score)) if both else None,
        auprc_resistant=float(average_precision_score(y,score)) if both else None,
        auprc_sensitive=float(average_precision_score(1-y,-score)) if both else None,
        resistant_prevalence=float(y.mean()),balanced_accuracy=float(balanced_accuracy_score(y,pred)),
        mcc=float(matthews_corrcoef(y,pred)),macro_f1=float(f1_score(y,pred,average='macro',zero_division=0)),
        brier_score=float(brier_score_loss(y,prob)),threshold=.5,ranking_score='resistant_minus_sensitive_logit',
        confusion_matrix_true_rows_sensitive_resistant=confusion_matrix(y,pred,labels=[0,1]).tolist())


def load_cohort(root,ds,kind,relation="state_15b"):
    folder=root/'data/cohorts'/ds
    cells=load_cells(folder/'cells.tsv',ds)
    if kind=='state_mlp':
        x=np.load(folder/'state_cell_embeddings.npy')
        assert x.shape==(len(cells),2048) and np.isfinite(x).all()
        return torch.from_numpy(x),cells,None
    info=json.loads((folder/'data_sources.json').read_text())
    graph_root=Path(info['graph_source']).parent
    raw=CellGraphDataset(graph_root,ds,relation)
    graphs=[]
    for row in cells.itertuples():
        g=raw.to_pyg(int(row.graph_row))
        assert g.cell_id==row.cell_id
        g.x=g.x[:,:1].clone()  # Expression only; upstream IG selects nodes, not an input score.
        g.y=torch.tensor([row.y],dtype=torch.long)
        g.validate(raise_on_error=True)
        graphs.append(g)
    print('Loaded',ds,kind,len(graphs),'graphs',flush=True)
    return graphs,cells,raw.genes.tolist()


def batches(data,cells,index,kind,batch_size,shuffle=False,seed=42):
    generator=torch.Generator().manual_seed(seed)
    if kind=='state_mlp':
        ds=TensorDataset(data[index],torch.tensor(cells.y.iloc[index].to_numpy(),dtype=torch.long))
        return TensorLoader(ds,batch_size=batch_size,shuffle=shuffle,generator=generator,num_workers=0)
    return GraphLoader([data[i] for i in index],batch_size=batch_size,shuffle=shuffle,generator=generator,num_workers=0)


def forward(model,batch,kind,device):
    if kind=='state_mlp':
        x,y=batch
        return model(x.to(device)),y.to(device)
    b=batch.to(device)
    return model(b.x,b.edge_index,b.edge_attr,b.batch,b.gene_id),b.y


def predict(model,data,cells,kind,device,batch_size):
    model.eval();out=[];all_logits=[]
    with torch.no_grad():
        for batch in batches(data,cells,np.arange(len(cells)),kind,batch_size):
            logits,_=forward(model,batch,kind,device)
            logits=logits.to(torch.float64)
            out.append(logits.softmax(-1)[:,1].cpu().numpy())
            all_logits.append(logits.cpu().numpy())
    result=cells[['cell_id','y','response','split']].copy()
    result['probability_resistant']=np.concatenate(out)
    logits=np.concatenate(all_logits)
    result['logit_sensitive']=logits[:,0]
    result['logit_resistant']=logits[:,1]
    result['margin_resistant_minus_sensitive']=logits[:,1]-logits[:,0]
    result['predicted']=(result.probability_resistant>=.5).astype(int)
    return result


def training_statistics(graphs,indices,n_genes):
    total=square=0.;count=0;observed=np.zeros(n_genes,dtype=bool)
    for i in indices:
        x=graphs[i].x[:,0].numpy().astype(np.float64)
        total+=x.sum();square+=(x*x).sum();count+=len(x)
        observed[graphs[i].gene_id.numpy()]=True
    return float(total/count),float(max(square/count-(total/count)**2,1e-12)**.5),observed


def restore(path,device):
    c=torch.load(path,map_location='cpu',weights_only=False)
    model=StateMLP(**c['model_config']) if c['model_type']=='state_mlp' else GraphClassifier(**c['model_config'])
    model.load_state_dict(c['state_dict'],strict=True)
    return model.to(device).eval(),c


def output_root(args,root):
    relation=getattr(args,'relation','state_15b')
    return root/'output' if relation=='state_15b' else root/'output/transfer'/relation


def train_one(args,root,ds,kind):
    registered=load_cells(Path(root)/'data/cohorts'/ds/'cells.tsv',ds,mode='source')
    args.splits_path=str(split_root())
    dest=output_root(args,root)/'within'/ds/kind;dest.mkdir(parents=True,exist_ok=True)
    if (dest/'metrics.json').exists() and (dest/'checkpoint.pt').exists():
        previous=json.loads((dest/'run_parameters.json').read_text())
        for key in ['seed','hidden','layers','heads','gene_dim','dropout','lr','weight_decay',
                    'graph_batch_size','mlp_batch_size','epochs','patience','min_delta']:
            if previous[key]!=vars(args)[key]:
                raise ValueError(f'Completed {ds}/{kind} uses a different {key}; choose a new experiment directory')
        saved=torch.load(dest/'checkpoint.pt',map_location='cpu',weights_only=False)
        for role,key in [('train','training_cell_ids'),('validation','validation_cell_ids')]:
            expected=registered.loc[registered.split.eq(role),'cell_id'].tolist()
            if saved[key]!=expected:raise ValueError('Existing checkpoint differs from JSON '+role+' IDs')
        predictions=pd.read_csv(dest/'predictions.tsv',sep='\t',keep_default_na=False)
        if set(predictions.loc[predictions.split.eq('test'),'cell_id'])!=set(registered.loc[registered.split.eq('test'),'cell_id']):
            raise ValueError('Existing test predictions differ from JSON IDs')
        print('Completed result matches JSON splits; skipping',ds,kind,flush=True);return
    seed_all(args.seed)
    data,cells,genes=load_cohort(root,ds,kind,getattr(args,"relation","state_15b"))
    cells=apply_splits(cells,ds,fold_from_path(root),mode='source')
    train=np.flatnonzero(cells.split.eq('train'));val=np.flatnonzero(cells.split.eq('validation'))
    assert (cells.y.iloc[train]>=0).all() and (cells.y.iloc[val]>=0).all()
    if kind=='state_mlp':
        model=StateMLP(dropout=args.dropout)
        model.feature_mean.copy_(data[train].mean(0))
        model.feature_std.copy_(data[train].std(0,unbiased=False).clamp_min(1e-6))
        observed=None
    else:
        mean,std,observed=training_statistics(data,train,len(genes))
        model=GraphClassifier(len(genes),args.hidden,args.layers,args.heads,args.gene_dim,args.dropout,mean,std)
    model=model.to(args.device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=args.weight_decay)
    counts=np.bincount(cells.y.iloc[train],minlength=2)
    weight=torch.tensor(len(train)/(2*counts),dtype=torch.float32,device=args.device)
    batch_size=args.mlp_batch_size if kind=='state_mlp' else args.graph_batch_size
    loader=batches(data,cells,train,kind,batch_size,True,args.seed)
    valid=batches(data,cells,val,kind,batch_size)
    best=float('inf');stale=0;history=[];start=time.perf_counter()
    print('TRAIN',ds,kind,'train/val/test',len(train),len(val),int(cells.split.eq('test').sum()),flush=True)
    for epoch in range(1,args.epochs+1):
        losses=[]
        for phase,stream in [('train',loader),('validation',valid)]:
            model.train(phase=='train');numerator=denominator=0.
            with torch.set_grad_enabled(phase=='train'):
                for batch in stream:
                    if phase=='train':optimizer.zero_grad(set_to_none=True)
                    logits,y=forward(model,batch,kind,args.device)
                    per_cell=nn.functional.cross_entropy(logits,y,weight=weight,reduction='none')
                    norm=weight[y].sum();loss=per_cell.sum()/norm
                    if phase=='train':
                        loss.backward();nn.utils.clip_grad_norm_(model.parameters(),5.);optimizer.step()
                    numerator+=float(per_cell.detach().sum());denominator+=float(norm)
            losses.append(numerator/denominator)
        history.append(dict(epoch=epoch,training_loss=losses[0],validation_loss=losses[1]))
        if losses[1]<best-args.min_delta:
            best=losses[1];stale=0
            torch.save(dict(state_dict={k:v.detach().cpu() for k,v in model.state_dict().items()},
                model_config=model.config,model_type=kind,dataset=ds,relation=getattr(args,'relation','state_15b') if kind=='gat_gine' else None,
                best_epoch=epoch,validation_loss=best,seed=args.seed,
                gene_symbols=genes,training_observed_gene_ids=np.flatnonzero(observed).tolist() if observed is not None else None,
                training_cell_ids=cells.cell_id.iloc[train].tolist(),validation_cell_ids=cells.cell_id.iloc[val].tolist(),
                class_weights=weight.cpu().tolist(),parameters=vars(args)),dest/'checkpoint.pt')
        else:stale+=1
        pd.DataFrame(history).to_csv(dest/'training_history.tsv',sep='\t',index=False)
        if epoch==1 or epoch%5==0:print(ds,kind,'epoch',epoch,'train',round(losses[0],5),'val',round(losses[1],5),'stale',stale,flush=True)
        if stale>=args.patience:break
    del optimizer,model
    if torch.cuda.is_available():torch.cuda.empty_cache()
    frozen,checkpoint=restore(dest/'checkpoint.pt',args.device)
    pred=predict(frozen,data,cells,kind,args.device,batch_size)
    pred.to_csv(dest/'predictions.tsv',sep='\t',index=False,float_format='%.17g')
    result={s:metrics(pred[pred.split.eq(s)]) for s in ['train','validation','test']}
    result.update(best_epoch=checkpoint['best_epoch'],epochs_run=epoch,training_seconds=time.perf_counter()-start,
        model_type=kind,dataset=ds,parameter_count=sum(p.numel() for p in frozen.parameters()),
        selection='save when global class-weighted validation cross entropy improves by more than min_delta; fixed threshold 0.5; no test selection',
        scope='stratified cell-level 70/20/10; not independent patient or sample generalization')
    write_json(dest/'metrics.json',result)
    write_json(dest/'run_parameters.json',vars(args))
    print('TEST',ds,kind,result['test'],flush=True)
    del data,frozen
    if torch.cuda.is_available():torch.cuda.empty_cache()


def transfer(args,root):
    source=getattr(args,'source_dataset','GSE223779')
    for kind in args.models:
        file=output_root(args,root)/'within'/source/kind/'checkpoint.pt'
        model,ckpt=restore(file,args.device)
        for target in args.datasets:
            if target==source:continue
            data,cells,genes=load_cohort(root,target,kind,getattr(args,"relation","state_15b"))
            assert not set(cells.cell_id)&set(ckpt['training_cell_ids']+ckpt['validation_cell_ids'])
            extras={}
            if kind=='gat_gine':
                assert genes==ckpt['gene_symbols'], 'Global gene vocabulary/order differs'
                seen=np.zeros(len(genes),bool);seen[ckpt['training_observed_gene_ids']]=True
                all_ids=np.concatenate([g.gene_id.numpy() for g in data])
                extras=dict(fraction_target_node_occurrences_unseen_in_source_training=float((~seen[all_ids]).mean()),
                    n_target_unique_genes_unseen=int((~seen[np.unique(all_ids)]).sum()))
            size=args.graph_batch_size if kind=='gat_gine' else args.mlp_batch_size
            pred=predict(model,data,cells,kind,args.device,size)
            dest=output_root(args,root)/'transfer'/f'{source}_to_{target}'/kind;dest.mkdir(parents=True,exist_ok=True)
            pred.to_csv(dest/'predictions.tsv',sep='\t',index=False,float_format='%.17g')
            result=dict(source_dataset=source,target_dataset=target,model_type=kind,relation=getattr(args,'relation','state_15b') if kind=='gat_gine' else None,checkpoint=str(file.resolve()),
                target_test=metrics(pred[pred.split.eq('test')]),target_all_labeled=metrics(pred),
                source_scaler_unchanged=True,target_training=False,target_threshold_tuning=False,**extras,
                scope='direct cross-disease/cross-drug transfer of reference response labels; not drug-specific mechanistic validation')
            write_json(dest/'metrics.json',result)
            print('TRANSFER',source,target,kind,result['target_test'],flush=True)
            del data
        del model
        if torch.cuda.is_available():torch.cuda.empty_cache()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--open_path',required=True)
    p.add_argument('--relation',choices=['state_15b','esm2_650m','qwen','ppi'],default='state_15b')
    p.add_argument('--stage',choices=['train','transfer','all'],default='all')
    p.add_argument('--datasets',nargs='+',choices=DATASETS,default=SOURCE_DATASETS)
    p.add_argument('--source_dataset',choices=SOURCE_DATASETS,default='GSE223779',
                   help='Source checkpoint cohort for frozen transfer')
    p.add_argument('--models',nargs='+',choices=['gat_gine','state_mlp'],default=['gat_gine','state_mlp'])
    p.add_argument('--device',default='cuda');p.add_argument('--seed',type=int,default=42)
    p.add_argument('--hidden',type=int,default=64);p.add_argument('--layers',type=int,default=2)
    p.add_argument('--heads',type=int,default=4);p.add_argument('--gene_dim',type=int,default=16)
    p.add_argument('--dropout',type=float,default=.2);p.add_argument('--lr',type=float,default=.001)
    p.add_argument('--weight_decay',type=float,default=.0001)
    p.add_argument('--graph_batch_size',type=int,default=32);p.add_argument('--mlp_batch_size',type=int,default=256)
    p.add_argument('--epochs',type=int,default=100);p.add_argument('--patience',type=int,default=15)
    p.add_argument('--min_delta',type=float,default=.0001);p.add_argument('--threads',type=int,default=4)
    a=p.parse_args();root=Path(a.open_path).resolve();torch.set_num_threads(a.threads)
    if a.device=='cuda' and not torch.cuda.is_available():raise RuntimeError('CUDA unavailable')
    if a.stage in ['train','all']:
        for ds in a.datasets:
            for kind in a.models:train_one(a,root,ds,kind)
    if a.stage in ['transfer','all']:transfer(a,root)


if __name__=='__main__':main()
