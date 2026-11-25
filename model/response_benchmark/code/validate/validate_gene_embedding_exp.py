"""Verify frozen feature lookup and exact reuse of the original graph computation."""

# Keep established imports and experiment-relative paths available.
import sys as _sys
from pathlib import Path as _Path
_code_root = _Path(__file__).resolve().parents[1]
if str(_code_root) not in _sys.path:
    _sys.path.insert(0, str(_code_root))
from benchmark_paths import CODE_ROOT, BENCH_ROOT, PROJECT_ROOT, script_path

from pathlib import Path
import argparse
import json
import numpy as np
import torch
import pandas as pd
from graph_model import GraphClassifier
from gene_embedding_model import GeneEmbeddingClassifier


def audit_reference_statistics(root):
    from train_response_exp import metrics
    project=root.parents[1]
    reference=pd.read_csv(project/'work/transfer_analysis/data/reassessed/primary_comparison.tsv',sep='\t')
    records=[]
    for task in sorted(p.name for p in (root/'data/grouped').iterdir() if p.is_dir()):
        paths=sorted((root/'output/grouped_adaptive'/task).glob('fold_*/state_15b/within/*/gat_gine/predictions.tsv'))
        frames=[pd.read_csv(path,sep='\t',float_precision='round_trip').query("split == 'test'") for path in paths]
        test=pd.concat(frames,ignore_index=True)
        assert test.cell_id.is_unique
        if task=='GSE165897_pre_EOC':
            patients=test.groupby('patient_id').agg(y=('y','first'),probability_resistant=('probability_resistant','mean')).reset_index()
            actual=metrics(patients.assign(margin_resistant_minus_sensitive=patients.probability_resistant))
            scope='patient_grouped_OOF_patients'
        else:
            values=[metrics(frame) for frame in frames]
            actual={key:np.mean([value[key] for value in values]) for key in ['auroc','balanced_accuracy','auprc_resistant']}
            scope='sample_set_macro' if 'PC9_DTEP' in task else 'replicate_macro'
        expected=reference[(reference.dataset==task)&reference.scope.eq(scope)&reference.model.eq('state_15b')]
        assert len(expected)==1
        for key in ['auroc','balanced_accuracy','auprc_resistant']:
            error=abs(float(actual[key])-float(expected.iloc[0][key]))
            assert error<1e-10,(task,key,error)
            records.append(dict(dataset=task,scope=scope,metric=key,absolute_difference=error))
    out=project/'work/transfer_analysis/data/gene_embedding'
    out.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(records).to_csv(out/'reference_aggregation_audit.tsv',sep='\t',index=False)
    print('PASS: reference aggregation matches all',len(records),'saved reference comparisons')


def smoke_frozen_transfer(root):
    from gene_embedding_model import restore
    import train_response_exp as core
    path=root/'output/gene_embedding/state_graph__state_15b/within/GSE223779/gat_gine/checkpoint.pt'
    if not path.exists(): return
    model,saved=restore(path,'cpu')
    graphs,cells,genes=core.load_cohort(root,'GSE140440','gat_gine','state_15b')
    assert genes==saved['gene_symbols']
    assert not set(cells.cell_id)&set(saved['training_cell_ids']+saved['validation_cell_ids'])
    original=(model.expr_mean.clone(),model.expr_std.clone())
    for batch in core.batches(graphs,cells,np.array([0,1]),'gat_gine',2):
        logits,_=core.forward(model,batch,'gat_gine','cpu')
        assert logits.shape==(2,2) and torch.isfinite(logits).all()
        assert torch.all(model.identity.weight[batch.gene_id].square().sum(1)>0)
    assert torch.equal(original[0],model.expr_mean) and torch.equal(original[1],model.expr_std)
    print('PASS: two real target cells, exact gene order, disjoint source cells, unchanged source scaler')


def audit_completed_phase(root):
    from gene_embedding_benchmark_exp import jobs,destination
    variant='state_graph__state_15b'
    vocab=pd.read_csv(root/'data/gene_embeddings/gene_vocab.tsv',sep='\t').gene_symbol.tolist()
    coverage=pd.read_csv(root/'data/gene_embeddings/state_15b_coverage.tsv',sep='\t').mapped.to_numpy(bool)
    records=[]
    for job in jobs():
        folder=destination(job,variant)/'within'/job['dataset']/'gat_gine'
        saved=torch.load(folder/'checkpoint.pt',map_location='cpu',weights_only=False)
        result=json.loads((folder/'metrics.json').read_text())
        config=saved['model_config']
        assert config['hidden']==64 and config['layers']==2 and config['heads']==4 and config['gene_dim']==5120
        assert saved['state_dict']['project.weight'].shape==(64,5121)
        assert 'identity.weight' not in saved['state_dict']
        assert saved['gene_symbols']==vocab
        assert config['embedding_path']==str((root/'data/gene_embeddings/state_15b.npy').resolve())
        cohort=Path(job['cohort_root'])/'data/cohorts'/job['dataset']
        cells=pd.read_csv(cohort/'cells.tsv',sep='\t')
        train=cells[cells.split.eq('train')];validation=cells[cells.split.eq('validation')];test=cells[cells.split.eq('test')]
        assert train.cell_id.tolist()==saved['training_cell_ids']
        assert validation.cell_id.tolist()==saved['validation_cell_ids']
        assert not set(test.cell_id)&set(saved['training_cell_ids']+saved['validation_cell_ids'])
        info=json.loads((cohort/'data_sources.json').read_text())
        if 'group_key' in info:
            key=info['group_key']
            assert not set(test[key])&set(cells.loc[cells.split.isin(['train','validation']),key])
        total=square=0.;count=0;observed=np.zeros(len(vocab),bool)
        with np.load(Path(info['graph_source'])/'signatures.npz',allow_pickle=False) as archive:
            ptr=archive['ptr'];features=archive['features'];ids=archive['gene_id']
        for row in train.itertuples():
            lo,hi=ptr[int(row.graph_row):int(row.graph_row)+2]
            expression=features[lo:hi,0].astype(np.float64)
            total+=expression.sum();square+=(expression*expression).sum();count+=len(expression)
            observed[ids[lo:hi]]=True
        mean=total/count;std=max(square/count-mean*mean,1e-12)**.5
        assert np.isclose(mean,config['expr_mean'],rtol=1e-10,atol=1e-10)
        assert np.isclose(std,config['expr_std'],rtol=1e-10,atol=1e-10)
        assert np.array_equal(np.flatnonzero(observed),saved['training_observed_gene_ids'])
        missing=int((~coverage[observed]).sum());assert missing==0
        counts=np.bincount(train.y.to_numpy(int),minlength=2)
        weights=len(train)/(2*counts)
        assert np.allclose(weights,saved['class_weights'],rtol=1e-6,atol=1e-7)
        reload_error=result['checkpoint_reload_max_margin_error'];assert reload_error<=1e-4
        records.append(dict(dataset=job['dataset'],fold=job['fold'],n_train=len(train),
            n_validation=len(validation),n_test=len(test),n_train_node_occurrences=count,
            expression_mean_error=abs(mean-config['expr_mean']),expression_std_error=abs(std-config['expr_std']),
            missing_train_gene_embeddings=missing,checkpoint_reload_max_margin_error=reload_error,
            hidden=64,gene_feature_dimension=5120))
    assert len(records)==14
    out=root.parents[1]/'work/transfer_analysis/data/gene_embedding'
    pd.DataFrame(records).to_csv(out/'completed_phase_validation.tsv',sep='\t',index=False,float_format='%.17g')
    print('PASS: 14 checkpoints, original splits, group isolation, train-only scaling, frozen 5120D features')


def main(completed_phase=False):
    torch.set_num_threads(2);torch.manual_seed(42)
    root=BENCH_ROOT
    path=root/'data/gene_embeddings/qwen.npy'
    values=np.load(path,mmap_mode='r');n,d=values.shape
    new=GeneEmbeddingClassifier(n,embedding_path=str(path),dropout=0.)
    reference=GraphClassifier(n,gene_dim=d,dropout=0.)
    state=new.state_dict();state['identity.weight']=torch.from_numpy(np.array(values))
    reference.load_state_dict(state,strict=True)
    ids=torch.tensor([0,1,2,3]);x=torch.tensor([[0.],[1.],[2.],[3.]])
    edges=torch.tensor([[0,1,2,3],[1,0,3,2]])
    edge_attr=torch.tensor([[.9],[.9],[.7],[.7]]);batch=torch.tensor([0,0,1,1])
    new.eval();reference.eval()
    a=new(x,edges,edge_attr,batch,ids);b=reference(x,edges,edge_attr,batch,ids)
    torch.testing.assert_close(a,b,rtol=0,atol=0)
    a.sum().backward()
    assert new.identity.weight.grad is None
    assert new.project.weight.grad is not None
    assert 'identity.weight' not in new.state_dict()
    assert new.project.in_features==d+1 and new.project.out_features==64
    assert new.config['gene_dim']==d
    print('PASS: exact original forward, frozen lookup, trainable projection, 64 hidden dimensions')
    audit_reference_statistics(root)
    smoke_frozen_transfer(root)
    if completed_phase: audit_completed_phase(root)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--completed_phase',action='store_true')
    args=parser.parse_args();main(args.completed_phase)
