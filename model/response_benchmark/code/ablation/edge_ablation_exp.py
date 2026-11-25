"""Retrain one shared no-edge model and compare transfer with four full graph models."""

# Keep established imports and experiment-relative paths available.
import sys as _sys
from pathlib import Path as _Path
_code_root = _Path(__file__).resolve().parents[1]
if str(_code_root) not in _sys.path:
    _sys.path.insert(0, str(_code_root))
from benchmark_paths import CODE_ROOT, BENCH_ROOT, PROJECT_ROOT, script_path

import argparse
import gc
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pandas as pd
import torch
import train_response_exp as core
from edge_null_graphs import remove_edges
from validate_results_exp import check_metrics

RELATIONS=['esm2_650m','qwen','state_15b','ppi']
SOURCE='GSE223779'
TARGETS=['GSE140440','GSE163836','GSE202234','GSE223779_TS485T']


def job_root(root):
    return root/'output/transfer/edge_ablation/no_edges/shared'


def full_root(root,relation):
    return root/'output' if relation=='state_15b' else root/'output/transfer'/relation


def load_variant(root,dataset,loader=None):
    loader=loader or core.load_cohort
    data,cells,genes=loader(root,dataset,'gat_gine','state_15b')
    rows=[]
    for g in data:
        x=g.x.clone();ids=g.gene_id.clone()
        item=remove_edges(g)
        assert torch.equal(x,g.x) and torch.equal(ids,g.gene_id)
        rows.append(item)
    qc=pd.DataFrame(rows)
    folder=job_root(root)/'graph_qc';folder.mkdir(parents=True,exist_ok=True)
    path=folder/f'{dataset}.tsv'
    if path.exists():
        old=pd.read_csv(path,sep='\t')
        pd.testing.assert_frame_equal(qc,old[qc.columns],check_dtype=False)
    qc.to_csv(path,sep='\t',index=False)
    assert all(g.edge_index.shape==(2,0) and g.edge_attr.shape==(0,1) for g in data)
    print('NO EDGES',dataset,len(data),'cells; inputs preserved',flush=True)
    return data,cells,genes


def run(root,device):
    original_loader=core.load_cohort
    c=torch.load(root/'output/within'/SOURCE/'gat_gine/checkpoint.pt',map_location='cpu',weights_only=False)
    cfg=SimpleNamespace(**c['parameters'])
    cfg.relation='state_15b';cfg.models=['gat_gine'];cfg.datasets=[SOURCE]+TARGETS
    cfg.device=device;cfg.open_path=str(root);cfg.ablation='no_edges'
    torch.set_num_threads(cfg.threads)
    def controlled_loader(root,dataset,kind,relation='state_15b'):
        assert kind=='gat_gine'
        return load_variant(root,dataset,original_loader)
    core.load_cohort=controlled_loader
    core.output_root=lambda args,root:job_root(root)
    try:
        core.train_one(cfg,root,SOURCE,'gat_gine')
        core.transfer(cfg,root)
    finally:core.load_cohort=original_loader
    for file in [job_root(root)/'within'/SOURCE/'gat_gine/metrics.json']+list((job_root(root)/'transfer').glob('*/*/metrics.json')):
        value=json.loads(file.read_text())
        value.update(edge_variant='no_edges',graph_relation='shared',shared_no_edges_model=True)
        core.write_json(file,value)


def validate(root,device):
    torch.set_num_threads(4);checks=[]
    folder=job_root(root);source=folder/'within'/SOURCE/'gat_gine'
    model,ckpt=core.restore(source/'checkpoint.pt',device)
    assert ckpt['parameters']['ablation']=='no_edges'
    full=torch.load(root/'output/within'/SOURCE/'gat_gine/checkpoint.pt',map_location='cpu',weights_only=False)
    for key in ['seed','hidden','layers','heads','gene_dim','dropout','lr','weight_decay','graph_batch_size','epochs','patience','min_delta']:
        assert ckpt['parameters'][key]==full['parameters'][key],key
    assert ckpt['gene_symbols']==full['gene_symbols']
    hist=pd.read_csv(source/'training_history.tsv',sep='\t');best=float('inf');epoch=None
    for row in hist.itertuples():
        if row.validation_loss<best-ckpt['parameters']['min_delta']:best=row.validation_loss;epoch=row.epoch
    assert epoch==ckpt['best_epoch'] and np.isclose(best,ckpt['validation_loss'])
    for ds in [SOURCE]+TARGETS:
        data,cells,genes=load_variant(root,ds)
        assert genes==ckpt['gene_symbols']
        path=source if ds==SOURCE else folder/'transfer'/f'{SOURCE}_to_{ds}'/'gat_gine'
        pred=pd.read_csv(path/'predictions.tsv',sep='\t',float_precision='round_trip')
        saved=json.loads((path/'metrics.json').read_text())
        for col in ['cell_id','y','split']:assert pred[col].tolist()==cells[col].tolist()
        if ds==SOURCE:
            train=np.flatnonzero(cells.split.eq('train'));val=np.flatnonzero(cells.split.eq('validation'))
            assert ckpt['training_cell_ids']==cells.cell_id.iloc[train].tolist()==full['training_cell_ids']
            assert ckpt['validation_cell_ids']==cells.cell_id.iloc[val].tolist()==full['validation_cell_ids']
            assert np.allclose(ckpt['class_weights'],len(train)/(2*np.bincount(cells.y.iloc[train],minlength=2)))
            mean,std,seen=core.training_statistics(data,train,len(genes))
            assert np.isclose(model.expr_mean.item(),mean,rtol=1e-6) and np.isclose(model.expr_std.item(),std,rtol=1e-6)
            assert ckpt['training_observed_gene_ids']==np.flatnonzero(seen).tolist()
            assert sum(p.numel() for p in model.parameters())==saved['parameter_count']
            for split in ['train','validation','test']:check_metrics(pred[pred.split.eq(split)],saved[split])
            idx=np.r_[train[:3],val[:3],np.flatnonzero(cells.split.eq('test'))]
        else:
            assert not set(cells.cell_id)&set(ckpt['training_cell_ids']+ckpt['validation_cell_ids'])
            assert not saved['target_training'] and not saved['target_threshold_tuning'] and saved['source_scaler_unchanged']
            assert Path(saved['checkpoint'])==source/'checkpoint.pt'
            check_metrics(pred,saved['target_all_labeled']);check_metrics(pred[pred.split.eq('test')],saved['target_test'])
            idx=np.r_[np.flatnonzero(cells.split.eq('test')),np.flatnonzero(cells.y.lt(0))[:1]]
            for relation in RELATIONS:
                ref=pd.read_csv(full_root(root,relation)/'transfer'/f'{SOURCE}_to_{ds}'/'gat_gine/predictions.tsv',sep='\t')
                for col in ['cell_id','y','split']:assert pred[col].tolist()==ref[col].tolist()
        fresh=core.predict(model,[data[i] for i in idx],cells.iloc[idx].reset_index(drop=True),'gat_gine',device,32)
        assert np.allclose(fresh.margin_resistant_minus_sensitive,pred.iloc[idx].margin_resistant_minus_sensitive,rtol=2e-5,atol=2e-5)
        assert np.allclose(fresh.probability_resistant,pred.iloc[idx].probability_resistant,rtol=2e-5,atol=2e-6)
        checks.append(dict(dataset=ds,n_cells=len(cells),n_fresh_predictions=len(idx),status='passed'))
        del data;gc.collect()
    core.write_json(root/'output/summary/edge_ablation_validation.json',dict(status='passed',source_models_checked=1,transfer_models_checked=3,checks=checks))
    print('Shared no-edge checkpoint and three transfers verified.',flush=True)


def report(root):
    rows=[];effects=[]
    for relation in RELATIONS:
        for target in TARGETS:
            locations={'full':full_root(root,relation),'no_edges':job_root(root)}
            saved={v:json.loads((base/'transfer'/f'{SOURCE}_to_{target}'/'gat_gine/metrics.json').read_text()) for v,base in locations.items()}
            for scope in ['target_test','target_all_labeled']:
                for variant,m in saved.items():
                    rows.append(dict(relation=relation,target=target,variant=variant,scope=scope,shared_control=variant=='no_edges',
                                     **{k:v for k,v in m[scope].items() if not isinstance(v,list)}))
                effects.append(dict(relation=relation,target=target,scope=scope,
                    **{f'delta_{key}':saved['full'][scope][key]-saved['no_edges'][scope][key]
                       for key in ['auroc','balanced_accuracy','auprc_resistant','auprc_sensitive','mcc','macro_f1']}))
    dest=root/'output/summary';dest.mkdir(exist_ok=True,parents=True)
    stats=pd.DataFrame(rows);delta=pd.DataFrame(effects)
    stats.to_csv(dest/'edge_ablation_metrics.tsv',sep='\t',index=False,float_format='%.17g')
    delta.to_csv(dest/'edge_ablation_effects.tsv',sep='\t',index=False,float_format='%.17g')
    test=stats[stats.scope.eq('target_test')];mean=test.groupby(['relation','variant']).auroc.mean().unstack().reindex(RELATIONS)
    lines=['# 去掉全部边：四图迁移消融','',
        '仅比较完整图与无边模型。源为GSE223779，目标为GSE140440、GSE149383、GSE186960。完整图复用既有结果；新增一个共享无边源模型及三项迁移，没有随机重连消融。','',
        '## 三目标等权平均 AUROC','','| 图 | 完整图 | 无边 | 完整−无边 |','|---|---:|---:|---:|']
    for relation,row in mean.iterrows():lines.append(f'| {relation} | {row["full"]:.3f} | {row["no_edges"]:.3f} | {row["full"]-row["no_edges"]:+.3f} |')
    lines+=['','## 各目标测试集 AUROC','','| 图 | 目标 | 完整图 | 无边 | 完整−无边 |','|---|---|---:|---:|---:|']
    for (relation,target),block in test.groupby(['relation','target'],sort=False):
        values=block.set_index('variant').auroc
        lines.append(f'| {relation} | {target} | {values["full"]:.3f} | {values["no_edges"]:.3f} | {values["full"]-values["no_edges"]:+.3f} |')
    ba=test.groupby(['relation','variant']).balanced_accuracy.mean().unstack().reindex(RELATIONS)
    lines+=['','## 三目标等权平均 balanced accuracy','','| 图 | 完整图 | 无边 | 完整−无边 |','|---|---:|---:|---:|']
    for relation,row in ba.iterrows():lines.append(f'| {relation} | {row["full"]:.3f} | {row["no_edges"]:.3f} | {row["full"]-row["no_edges"]:+.3f} |')
    lines+=['','## 当前证据','']
    changes=delta[delta.scope.eq('target_test')]
    for relation,row in mean.iterrows():
        block=changes[changes.relation.eq(relation)]
        count=int(block.delta_auroc.gt(0).sum())
        difference=row['full']-row['no_edges']
        conclusion='仅部分目标为正，不是所有目标均有提升' if 0<count<3 else ('本轮三个目标均为正向差值，仍需独立重复' if count==3 else '本轮不支持该图边带来迁移收益')
        lines.append(f'- {relation}：平均AUROC差值{difference:+.3f}，{count}/3个目标为正；{conclusion}。')
    lines+=['','## 统计与结论边界','',
        '- 无边模型保留相同节点、表达、可训练gene identity、参数数量、残差、GINE自身项和池化，删除全部跨基因边；不重新选择signature。四图去边后的输入相同，因此只训练一次。表内共享数值不是四次独立重复。',
        '- 仅在源训练细胞拟合缩放、在源验证细胞选择checkpoint。目标不训练、不校准、不翻转方向，概率阈值0.5。架构、训练预算、seed=42及70/20/10细胞划分与完整图一致。',
        '- 主表单位为图×目标的原有test细胞；指标另报全体有标签目标，未知标签不计。AUROC/AP使用耐药减敏感logit；敏感AP取其相反数。等权均值是三个目标指标的平均，不合并细胞。差值均为完整图减无边。',
        '- 完整图优于无边，支持在当前设定下加入边及跨基因消息传递有收益；不优于无边则不支持其边有益。不能由此单独证明PPI或embedding特异连接优于随机连接，也未拆分拓扑与权重贡献。',
        '- 本次仅一个训练种子，没有独立重复的显著性结论；跨疾病/药物标签迁移、signature长度及细胞级划分的限制仍保留，不能解释为已证实的因果基因交互。',
        '',f'[完整指标]({(dest/"edge_ablation_metrics.tsv").as_posix()}) · [差值]({(dest/"edge_ablation_effects.tsv").as_posix()}) · [核验]({(dest/"edge_ablation_validation.json").as_posix()}) · [命令]({(root/"code/exp.md").as_posix()})']
    (dest/'边效应消融结果解读.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(mean.to_string(),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--open_path',required=True)
    p.add_argument('--stage',choices=['all','run','validate','report'],default='all');p.add_argument('--device',default='cuda')
    a=p.parse_args();root=Path(a.open_path).resolve()
    if a.stage in ['all','run']:run(root,a.device)
    if a.stage in ['all','validate']:validate(root,a.device)
    if a.stage in ['all','report']:report(root)


if __name__=='__main__':main()
