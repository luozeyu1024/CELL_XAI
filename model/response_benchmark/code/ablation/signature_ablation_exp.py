"""Train a structure-matched random-signature STATE model and compare direct transfer."""

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
from cell_graph_dataset import CellGraphDataset
from validate_results_exp import check_metrics

RELATIONS=['state_15b']
SOURCE='GSE223779'
TARGETS=['GSE140440','GSE163836','GSE202234','GSE223779_TS485T']

from split_registry import load_cells


def job_root(root):
    return root/'output/transfer/signature_ablation/random_genes/state_15b'


def full_root(root,relation):
    return root/'output' if relation=='state_15b' else root/'output/transfer'/relation


def package_root(root):
    return PROJECT_ROOT/'gene_sim_pathway/data/cell_signature_graphs_state_ablation'


def load_variant(root,dataset,loader=None):
    package=package_root(root)
    status=json.loads((package/'validation.json').read_text())
    assert status['status']=='passed'
    info=json.loads((package/dataset/'parameters.json').read_text())
    assert info['n_matching_passed']==info['n_cells']
    raw=CellGraphDataset(package,dataset,'state_15b')
    cells=load_cells(root/'data/cohorts'/dataset/'cells.tsv',dataset)
    assert raw.cells.cell_id.tolist()==cells.cell_id.tolist()
    graphs=[]
    for row in cells.itertuples():
        g=raw.to_pyg(int(row.graph_row))
        assert g.cell_id==row.cell_id and g.x.shape[1]==1
        g.y=torch.tensor([row.y],dtype=torch.long)
        assert torch.isfinite(g.x).all() and g.x.gt(0).all()
        g.validate(raise_on_error=True);graphs.append(g)
    print('RANDOM SIGNATURE',dataset,len(graphs),'STATE graphs',flush=True)
    return graphs,cells,raw.genes.tolist()


def run(root,device):
    original_loader=core.load_cohort
    c=torch.load(root/'output/within'/SOURCE/'gat_gine/checkpoint.pt',map_location='cpu',weights_only=False)
    cfg=SimpleNamespace(**c['parameters'])
    cfg.relation='state_15b';cfg.models=['gat_gine'];cfg.datasets=[SOURCE]+TARGETS
    cfg.device=device;cfg.open_path=str(root);cfg.ablation='random_genes_structure_matched';cfg.signature_package=str(package_root(root))
    cfg.signature_sampling_parameters={ds:{k:v for k,v in json.loads((package_root(root)/ds/'parameters.json').read_text()).items() if k in ['seed','count_jitter','density_tolerance','degree_cdf_tolerance','isolation_tolerance','max_steps','restarts']} for ds in [SOURCE]+TARGETS}
    previous=job_root(root)/'within'/SOURCE/'gat_gine/run_parameters.json'
    if previous.exists():assert json.loads(previous.read_text())['signature_sampling_parameters']==cfg.signature_sampling_parameters
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
        value.update(signature_variant='random_genes_structure_matched',graph_relation='state_15b',signature_package=str(package_root(root)))
        core.write_json(file,value)


def validate(root,device):
    torch.set_num_threads(4);checks=[]
    folder=job_root(root);source=folder/'within'/SOURCE/'gat_gine'
    model,ckpt=core.restore(source/'checkpoint.pt',device)
    assert ckpt['parameters']['ablation']=='random_genes_structure_matched'
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
    core.write_json(root/'output/summary/signature_ablation_validation.json',dict(status='passed',source_models_checked=1,transfer_models_checked=3,checks=checks))
    print('Random-signature checkpoint and three transfers verified.',flush=True)


def report(root):
    rows=[];effects=[]
    for target in TARGETS:
        locations={'signature_state':root/'output',
            'random_signature_state':job_root(root),
            'signature_no_edges':root/'output/transfer/edge_ablation/no_edges/shared'}
        values={name:json.loads((base/'transfer'/f'{SOURCE}_to_{target}'/'gat_gine/metrics.json').read_text()) for name,base in locations.items()}
        for scope in ['target_test','target_all_labeled']:
            for variant,m in values.items():
                rows.append(dict(source=SOURCE,target=target,variant=variant,scope=scope,
                    **{k:v for k,v in m[scope].items() if not isinstance(v,list)}))
            for control in ['random_signature_state','signature_no_edges']:
                effects.append(dict(target=target,scope=scope,comparison='signature_state_minus_'+control,
                    **{f'delta_{key}':values['signature_state'][scope][key]-values[control][scope][key]
                       for key in ['auroc','balanced_accuracy','auprc_resistant','auprc_sensitive','mcc','macro_f1']}))
    dest=root/'output/summary';stats=pd.DataFrame(rows);delta=pd.DataFrame(effects)
    stats.to_csv(dest/'signature_ablation_metrics.tsv',sep='\t',index=False,float_format='%.17g')
    delta.to_csv(dest/'signature_ablation_effects.tsv',sep='\t',index=False,float_format='%.17g')
    test=stats[stats.scope.eq('target_test')]
    order=['signature_state','random_signature_state','signature_no_edges']
    means=test.groupby('variant')[['auroc','balanced_accuracy']].mean().reindex(order)
    lines=['# signature归因选基因消融：STATE-15B迁移','',
        '原signature＋STATE图，与结构匹配的随机gene signature＋同一STATE图比较；此前原signature无边模型作为背景参照。只新增一个GSE223779源模型及三个目标迁移。','',
        '## 三目标等权平均','',
        '| 输入与图 | AUROC | Balanced accuracy |','|---|---:|---:|']
    for variant,row in means.iterrows():lines.append(f'| {variant} | {row.auroc:.3f} | {row.balanced_accuracy:.3f} |')
    lines+=['','## 各目标原有测试细胞','',
        '| 目标 | 设置 | 细胞数 | AUROC | BA | AP耐药 | AP敏感 |','|---|---|---:|---:|---:|---:|---:|']
    for target in TARGETS:
        block=test[test.target.eq(target)].set_index('variant')
        for variant in order:
            row=block.loc[variant]
            lines.append(f'| {target} | {variant} | {row.n_cells} | {row.auroc:.3f} | {row.balanced_accuracy:.3f} | {row.auprc_resistant:.3f} | {row.auprc_sensitive:.3f} |')
    changes=delta[(delta.scope=='target_test')&(delta.comparison=='signature_state_minus_random_signature_state')]
    mean_change=means.loc['signature_state','auroc']-means.loc['random_signature_state','auroc']
    lines+=['','## 当前最支持的结论','',
        f'原signature相对随机signature的三目标平均AUROC差值为 **{mean_change:+.3f}**；{int(changes.delta_auroc.gt(0).sum())}/3个目标的差值为正。',
        '正差值支持在本结构匹配对照下原signature有额外预测信息；负差值不支持其优于该随机对照。应逐目标解释，不能用均值掩盖反例。',
        f'原signature＋STATE的平均AUROC为{means.loc["signature_state","auroc"]:.3f}；相对随机对照更好不等于获得可靠的跨队列耐药预测。','',
        '## 统计与方法','',
        '- 候选背景固定为原来的实测、唯一HGNC映射、SCimilarity及四图共同词表，再保留该细胞表达为正的基因；原signature的所选基因表达也均为正。随机选择不读取IG数值或排名。',
        '- 每细胞节点数在原数量附近约±5%随机抽取；用原STATE诱导子图作结构参照，边密度相对误差≤2%、度数经验CDF最大距离≤0.05、孤立比例绝对差≤0.02。困难细胞仅增加随机搜索并在同一范围内重抽数量，不使用标签或预测结果。',
        '- 固定STATE全局加权边，不重连、不重新训练蛋白向量。原图节点数、结构及度数分层用于约束随机对照；因此是结构条件化的启发式随机signature，不是所有基因的完全均匀随机抽样。',
        '- 图节点输入只有同样预处理的表达和可训练gene identity；随机对照不伪造归因特征。仅源训练节点拟合表达缩放，源验证集选checkpoint；目标不训练、不校准，阈值0.5。共享原70/20/10细胞划分、架构、训练参数和seed=42。',
        '- 主比较为各目标相同test细胞；另保存全部已标注目标指标，未知标签不计指标。AUROC/AP使用logit差，等权均值按三个数据集计算。',
        '- 某些细胞表达候选接近所需节点数，随机集与原集可能重合较多；不强行排除原基因，以免改变可选背景或导致无法构图。',
        '- 原SIGnature–IG解释的是SCimilarity细胞embedding，不是耐药分类logit；此处检验它作为节点筛选策略的迁移用途，不等同于检验归因公式正确与否。',
        '- 单个随机signature实例、单训练种子；不提供独立重复显著性或因果证明。数量与结构近似匹配并非逐图完全相同，差值仍可能受残余结构/表达差异影响；当前不是完整2×2节点选择×有无边交互实验。',
        '',f'[完整指标]({(dest/"signature_ablation_metrics.tsv").as_posix()}) · [差值]({(dest/"signature_ablation_effects.tsv").as_posix()}) · [核验]({(dest/"signature_ablation_validation.json").as_posix()}) · [命令]({(root/"code/exp.md").as_posix()})']
    (dest/'signature归因消融结果解读.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(means.to_string(),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--open_path',required=True)
    p.add_argument('--stage',choices=['all','run','validate','report'],default='all');p.add_argument('--device',default='cuda')
    a=p.parse_args();root=Path(a.open_path).resolve()
    if a.stage in ['all','run']:run(root,a.device)
    if a.stage in ['all','validate']:validate(root,a.device)
    if a.stage in ['all','report']:report(root)


if __name__=='__main__':main()
