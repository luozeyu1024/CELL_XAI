"""Train source graph models, transfer unchanged, and compare four graph priors."""

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
import subprocess
import sys
import numpy as np
import pandas as pd
import torch
from train_response_exp import load_cohort, restore, predict, training_statistics, write_json
from validate_results_exp import check_metrics

RELATIONS=['esm2_650m','qwen','ppi']
TARGETS=['GSE140440','GSE163836','GSE202234','GSE223779_TS485T']
SOURCE='GSE223779'


def validate(root,device):
    torch.set_num_threads(4)
    checks=[]
    for relation in RELATIONS:
        base=root/'output/transfer'/relation
        folder=base/'within'/SOURCE/'gat_gine'
        model,ckpt=restore(folder/'checkpoint.pt',device)
        assert ckpt['relation']==relation and ckpt['dataset']==SOURCE
        data,cells,genes=load_cohort(root,SOURCE,'gat_gine',relation)
        train=np.flatnonzero(cells.split.eq('train'))
        val=np.flatnonzero(cells.split.eq('validation'))
        assert ckpt['training_cell_ids']==cells.cell_id.iloc[train].tolist()
        assert ckpt['validation_cell_ids']==cells.cell_id.iloc[val].tolist()
        assert ckpt['gene_symbols']==genes
        mean,std,observed=training_statistics(data,train,len(genes))
        assert np.isclose(model.expr_mean.item(),mean,rtol=1e-6)
        assert np.isclose(model.expr_std.item(),std,rtol=1e-6)
        assert ckpt['training_observed_gene_ids']==np.flatnonzero(observed).tolist()
        hist=pd.read_csv(folder/'training_history.tsv',sep='\t')
        best=float('inf');best_epoch=None
        for row in hist.itertuples():
            if row.validation_loss<best-ckpt['parameters']['min_delta']:
                best=row.validation_loss;best_epoch=row.epoch
        assert best_epoch==ckpt['best_epoch'] and np.isclose(best,ckpt['validation_loss'])
        saved=json.loads((folder/'metrics.json').read_text())
        stored=pd.read_csv(folder/'predictions.tsv',sep='\t',float_precision='round_trip')
        assert stored.cell_id.tolist()==cells.cell_id.tolist()
        assert stored.y.tolist()==cells.y.tolist() and stored.split.tolist()==cells.split.tolist()
        for split in ['train','validation','test']:
            check_metrics(stored[stored.split.eq(split)],saved[split])
        idx=np.concatenate([train[:3],val[:3],np.flatnonzero(cells.split.eq('test'))])
        fresh=predict(model,[data[i] for i in idx],cells.iloc[idx].reset_index(drop=True),'gat_gine',device,32)
        assert np.allclose(fresh.margin_resistant_minus_sensitive,stored.iloc[idx].margin_resistant_minus_sensitive,atol=2e-5,rtol=2e-5)
        checks.append(dict(relation=relation,dataset=SOURCE,scope='source',n_reloaded_predictions=len(idx),status='passed'))
        del data;gc.collect()
        for target in TARGETS:
            data,cells,genes=load_cohort(root,target,'gat_gine',relation)
            assert genes==ckpt['gene_symbols']
            assert not set(cells.cell_id)&set(ckpt['training_cell_ids']+ckpt['validation_cell_ids'])
            dest=base/'transfer'/f'{SOURCE}_to_{target}'/'gat_gine'
            saved=json.loads((dest/'metrics.json').read_text())
            assert saved['relation']==relation and not saved['target_training'] and not saved['target_threshold_tuning']
            assert Path(saved['checkpoint'])==folder/'checkpoint.pt'
            stored=pd.read_csv(dest/'predictions.tsv',sep='\t',float_precision='round_trip')
            assert stored.cell_id.tolist()==cells.cell_id.tolist()
            assert stored.y.tolist()==cells.y.tolist() and stored.split.tolist()==cells.split.tolist()
            check_metrics(stored,saved['target_all_labeled'])
            check_metrics(stored[stored.split.eq('test')],saved['target_test'])
            idx=np.concatenate([np.flatnonzero(cells.split.eq('test')),np.flatnonzero(cells.y.lt(0))[:1]])
            fresh=predict(model,[data[i] for i in idx],cells.iloc[idx].reset_index(drop=True),'gat_gine',device,32)
            assert np.allclose(fresh.margin_resistant_minus_sensitive,stored.iloc[idx].margin_resistant_minus_sensitive,atol=2e-5,rtol=2e-5)
            assert np.allclose(fresh.probability_resistant,stored.iloc[idx].probability_resistant,atol=2e-6,rtol=2e-5)
            checks.append(dict(relation=relation,dataset=target,scope='transfer',n_reloaded_predictions=len(idx),status='passed'))
            del data;gc.collect()
        del model
        if torch.cuda.is_available():torch.cuda.empty_cache()
    write_json(root/'output/summary/four_graph_validation.json',dict(status='passed',source_models_checked=3,transfer_models_checked=9,checks=checks))
    print('Three source checkpoints and nine direct transfers verified.',flush=True)


def report(root):
    rows=[]
    for relation in ['state_15b']+RELATIONS:
        base=root/'output' if relation=='state_15b' else root/'output/transfer'/relation
        for target in TARGETS:
            saved=json.loads((base/'transfer'/f'{SOURCE}_to_{target}'/'gat_gine/metrics.json').read_text())
            for scope in ['target_test','target_all_labeled']:
                rows.append(dict(source=SOURCE,target=target,relation=relation,model='gat_gine',scope=scope,
                    **{k:v for k,v in saved[scope].items() if not isinstance(v,list)}))
    for target in TARGETS:
        saved=json.loads((root/'output/transfer'/f'{SOURCE}_to_{target}'/'state_mlp/metrics.json').read_text())
        for scope in ['target_test','target_all_labeled']:
            rows.append(dict(source=SOURCE,target=target,relation='native_state_embedding',model='state_mlp',scope=scope,
                **{k:v for k,v in saved[scope].items() if not isinstance(v,list)}))
    df=pd.DataFrame(rows);dest=root/'output/summary';dest.mkdir(exist_ok=True,parents=True)
    df.to_csv(dest/'four_graph_comparison_metrics.tsv',sep='\t',index=False,float_format='%.17g')
    test=df[df.scope.eq('target_test')]
    table=test.pivot(index='target',columns='relation',values='auroc').reindex(TARGETS)
    columns=['esm2_650m','qwen','state_15b','ppi','native_state_embedding']
    table=table[columns]
    lines=['# 四种图的直接迁移比较','',
        '源数据集固定为 GSE223779；每种图训练一个 GATv2＋GINE，再直接推理三个目标 GSE。STATE-15B 和 STATE＋MLP 复用既有结果。','',
        '## 目标测试集 AUROC','',
        '| 目标数据集 | ESM2-650M | Qwen | STATE-15B | PPI | STATE＋MLP |',
        '|---|---:|---:|---:|---:|---:|']
    for name,row in table.iterrows():lines.append('| '+name+' | '+' | '.join(f'{row[c]:.3f}' for c in columns)+' |')
    means=table.mean();lines.append('| 三目标等权均值 | '+' | '.join(f'{means[c]:.3f}' for c in columns)+' |')
    ba=test.pivot(index='target',columns='relation',values='balanced_accuracy')[columns].reindex(TARGETS)
    lines+=['','## 目标测试集 balanced accuracy（阈值 0.5）','','| 目标数据集 | ESM2-650M | Qwen | STATE-15B | PPI | STATE＋MLP |','|---|---:|---:|---:|---:|---:|']
    for name,row in ba.iterrows():lines.append('| '+name+' | '+' | '.join(f'{row[c]:.3f}' for c in columns)+' |')
    lines+=['','## 统计口径','',
        '- 相同 signature 节点、表达与 gene identity、全局基因词表、共享 70/20/10 划分、seed=42、结构与超参数；只更换图及其原始边权。PPI 使用已有 conn≥0.5 图，KNN 使用已有 top30 图。',
        '- 只训练源数据集；以源验证集加权交叉熵选 checkpoint，固定源缩放和阈值，目标不训练、不校准、不选阈值。新增三个源模型、九项迁移。源数据集内部指标仅作训练核查。',
        '- 主比较为各目标原先的相同测试细胞；完整有标签目标结果单独保存，未知标签不计指标。AUROC/双向 AUPRC 用 logit 差排名，balanced accuracy 用概率≥0.5。',
        '- 等权均值是三个目标数据集指标的算术平均，不把所有细胞拼成一个队列；缺少多随机种子与独立生物学重复，不能据小差异声称显著优越。',
        '- 跨疾病、跨药物、细胞级划分及既有 signature 选择的限制沿用主实验。图先验不等同于细胞内已测物理互作。',
        '',f'按三目标等权 AUROC，本轮最高为 **{means.idxmax()}（{means.max():.3f}）**；具体表现须同时看各目标及 balanced accuracy，不能据单一均值宣称通用耐药机制。',
        '',f'[完整指标]({(dest/"four_graph_comparison_metrics.tsv").as_posix()})；[复现命令]({(root/"code/exp.md").as_posix()})。']
    (dest/'四图迁移结果解读.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(table.to_string(),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--open_path',required=True)
    p.add_argument('--stage',choices=['all','run','validate','report'],default='all')
    p.add_argument('--device',default='cuda')
    a=p.parse_args();root=Path(a.open_path).resolve()
    if a.stage in ['all','run']:
        for relation in RELATIONS:
            cmd=[sys.executable,str(script_path('train_response_exp.py')),'--open_path',str(root),
                '--models','gat_gine','--relation',relation,'--device',a.device,'--seed','42',
                '--hidden','64','--layers','2','--heads','4','--gene_dim','16','--dropout','0.2',
                '--lr','0.001','--weight_decay','0.0001','--graph_batch_size','32','--mlp_batch_size','256',
                '--epochs','100','--patience','15','--min_delta','0.0001','--threads','4']
            subprocess.run(cmd+['--stage','train','--datasets',SOURCE],check=True)
            subprocess.run(cmd+['--stage','transfer','--datasets',SOURCE]+TARGETS,check=True)
    if a.stage in ['all','validate']:validate(root,a.device)
    if a.stage in ['all','report']:report(root)


if __name__=='__main__':main()
