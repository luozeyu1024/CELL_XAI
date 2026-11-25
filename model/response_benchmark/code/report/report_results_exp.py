"""Summarize saved paired-model metrics without training or changing predictions."""

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
import pandas as pd


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--open_path',required=True)
    a=p.parse_args();root=Path(a.open_path).resolve();within=[];transfer=[]
    for path in sorted((root/'output/within').glob('*/*/metrics.json')):
        m=json.loads(path.read_text())
        for split in ['train','validation','test']:
            within.append(dict(dataset=m['dataset'],model=m['model_type'],split=split,best_epoch=m['best_epoch'],
                epochs_run=m['epochs_run'],parameter_count=m['parameter_count'],
                **{k:v for k,v in m[split].items() if k!='confusion_matrix_true_rows_sensitive_resistant'}))
    for path in sorted((root/'output/transfer').glob('*/*/metrics.json')):
        m=json.loads(path.read_text())
        for split in ['target_test','target_all_labeled']:
            transfer.append(dict(source=m['source_dataset'],target=m['target_dataset'],model=m['model_type'],scope=split,
                **{k:v for k,v in m[split].items() if k!='confusion_matrix_true_rows_sensitive_resistant'}))
    w=pd.DataFrame(within);t=pd.DataFrame(transfer)
    assert len(w)==24 and len(t)==12, 'Eight fitted models and six transfer results are required'
    (root/'output/summary').mkdir(parents=True,exist_ok=True)
    w.to_csv(root/'output/summary/within_metrics.tsv',sep='\t',index=False)
    t.to_csv(root/'output/summary/transfer_metrics.tsv',sep='\t',index=False)
    cohorts=pd.read_csv(root/'data/dataset_summary.tsv',sep='\t')
    def link(path):return f'[{path.name}]({path.resolve().as_posix()})'
    lines=['# 四个GSE的耐药参考标签分类','',
        '本轮比较同一细胞划分上的STATE-15B图GATv2＋GINE与原生SE细胞向量＋MLP；只训练和预测，不做消融或重新计算归因。','',
        '## 当前最支持的结论','',
        f'数据集内test的AUROC为{w.loc[w.split.eq("test"),"auroc"].min():.3f}–{w.loc[w.split.eq("test"),"auroc"].max():.3f}，而固定GSE223779模型直接迁移的目标test AUROC为{t.loc[t.scope.eq("target_test"),"auroc"].min():.3f}–{t.loc[t.scope.eq("target_test"),"auroc"].max():.3f}。当前证据支持各队列内部的参考标签高度可分，尚不支持最大单一队列训练后即可获得跨疾病、跨药物的通用耐药预测能力。',
        '高数据集内分数不能排除样本来源或批次差异；迁移结果也不能单独区分疾病、药物、来源样本、signature长度及表示流程差异的贡献。本轮不对这些因素做消融。','',
        '## 数据与划分','',
        '| GSE | 有标签细胞 | 训练 | 验证 | 测试 | 未知标签 |',
        '|---|---:|---:|---:|---:|---:|']
    for r in cohorts.itertuples():lines.append(f'| {r.dataset} | {r.n_labeled} | {r.n_train} | {r.n_validation} | {r.n_test} | {r.n_unknown} |')
    lines+=['', '按响应标签分层随机划分约70/20/10，seed=42，取整后以表内数量为准。两模型使用完全相同的细胞；未知标签不用于拟合或指标。','',
        '这是细胞级划分，不能作为独立患者或培养样本验证。既有adaptive signature K在划分前已选定；本轮不声称嵌套、全流程独立验证。','',
        f'## 数据集内测试结果 {link(root/"output/summary/within_metrics.tsv")}','',
        '单位为每个GSE×模型的留出测试细胞。checkpoint只按验证加权交叉熵和预设min_delta选择；概率阈值固定0.5。AUROC及耐药AUPRC使用耐药减敏感logit，敏感AUPRC使用相反数，避免极端概率的浮点饱和并列；Brier和分类使用float64 softmax概率。AUPRC随机排名基线是对应类别比例。','',
        '| GSE | 模型 | AUROC | AP耐药 | AP敏感 | Balanced accuracy | MCC | 最佳epoch |',
        '|---|---|---:|---:|---:|---:|---:|---:|']
    for r in w[w.split.eq('test')].itertuples():
        lines.append(f'| {r.dataset} | {r.model} | {r.auroc:.3f} | {r.auprc_resistant:.3f} | {r.auprc_sensitive:.3f} | {r.balanced_accuracy:.3f} | {r.mcc:.3f} | {r.best_epoch} |')
    lines+=['', f'## GSE223779直接迁移 {link(root/"output/summary/transfer_metrics.tsv")}','',
        '源数据集依据有标签细胞数预先选定，不按目标测试表现选源。训练好的checkpoint和源训练集缩放参数保持不变，不用目标标签训练、校准或调阈值。主表只使用目标原有test细胞，因此与上表可直接比较；TSV另报全体有标签目标细胞，不能将其与目标本地test指标混为一谈。','',
        '| 目标GSE | 模型 | test细胞 | AUROC | AP耐药 | AP敏感 | Balanced accuracy | MCC |',
        '|---|---|---:|---:|---:|---:|---:|---:|']
    for r in t[t.scope.eq('target_test')].itertuples():
        lines.append(f'| {r.target} | {r.model} | {r.n_cells} | {r.auroc:.3f} | {r.auprc_resistant:.3f} | {r.auprc_sensitive:.3f} | {r.balanced_accuracy:.3f} | {r.mcc:.3f} |')
    lines+=['', '## 结果如何解释','',
        '- 标签是各研究提供的敏感／耐药参考状态，不是每个细胞独立进行的药敏测量。高数据集内得分也可能利用细胞系、来源样本或处理批次差异。',
        '- GSE223779的细胞最多，但主要是一个类器官背景，不能称为疾病覆盖最全面的队列。迁移涉及不同疾病和药物；表现下降不等同于STATE或图模型失效，表现提高也不证明共同耐药机制。',
        '- 图模型的原有signature长度随GSE不同；模型支持变长图，但这仍可能影响迁移。目标中训练时未见基因的节点比例记录在迁移metrics.json，未在目标数据上重训这些identity。',
        '- 既有GSE186960/223779 SE输出复用历史原生向量；历史记录未保存权重校验，不能声称与新推理逐位一致。抽样重编码兼容性记录在data/encoder_compatibility_summary.tsv；表示流程的数值差异仍可能影响迁移。',
        '- 两模型的输入路径不同：图模型用SCimilarity–IG筛出的表达节点和STATE蛋白KNN边，MLP用完整SE细胞表示。当前比较不是严格参数量或输入信息量匹配的消融。','',
        '## 文件与复现','',
        '每模型目录包含checkpoint.pt、training_history.tsv、predictions.tsv、metrics.json和run_parameters.json。预测表保留全部可编码细胞，指标只使用已知标签；未知标签的预测不计作验证。训练、验证、测试的全部指标见汇总TSV，混淆矩阵与Brier见各metrics.json。','',
        f'环境与绝对路径命令：{link(root/"code/exp.md")}；数据来源和细胞对齐：{link(root/"data/dataset_summary.tsv")}。']
    comparison=root/'output/summary/four_graph_comparison_metrics.tsv'
    if comparison.exists():
        extra=pd.read_csv(comparison,sep='\t')
        scores=extra[extra.scope.eq('target_test')].groupby('relation').auroc.mean()
        lines+=['','## 四图直接迁移补充','',
            '另外三个图各新增一个GSE223779源模型及三个目标迁移；原STATE-15B和MLP复用。完成后共11个分类模型、15项迁移。',
            f'三目标等权AUROC最高为{scores.idxmax()}（{scores.max():.3f}）；单随机种子、跨疾病药物的结果尚不能证明通用耐药机制或显著方法优势。',
            f'逐目标四图与MLP比较见 {link(root/"output/summary/四图迁移结果解读.md")}；全部指标见 {link(comparison)}。']
    ablation=root/'output/summary/edge_ablation_metrics.tsv'
    if ablation.exists():
        lines+=['','## 去掉全部边的迁移消融','',
            '仅新增一个共享无边源模型及三个目标迁移；四种完整图模型复用。比较单位、逐目标差值和结论边界见下列解读。没有运行随机重连消融。',
            f'{link(root/"output/summary/边效应消融结果解读.md")}；{link(ablation)}。']
    signature=root/'output/summary/signature_ablation_metrics.tsv'
    if signature.exists():
        lines+=['','## signature选基因的随机对照','',
            '保留STATE全局边，以节点数、边密度和度数分布相近的随机gene signature替代IG选基因；只新增一个源模型及三项迁移。',
            f'{link(root/"output/summary/signature归因消融结果解读.md")}；{link(signature)}。']
    (root/'结果解读.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print('Saved within/transfer tables and interpretation.',flush=True)


if __name__=='__main__':main()
