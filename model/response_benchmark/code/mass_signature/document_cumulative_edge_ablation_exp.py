"""Write figure-specific interpretation of matched fixed-K/cumulative edge controls."""
from pathlib import Path
import pandas as pd,numpy as np
from project_paths import project_root
ROOT=project_root();FIG=ROOT/'work/cumulative_signature';TABLES=FIG/'source_data/edge_ablation'
control=pd.read_csv(TABLES/'fixedK_vs_cumulative_no_edges.tsv',sep='\t');effects=pd.read_csv(TABLES/'formal_edge_effects.tsv',sep='\t');source=pd.read_csv(TABLES/'source_validation.tsv',sep='\t');target=pd.read_csv(TABLES/'transfer_validation.tsv',sep='\t')
assert len(source)==14 and len(target)==5 and len(control)==11 and len(effects)==308
names={'state_15b':'ESM2-15B','ppi':'PPI','qwen':'Qwen','esm2_650m':'ESM2-650M'}
lines=['# 固定K与累计归因：去边消融结果解读','','## 实验完成范围','','累计归因方案14项设计各训练一个共享无边模型；旧固定K的13个既有无边参考保留，另补齐GSE140440的K1536无边参考。源模型及新旧原始划分均已核验。FA34冻结迁移到4个目标队列，EOC四折模型集成到外部EPI患者；共11个正式评价范围，与完整图主表一致。GSE202234及单类别SUM149PR保留作额外诊断，不混入正式11范围。','','四图共享每个细胞的节点、表达和16维gene identity；全部边删除后输入相同，故无边对照只训练一次。共享对照在四图比较中重复引用，不是四次独立训练。没有加入随机边或其他新消融。','','## 固定什么、改变什么','','同一节点政策内，完整图与无边使用完全相同的节点和表达、训练集缩放、基因词表、参数数量、模型架构、原train/validation/test、seed42、优化器及训练预算。只删除输入边（edge_index=2×0，edge_attr=0×1）；每个训练和推理batch检查零边。保留GAT偏置/残差、GINE自身变换、可训练gene identity、池化与分类头。因此不是简单表达平均，也不能将无边理解为完全没有基因间联合判别。','','旧固定K按数据集使用原设置：GSE140440为1536，FA34为1024，PC9/EOC为768，三类ReSis为1024；上限K仍受该细胞有效正归因候选数限制。新方案沿用已经冻结的累计阈值，分母为原始全部模型基因归因和，无固定K上限；迁移继承源阈值。新旧节点组成/数量及训练节点缩放分布可能不同，故新旧无边差异不是边的作用。','','仅源训练细胞拟合缩放，源验证集选checkpoint；目标不训练、不校准、不翻转方向。概率阈值0.5。全部是seed42单次拟合，差值不是显著性或独立重复结论。','','## 直接比较旧、新无边模型','','统计单位沿用正式评价：PC9和ReSis为留出样本集合/重复指标等权均值；EOC为11名患者的合并折外预测；临床外部为4名有标签患者；其余为指定细胞系或药物范围的全部合格有标签目标细胞。unknown排除指标，不补0。AP使用同一正类方向，并结合阳性基率，不直接跨队列比较AP高低。','','|队列／范围|固定K无边AUROC|累计无边AUROC|差值|固定K／累计BA|','|---|---:|---:|---:|---:|']
for r in control.itertuples():lines.append(f'|{r.dataset} / {r.scope}|{r.fixed_k_auroc:.3f}|{r.cumulative_auroc:.3f}|{r.cumulative_auroc-r.fixed_k_auroc:+.3f}|{r.fixed_k_balanced_accuracy:.3f} / {r.cumulative_balanced_accuracy:.3f}|')
lines+=['','## 完整图是否提供额外收益','','每格为旧固定K的“完整图−共享无边”AUROC差值 → 新累计方案的相同差值。正值表示完整图较好，负值表示去边较好。先看这一配对差值，再看绝对AUROC和BA；相对改善而绝对表现低于0.5不能称为有效迁移。','','|队列／范围|'+'|'.join(names.values())+'|','|---|'+'---:|'*4]
auc=effects[effects.metric.eq('auroc')]
for r in control.itertuples():
 row=[r.dataset+' / '+r.scope]
 for rel in names:
  e=auc[auc.dataset.eq(r.dataset)&auc.scope.eq(r.scope)&auc.relation.eq(rel)].iloc[0];row.append(f'{e.fixed_k_edge_gain:+.3f} → {e.cumulative_edge_gain:+.3f}')
 lines.append('| '+' | '.join(row)+' |')
lines+=['','## 当前结果的整体含义','']
change=control.cumulative_auroc-control.fixed_k_auroc
lines.append(f'新累计无边相对旧固定K无边：{int((change>1e-8).sum())}/11个范围AUROC提高，{int((change < -1e-8).sum())}个降低，{int((np.abs(change)<=1e-8).sum())}个近似持平。这是描述性计数，范围并非独立重复，不能按多数投票宣布最优。')
for rel,name in names.items():
 e=auc[auc.relation.eq(rel)];positive=int((e.cumulative_edge_gain>1e-8).sum());negative=int((e.cumulative_edge_gain < -1e-8).sum())
 lines.append(f'- {name}：累计方案完整图在{positive}/11个范围AUROC高于共享无边，{negative}个低于；边优势相对旧方案在{int((e.edge_gain_change>1e-8).sum())}个范围增加、{int((e.edge_gain_change < -1e-8).sum())}个减少。')
strong=auc[(auc.cumulative_edge_gain>0)&(auc.cumulative_full>=.7)]
if len(strong):
 lines+=['','兼有完整图AUROC≥0.7且优于无边的范围（描述性筛选，不是预注册显著性阈值）：']
 for r in strong.itertuples():lines.append(f'- {r.dataset} / {r.scope} / {names[r.relation]}：完整图{r.cumulative_full:.3f}，无边{r.cumulative_no_edges:.3f}，差值{r.cumulative_edge_gain:+.3f}。')
lines+=['','差中之差=(累计完整−累计无边)−(固定K完整−固定K无边)，用于描述选节点方式变化后边优势如何变化。它不是因果difference-in-differences研究：节点预算、组成和源缩放同时改变，只有一个训练种子，也没有随机连接对照。不能据正值证明特定真实网络优于随机网络，或将边解释为已验证的因果基因交互。','','PC9代表DTEP处理状态，ReSis代表特定处理下存活命运，NK并非化疗；FA34→TS485T是有限类器官/TKI背景；跨癌种/跨药物为压力测试。EOC临床样本量小，不能用细胞数量代替患者数。', '',f'## [无边模型：固定K与累计归因]({(FIG/"figures/09_no_edges_fixedK_vs_cumulative.pdf").as_posix()})','','两页分别为AUROC和balanced accuracy，11行正式评价，三面板为旧无边、新无边与新−旧。绝对色阶0–1，差值紫红为下降、蓝绿为上升。每格是该范围的指标，不是逐细胞归因；零差值与缺失不同，本次11个范围齐全。源表fixedK_vs_cumulative_no_edges.tsv。','',f'## [边优势：旧固定K与累计归因]({(FIG/"figures/10_edge_advantage_fixedK_vs_cumulative.pdf").as_posix()})','','两页分别为AUROC和BA。每列一种图，11行正式评价；左为旧完整−旧无边，中为新完整−新无边，右为这两个差值之差。色阶关于0对称，正值/负值不代表显著性。四列无边对照共享，不能当独立重复。源表formal_edge_effects.tsv，另含两类AP、MCC、macro-F1、Brier等7项指标；Brier越低越好，其差值不能沿用AUROC的正值有益解释。','',f'## [14项训练逐项无边对比]({(FIG/"figures/11_all_14_designs_no_edges.pdf").as_posix()})','','第一页为14项测试细胞的旧/新无边AUROC与BA；第二页为各图旧/新边优势及变化。前两行为探索性随机细胞划分，用横线与分组外推区分；其余为原固定折，不选最好一折。EOC逐折是测试细胞指标，临床主结论仍以合并折外患者评价为准。完整折定义和实际留出单位见原fold_definitions.tsv。源表training_design_edge_effects.tsv和split_metrics.tsv。','','## 核验与复现','','source_validation.tsv核查14个新控制及14个旧控制：原划分、训练集缩放、类别权重、同特征同参数、全部split指标回算及新检查点CPU抽样重载；每个新模型还在训练结束后重载全部测试细胞。transfer_validation.tsv核查4个FA34目标和1个临床外部配置：原目标细胞/标签、源阈值、指标回算和CPU重载；临床另核查4折预测均值、患者均值。graph_input_validation.tsv核查12个队列/阈值组合的真实节点特征与四图一致、零边。', '',f'[CMD命令与依赖环境]({(FIG/"edge_ablation_exp.md").as_posix()})。本次未重新计算上游IG、未安装依赖，检查点位于model/response_benchmark/output/cumulative_signature/no_edges。']
(FIG/'边消融结果解读.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
print('Edge ablation interpretation written')
