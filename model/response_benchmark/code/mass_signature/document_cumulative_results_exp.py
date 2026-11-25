"""Write figure-specific interpretation and explicit experiment completion state."""
import json
from pathlib import Path
from project_paths import project_root
import numpy as np,pandas as pd
ROOT=project_root();FIG=ROOT/'work/cumulative_signature';DATA=ROOT/'gene_sim_pathway/data/cumulative_signature_graphs'
s=pd.read_csv(FIG/'source_data/graph_summary.tsv',sep='\t');status=pd.read_csv(FIG/'source_data/training_completion.tsv',sep='\t') if (FIG/'source_data/training_completion.tsv').exists() else pd.DataFrame();policies=json.loads((FIG/'source_data/selected_policies.json').read_text(encoding='utf-8'));cost=pd.read_csv(FIG/'source_data/gpu_cost.tsv',sep='\t');inputs=pd.read_csv(ROOT/'work/signature_coverage/source_data/inputs.tsv',sep='\t')
lines=['# 累计归因图构建与四图预测：结果解读','','## 当前完成范围','',f'图结构描述覆盖{s[s.scope.eq("all_cells")].dataset.nunique()}/11个分析队列；GPU实测覆盖{len(cost[["dataset","fold"]].drop_duplicates())}/14项训练设计；已冻结{len(policies)}/14项源政策。训练核验完成{int(status.complete.sum()) if len(status) else 0}/56项，未完成不能填成零性能。','', '四种图统一16维可训练gene identity＋表达，比较图先验的连接关系与原始边权；该完整图主实验使用16维可训练gene identity；后续已追加共享无边控制，并在全部模型总图中纳入既有冻结蛋白gene向量、随机signature及SE＋MLP对照。四档阈值仍解释SCimilarity embedding，不直接解释耐药输出。','','## 已选政策','','|源任务|累计IG阈值|源训练细胞数|所选阈值实测峰值GiB|','|---|---:|---:|---:|']
for name,row in policies.items():lines.append(f'|{name}|{row["threshold"]:.0%}|{row["n_calibration_cells"]}|{row["observed_selected_peak_allocated_GiB"]:.2f}|')
checks=FIG/'source_data/completed_experiment_validation.tsv';targets=FIG/'source_data/frozen_transfer_validation.tsv'
if checks.exists() and targets.exists():
 source_checks=pd.read_csv(checks,sep='\t');target_checks=pd.read_csv(targets,sep='\t')
 lines+=['',f'独立核查通过{int(source_checks.status.eq("pass").sum())}/56个源检查点和{int(target_checks.status.eq("pass").sum())}/20个冻结目标配置：包括原始细胞/分组、训练集表达缩放、类别权重、源阈值继承及目标指标回算。FA34目标另抽取真实细胞进行CPU检查点预测重载。核查表为completed_experiment_validation.tsv和frozen_transfer_validation.tsv。']
seen_rationales=set()
for name,row in policies.items():
 if row['rationale'] not in seen_rationales:
  lines+=['',f'**{name.split("__")[0]}**：{row["rationale"]}']
  seen_rationales.add(row['rationale'])
lines+=['','政策依据原训练细胞的图结构及GPU成本审阅，四图共享；不以测试表现挑选。描述性全细胞统计不代替训练侧校准。外部队列须继承源模型阈值，临床集成的每个源折分别继承其政策。','','## 完整counts依赖','','|队列|原counts目前可读|新增节点表达已核验|','|---|---|---|']
for r in inputs.itertuples():
 m=json.loads(Path(r.manifest).read_text(encoding='utf-8'))['input'];ready=(DATA/'candidates'/r.dataset/'expression_validation.json').exists();lines.append(f'|{r.dataset}|{"是" if Path(m["h5ad"]).exists() else "否"}|{"是" if ready else "待准备"}|')
lines+=['','完整counts已从原路径读取，D盘IG只用于节点选择和图拓扑，不代替新增节点表达。未以零值、IG分数或随机值伪造分类器表达输入。原counts和旧模型结果均未修改。','','## 图结构统计口径','','统计单位是每个细胞的诱导子图，四图共享同一细胞和基因集合。阈值分母为原始全部模型基因归因和；不可达细胞保留全部正归因候选并标记未达标。不得在过滤后重新归一化。无向边数=CSR非零数/2；密度=2E/[N(N−1)]；孤立比例=度0节点/N；最大连通分量=最大组件节点/N。保留孤立节点，PPI边沿用≥0.5，蛋白图沿用现有KNN。','','全细胞描述去重后为111,725个细胞，交叉验证折不重复计数；训练校准表另外按14项源训练集计算。曲线点为细胞中位数，阴影为P25–P75，不是生物学重复的置信区间，不进行细胞伪重复显著性检验。']
for name,title in [('01_edge_counts.pdf','实际无向边数'),('02_isolation.pdf','孤立节点比例'),('03_largest_component.pdf','最大连通分量比例')]:
 lines+=['',f'## [{title}]({(FIG/"figures"/name).as_posix()})','','每队列一个面板，横轴为目标累计IG质量，四种颜色分别对应ESM2-15B、PPI、Qwen和ESM2-650M图；点/阴影为全细胞中位数/IQR。源表为graph_summary.tsv的all_cells行。边数和连通性随新增节点可能改变；更稠密或更连通不等于分类更好或生物学更正确。PPI更稀疏可能反映证据网络覆盖差异，不能据此判定其质量低。']
lines+=['',f'## [GPU训练步成本]({(FIG/"figures/04_measured_gpu_cost.pdf").as_posix()})','','只使用原训练细胞，按节点数分位选32个代表细胞，另取边数最大的32个训练细胞构成尾部批次。使用相同GATv2/GINE架构的随机初始化原型和真实表达，float32、batch32；交替0/1标签和无类别权重CE仅用于资源测试，不用于预测评估或效果选参，包含前向、反向、裁剪和AdamW状态；预热一步后实测两步，同步CUDA计时。图展示两个批次中较大的peak allocated，完整reserved、耗时和节点/有向边规模见gpu_cost.tsv。GPU为RTX3080 Laptop 16GiB。不是整个训练的最大显存保证，也不是只统计张量文件大小。完成后另存训练及推理全过程峰值，字段training_and_inference_peak_allocated_GiB。','','## 当前模型结果','','training_split_metrics.tsv逐任务保存train/validation/test，只有通过checkpoint重载与指标回算才算完成。GSE140440和FA34源队列随机细胞划分属于探索性队列内结果；患者/样本/重复外推及冻结迁移才检验未见生物学单位。未完成的组不能用于总体优劣判断。','','正式比较继续采用11个范围×4图：PC9样本集合等权、EOC患者OOF、三个ReSis重复等权，以及DU145、PC3、FCIBC02、TS485T同药/跨药和外部EPI患者。GSE202234为身份/处理混杂诊断。临床先平均各源折细胞概率，再平均患者内细胞概率；unknown不参与分类指标。','','## 结果边界','','当前可以检验固定K是否保留不均等的归因质量，以及更高覆盖的结构/资源代价。最终预测是否提升必须看同划分、同特征、同图的配对结果；单次seed42不能支持稳定最优或显著性结论。完整图主实验本身不能分离节点组成、图规模、拓扑与边权各自贡献；追加去边控制可比较整体消息传递作用，但未独立拆分拓扑与边权，也不构成因果机制验证。节点数量可能与测序深度、样本或细胞状态混杂，均值池化也不完全消除这种风险。','','入口和CMD命令见[exp.md]('+(FIG/'exp.md').as_posix()+')。图结构输入和逐细胞统计位于source_data/profiles，政策来源为selected_policies.json，训练状态为training_completion.tsv。']
metrics_path=FIG/'source_data/primary_metrics.tsv'
contrast_path=FIG/'source_data/matched_fixedK_differences.tsv'
if metrics_path.exists() and metrics_path.stat().st_size>5:
 result=pd.read_csv(metrics_path,sep='\t')
 if len(result):
  contrast=pd.read_csv(contrast_path,sep='\t');delta=contrast[contrast.metric.eq('auroc')]
  relations=['state_15b','ppi','qwen','esm2_650m'];names=['ESM2-15B图','PPI图','Qwen图','ESM2-650M图']
  lines+=['','## 当前结果最支持什么','']
  scopes=list(result.groupby(['dataset','scope'],sort=False))
  below=[ds+' / '+scope for (ds,scope),g in scopes if g.model.nunique()==4 and g.auroc.lt(.5).all()]
  if below:lines+=['四种图均低于0.5 AUROC的范围：'+ '；'.join(below)+'。这表明当前模型在这些目标上没有获得按原正类方向的有效排序；不能翻转分数后报告为成功迁移。']
  ts=result[result.dataset.eq('GSE223779_TS485T')]
  if len(ts)==8:lines+=['',f'FA34→TS485T两种药物范围的四图AUROC为{ts.auroc.min():.3f}–{ts.auroc.max():.3f}，balanced accuracy为{ts.balanced_accuracy.min():.3f}–{ts.balanced_accuracy.max():.3f}。该信号局限于特定类器官/TKI背景，不能推广为跨疾病或临床通用耐药预测。']
  clinical_oof=result[result.dataset.eq('GSE165897_pre_EOC')]
  if len(clinical_oof)==4 and clinical_oof.auroc.nunique()==1:
   folder=ROOT/'model/response_benchmark/output/cumulative_signature';patients=pd.read_csv(folder/'state_15b/grouped/GSE165897_pre_EOC/summary/oof_patient_predictions.tsv',sep='\t');pairs=int(patients.y.eq(0).sum()*patients.y.eq(1).sum());auc=float(clinical_oof.auroc.iloc[0])
   lines+=['',f'EOC患者OOF四图AUROC均为{auc:.3f}：{len(patients)}位患者形成{pairs}个敏感–耐药配对，当前各模型恰有约{auc*pairs:g}个正确排序当量。相同AUROC不代表患者概率或分类完全相同；配对数也不是独立生物学样本量。应同时查看患者概率和balanced accuracy，不能将一致低分当成独立生物学验证。']
  lines+=['','### 按研究背景理解','']
  for ds,label,meaning in [
   ('GSE149383_PC9_DTEP_sample_grouped','PC9样本集合外推','区分未处理与DTEP处理状态，不等同于稳定获得性耐药或患者药敏'),
   ('GSE223003_carboplatin_pre','Carboplatin预处理命运','检验处理前状态能否预测后续存活命运；按原重复等权汇总'),
   ('GSE223003_nk_pre','NK预处理命运','属于免疫细胞杀伤背景，不是化疗药物耐药；按原重复等权汇总'),
   ('GSE223003_olaparib_pre','Olaparib预处理命运','检验PARP抑制剂背景的后续存活命运；不能与NK或其他药物视为同一机制'),
   ('GSE154600_EPI_external','卵巢癌外部患者','仅4位有标签患者，确切用药一致性和上皮细胞恶性身份仍需独立核实')]:
   subset=result[result.dataset.eq(ds)]
   if len(subset)==4:
    lines.append(f'- **{label}**：四图AUROC {subset.auroc.min():.3f}–{subset.auroc.max():.3f}，balanced accuracy {subset.balanced_accuracy.min():.3f}–{subset.balanced_accuracy.max():.3f}。{meaning}。')
  peaks=cost.groupby('threshold').peak_allocated_GiB.max()
  lines+=['',f'资源实测的跨任务最大allocated峰值：50%为{peaks.loc[.5]:.2f} GiB，60%为{peaks.loc[.6]:.2f} GiB，70%为{peaks.loc[.7]:.2f} GiB，80%为{peaks.loc[.8]:.2f} GiB。四档均未因显存不足失败；选择60%/70%/80%是结构与成本折中，不是因为80%无法运行，也没有证明所选阈值的预测性能最优。','', '应同时看同图旧固定K差值和绝对表现：相对提高但仍低于0.5、或高AUROC却仅预测单一类别，都不能称为可靠分类。完整分组结果尚未完成时，不提前宣布某种图或累计归因方法整体获胜。','','## 当前已核验的正式评价结果','',f'当前有{len(result)}/44个正式范围×图配置结果。下表是新累计阈值方案AUROC（相对同图旧固定K的差值）；缺失表示尚未完成，不补零。指标保留方向，未按目标标签翻转分数。','','|队列／范围|'+'|'.join(names)+'|','|---|'+'---:|'*4]
  for (ds,scope),group in result.groupby(['dataset','scope'],sort=False):
   row=[ds+' / '+scope]
   for rel in relations:
    g=group[group.model.eq(rel)];d=delta[delta.dataset.eq(ds)&delta.scope.eq(scope)&delta.relation.eq(rel)]
    row.append('NA' if not len(g) else f'{g.iloc[0].auroc:.3f} ({d.iloc[0].difference:+.3f})')
   lines.append('| '+' | '.join(row)+' |')
  lines+=['','## 固定0.5阈值的分类表现','','AUROC衡量排序，不能替代实际检出。下表为相同范围的新方案balanced accuracy；接近0.5可能是无区分度或几乎只预测单一类别，结合混淆矩阵、敏感/耐药AP及阳性基率解读。','','|队列／范围|'+'|'.join(names)+'|','|---|'+'---:|'*4]
  for (ds,scope),group in result.groupby(['dataset','scope'],sort=False):
   row=[ds+' / '+scope]
   for rel in relations:
    g=group[group.model.eq(rel)];row.append('NA' if not len(g) else f'{g.iloc[0].balanced_accuracy:.3f}')
   lines.append('| '+' | '.join(row)+' |')
  lines+=['','## 如何判断收益','','|图|已比较范围|AUROC上升／下降／近似持平|','|---|---:|---|']
  for rel,name in zip(relations,names):
   d=delta[delta.relation.eq(rel)].difference.to_numpy();tie=np.isclose(d,0,rtol=0,atol=1e-8);lines.append(f'|{name}|{len(d)}|{int(((d>0)&~tie).sum())}／{int(((d<0)&~tie).sum())}／{int(tie.sum())}|')
  lines+=['','这是描述性范围计数，不是独立重复或显著性检验；DU145/PC3、TS485T同药/跨药等范围存在背景或细胞重叠。比较计数的数值近似持平容差为1e-8，不是生物学等效界值。不能据更多上升范围就宣布普遍最优。','','临床OOF只有11位患者，外部临床为4位有标签患者，不能将细胞数当作有效患者样本量。PC9为DTEP处理状态，NK为预处理存活命运；跨癌种/跨药物迁移为压力测试，不是等价的同机制耐药验证。若只在队列内高分而外部仍接近随机，应解释为背景依赖。','','累计归因规则同时改变节点组成、节点数量、边数和训练节点表达缩放分布。本轮保持四图之间的节点政策一致，但与旧固定K相比不是等节点预算对照；无消融时不能将全部性能变化归因于真实基因交互或覆盖机制本身。']
  for file,title,method in [('05_paired_AUROC_changes.pdf','同图固定K与累计归因的AUROC差值','每格=新累计方案AUROC−同图旧固定K AUROC，使用同一评价范围及原划分。紫红为下降、蓝绿为上升，缺失显示NA。颜色不代表显著性。'),('06_absolute_AUROC_comparison.pdf','固定K与累计归因的绝对AUROC','两个面板使用同一0–1色阶，分别显示旧固定K和新累计方案。与差值图一起判断：相对提高仍可能低于有效区分水平。')]:
   lines+=['',f'## [{title}]({(FIG/"figures"/file).as_posix()})','',method]
split_path=FIG/'source_data/training_split_metrics.tsv'
if split_path.exists() and split_path.stat().st_size>5:
 split=pd.read_csv(split_path,sep='\t');simple=split[split.dataset.isin(['GSE140440','GSE223779'])&split.split.eq('test')]
 if len(simple):
  lines+=['','## 探索性随机细胞划分结果','','|队列|图|测试细胞|AUROC|Balanced accuracy|','|---|---|---:|---:|---:|']
  for r in simple.itertuples():lines.append(f'|{r.dataset}|{r.relation}|{r.n_cells}|{r.auroc:.4f}|{r.balanced_accuracy:.3f}|')
  lines+=['','这些细胞与训练细胞来自相同队列背景，接近满分不证明患者、克隆或药物间泛化。所有显示值均经过小数位舍入，精确指标见源表。']
design_table=FIG/'source_data/training_design_comparison.tsv'
if design_table.exists():
 designs=pd.read_csv(design_table,sep='\t')
 fold_rows=[]
 for ds,fold in designs[['dataset','fold']].drop_duplicates().itertuples(index=False,name=None):
  cohort=ROOT/'model/response_benchmark'
  if fold!='single':cohort=cohort/'data/grouped'/ds/fold
  folder=cohort/'data/cohorts'/ds
  info=json.loads((folder/'data_sources.json').read_text(encoding='utf-8'));cells=pd.read_csv(folder/'cells.tsv',sep='\t',keep_default_na=False);group=info.get('group_key')
  row=dict(dataset=ds,fold=fold,group_key=group or 'random_cells',cells_path=str(folder/'cells.tsv'))
  for part in ['train','validation','test']:
   selected=cells[cells.split.eq(part)];row['n_'+part+'_cells']=len(selected);row[part+'_groups']='; '.join(sorted(map(str,selected[group].unique()))) if group else 'random cell split'
  row['inner_validation_group_disjoint']=not bool(set(cells.loc[cells.split.eq('train'),group])&set(cells.loc[cells.split.eq('validation'),group])) if group else False
  fold_rows.append(row)
 pd.DataFrame(fold_rows).to_csv(FIG/'source_data/fold_definitions.tsv',sep='\t',index=False)
 lines+=['','## 14项训练与11行正式评价如何对应','','14项是每种图的训练设计数，不是14个独立数据集：GSE140440随机细胞划分1项、FA34随机细胞划分1项、PC9两折、EOC四折、Carboplatin/NK/Olaparib各两折，共14项；四图共56次训练。','','图05/06保留正式汇总口径：PC9两折平均1行、EOC折外患者1行、三类ReSis各1行，共5行；另外6行为冻结迁移（DU145、PC3、FCIBC02、TS485T同药/跨药、外部EOC患者）。这些迁移复用源模型，不是新增训练。图05中的DU145/PC3不是GSE140440本地训练结果。','',f'## [全部14项训练设计：测试表现]({(FIG/"figures/07_all_14_training_designs.pdf").as_posix()})','','14行逐项展示56个模型的测试AUROC和固定0.5阈值balanced accuracy。每格以该设计的测试细胞为统计单位；不跨折平均。灰线区分前两项探索性随机细胞划分与后12项生物学分组外推。EOC这里是单折测试细胞指标，并非单折患者AUROC；临床结论仍采用图05/06中合并折外预测后按患者计算的结果。患者/样本/重复数远少于细胞数，不能将细胞当独立生物学重复。','',f'## [全部14项训练设计：固定K配对比较]({(FIG/"figures/08_all_14_designs_fixedK_comparison.pdf").as_posix()})','',f'三面板分别为旧固定K、新累计归因和新减旧AUROC。新结果56格齐全，旧结果可配对{int(designs.fixed_k_auroc.notna().sum())}格。{("GSE140440的PPI/Qwen/650M旧固定K参考已按原K1536、原划分和原优化参数补跑，并通过检查点重载核验；全部56格均可配对。" if designs.fixed_k_auroc.notna().sum()==56 else "仍有旧固定K结果缺失，相应旧值和差值标NA，不补零。")}所有可配对格已核对完整细胞ID、标签与split一致，并从保存的预测独立回算AUROC和balanced accuracy。源表为training_design_comparison.tsv。','','该图用于查明具体折和划分的差异，不能把14项当成等价的14次独立验证；特别是随机细胞划分高分不能替代患者外推，跨折细胞指标也不能替代患者级OOF评价。']
if design_table.exists():
 lines+=['','## fold_1是什么意思','','fold_1是第1个预先固定的训练/验证/测试划分，不是第1次随机种子重复。fold_2等同理。每折重新训练一个模型；四种图以及新旧选基因方案沿用同一折的细胞ID和标签。所有训练仍是seed42。single表示只有一次划分。','','|数据集|折|测试中留出的生物学单位|测试细胞数|','|---|---|---|---:|']
 for row in fold_rows:lines.append(f'|{row["dataset"]}|{row["fold"]}|{row["test_groups"]}|{row["n_test_cells"]}|')
 lines+=['','PC9的两折交换两组样本作为外部测试；EOC四折分别留出不同患者；ReSis两折交换生物学重复1/2作为测试。训练集、验证集和测试集的完整分组及细胞数见fold_definitions.tsv，各细胞归属以其cells_path指向的原cells.tsv为准。','','外部测试分组不重叠不代表内部验证也全部按组划分：PC9和ReSis内部验证仍是源重复/样本内的随机细胞；EOC内部验证按患者分组。GSE140440和FA34的single为随机细胞划分，不能解释为患者外推。不同折难度和组成不同，应看逐折结果与按生物学单位的汇总，不能仅挑最好的一折。']
if (FIG/'边消融结果解读.md').exists():
 lines+=['','## 补充：边消融与全部模型总览','',f'[旧固定K与累计归因边消融]({(FIG/"边消融结果解读.md").as_posix()})；[全部13配置总图与统计方法]({(FIG/"全部模型对比解读.md").as_posix()})。保留原完整图结果，本轮另补共享无边控制。']
(FIG/'结果解读.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
print('Interpretation synchronized with current verified completion state')



