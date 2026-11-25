"""Write figure-specific interpretation from saved attribution statistics."""
import argparse
from pathlib import Path
import sys
import pandas as pd

CODE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE))
from benchmark_paths import BENCH_ROOT


def link(path, title=None):
    path=Path(path).resolve()
    return "[" + (title or path.name) + "](" + path.as_posix() + ")"


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--open_path",type=Path,default=BENCH_ROOT/'output/attribution')
    a=parser.parse_args();root=a.open_path.resolve();summary=root/'summary';fig=root/'figures'
    quality=pd.read_csv(summary/'convergence_summary.tsv',sep='\t')
    validated=pd.read_csv(summary/'validation.tsv',sep='\t')
    rho=pd.read_csv(summary/'spearman_gene_mean.tsv',sep='\t')
    primary=rho[rho.baseline.eq('sensitive') & rho.background.eq('presence_10pct') & rho.metric.eq('mean_relative_absolute_ig') & rho.scope.isin(['all','crizotinib','alectinib'])]
    primary=primary[~(primary.dataset.eq('GSE223779_TS485T') & primary.scope.eq('all'))]
    cells=pd.read_csv(summary/'selected_cells.tsv',sep='\t',keep_default_na=False)
    consensus=pd.read_csv(summary/'directional_gene_consensus.tsv',sep='\t')
    lines=['# 三图模型归因与相关性解读','',
      '当前结果支持：三种完整图分类器对重要基因的排序有中等至较高一致性，但贡献方向与具体候选仍随模型和生物学背景变化。相关性不能证明耐药机制或因果驱动。','',
      f'共4个分析队列、7个测试／迁移范围，{cells.cell_id.nunique()}个不同细胞；三模型合计{int(validated.n_cells.sum())}份模型—细胞结果。两种基线共{int(quality.n_cells.sum())}次IG，{int(quality.n_converged.sum())}次达到误差阈值0.005；最大误差{quality.max_relative_error.max():.6f}。与既有预测的最大margin差为{validated.max_saved_margin_error.max():.3g}。','',
      '细胞由权威split JSON限定，再按生物学分组×真实标签以seed=42最多抽取32个；三个模型使用相同清单，不按预测正确与否挑选。这是均衡解释样本，不代表完整队列的细胞比例。PC9为DTEP状态，ReSisTrace为作者推断的治疗前命运；TS485T为条件相关敏／耐标签，不能统称逐细胞实测药敏。','',
      '模型为固定K完整GATv2＋GINE：STATE-15B、ESM2-650M、Qwen仅定义不同图先验；节点输入均为16维可训练gene identity＋表达。没有使用消融模型，也未重新训练。IG只解释表达通道，节点身份、signature选择和边保持固定。','',
      '## '+link(fig/'01_gene_spearman_sensitive.pdf','平均基因归因相关性：训练集敏感参考'),' ',
      '左：每个细胞先将绝对IG归一到总和1000，再对同一基因跨所选细胞取均值，比较模型间排序。右：直接比较原始signed IG均值的排序。共同背景限定为至少出现在10%所选细胞中的基因；格内n为实际基因数。两类统计均先在每折内计算，不将不同折当作重复随机种子。','',
      '基因未进入某细胞的固定图时，其条件模型贡献记0；在整个解释样本从未出现的基因不纳入背景。不同模型节点集合经核对相同，因此主要比较学习到的条件归因，而非词表大小。常数向量相关性记NA、显示灰色，不填0。','',
      '| 数据／范围 | 基因数 | STATE / 650M | STATE / Qwen | 650M / Qwen |','|---|---:|---:|---:|---:|']
    for (ds,fold,scope),part in primary.groupby(['dataset','fold','scope'],sort=False):
        values={(r.model_a,r.model_b):r.rho for r in part.itertuples()}
        name=ds+' / '+(fold if scope=='all' else scope)
        lines.append('| '+name+' | '+str(int(part.n_genes.iloc[0]))+' | '+' | '.join(f'{values[pair]:.3f}' for pair in [('state_15b','esm2_650m'),('state_15b','qwen'),('esm2_650m','qwen')])+' |')
    lines += ['',f'源表：{link(summary/"spearman_gene_mean.tsv")}。`all_observed`保存不设10%出现率门槛的敏感性统计；label_0／label_1单独统计也在此表。',
      '',f'训练敏感参考是在源训练敏感细胞的选中节点出现中按gene取表达均值；未覆盖gene回退为0并标记。各模型／折平均回退节点比例最高为{quality.mean_reference_missing_fraction.max():.2%}。参考细胞ID及每gene覆盖数在各模型目录，未用外部目标拟合基线。','',
      '## '+link(fig/'01_gene_spearman_zero.pdf','平均基因归因相关性：零表达参考'),' ',
      '统计单位、共同基因背景与上一图相同，唯基线改为输入表达为0。模型内部仍执行源训练缩放，0表达不等于标准化后的0。结果回答相对零表达的模型贡献，不能与敏感参考结果合并为同一套方向解释。signed IG始终针对class-1减class-0 logit；正值偏向class 1，负值偏向class 0。','',
      '## '+link(fig/'02_cellwise_spearman.pdf','逐细胞归因相关性'),' ',
      '每个细胞在自己的signature gene上比较两模型signed／absolute IG的Spearman，格内显示细胞相关性的中位数[P25,P75]。每个细胞只与同一ID比较。这里的四分位范围是所选细胞分布，不是独立实验／患者置信区间；不能用细胞数扩大生物学重复数。源表：'+link(summary/'spearman_per_cell.tsv')+'。','',
      '## '+link(fig/'03_positive_gene_IG_sensitive.pdf','class-1细胞的正向平均IG候选'),' ',
      '使用敏感参考，只在真实class-1细胞中按原始signed IG均值取前8个正值gene，要求出现率至少20%。两测试折按细胞数加权汇总；某gene在整折未出现也按该折全部所选细胞计入分母。TS485T两个药物分开，未处理细胞是共享参考。横轴不是表达量，也不是归一化到1000后的绝对归因。该图没有要求gene在class 0贡献更低。','',
      '更严格的候选另见'+link(summary/'directional_gene_consensus.tsv')+'：要求每个模型／折在class 1中的均值为正、比class 0更高且出现率≥20%。模型和折共同支持仍不是独立生物学验证。代表候选如下：','',
      '| 数据／条件 | 跨模型及折一致的候选（按平均label差值排序） |','|---|---|']
    for (ds,scope),part in consensus[consensus.consistent_across_models_and_folds].groupby(['dataset','scope'],sort=False):
        part=part.sort_values('mean_label_contrast',ascending=False).head(5)
        lines.append('| '+ds+' / '+scope+' | '+', '.join(part.gene_symbol)+' |')
    lines += ['', '这些候选用于后续功能／扰动验证；目前没有据此证明某gene驱动耐药，也没有证明解释优于DE/HVG。对于PC9应称治疗存活状态相关的模型候选。','']
    lines += ['## 边解释图的共同规则','',
      '每列一个模型，每行一个预定真实类别的细胞；标题注明真实／预测类别，ALK注明实际处理。目标为模型预测类别。显示mask最高的15个无向边对，双向mask取均值；线宽为mask，节点颜色为模型、文字为gene。布局仅区分显示分量，不表示生物距离。',
      '', 'mask是固定图上的优化解释分数，原边权为embedding cosine；不是调控方向、结合强度或所有gene对的交互。本轮未做实际删边忠实度验证，代表细胞也不等于细胞群网络共识。','']
    for pdf in sorted(fig.glob('04_edge_subgraphs_*.pdf')):
        lines += ['## '+link(pdf,'代表细胞边子图：'+pdf.stem.replace('04_edge_subgraphs_','')),'',
          '本图来自标题对应队列／折，每真实类别1个确定性选定细胞；统计与显示按上述共同规则。底层gene ID、边、mask与prior weight保存在对应模型的`gnn_class_0/1.npz`。','']
    lines += ['## 复现与边界','',
      '完整命令、来源代码、采样、基线与环境见'+link(CODE/'explain/exp.md')+'；数值核对见'+link(summary/'validation.tsv')+'；收敛汇总见'+link(summary/'convergence_summary.tsv')+'。PDF由已有scNET环境绘制，Arial、可编辑字体、magenta–white–teal相关性色阶[-1,1]；没有新增依赖。',
      '', '高absolute IG一致性表明模型强调相近的输入信息；signed一致性较低表明方向解释更依赖模型与参考。三个模型共享节点选择、表达和架构，因此不是三份独立机制证据。下一步应优先在独立样本／重复与同药物背景验证候选，再开展扰动或删基因／边对照。']
    (root/'结果解读.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


if __name__=='__main__':main()
