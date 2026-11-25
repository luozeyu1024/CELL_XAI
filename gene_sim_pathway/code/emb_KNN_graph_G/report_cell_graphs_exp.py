"""Regenerate a concise cell-graph data report from saved tables, without recomputation."""
import argparse
import json
from pathlib import Path
import pandas as pd


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--open_path',required=True)
    args=p.parse_args();root=Path(args.open_path).resolve()
    summary=json.loads((root/'summary.json').read_text(encoding='utf-8'))
    table=[]
    for dataset in summary['datasets']:
        name=dataset['dataset'];folder=root/name
        qc=pd.read_csv(folder/'signature_qc.tsv',sep='\t')
        size=sum(f.stat().st_size for f in folder.iterdir() if f.is_file())
        row={'dataset':name,'cells':len(qc),'eligible_genes':dataset['n_eligible_genes'],
             'median_nodes':qc.n_nodes.median(), 'min_nodes':int(qc.n_nodes.min()),
             'max_nodes':int(qc.n_nodes.max()), 'top_k':dataset['top_k'], 'cells_below_requested_k':int(qc.n_nodes.lt(dataset['top_k']).sum()),
             'median_eligible_importance_fraction':qc.eligible_importance_fraction.median(),
             'median_selected_importance_fraction':qc.selected_importance_fraction.median(),
             'size_bytes':size}
        for relation in ['esm2_650m','qwen','state_15b','ppi']:
            for metric in ['undirected_edges','isolated_fraction','largest_component_fraction']:
                row[f'median_{relation}_{metric}']=qc[f'{relation}_{metric}'].median()
        table.append(row)
    pd.DataFrame(table).to_csv(root/'dataset_summary.tsv',sep='\t',index=False)
    global_params=json.loads((root/'global_graphs/parameters.json').read_text(encoding='utf-8'))
    lines=['# SCimilarity–IG dataset-specific topK 逐细胞子图','',
           '六个GSE使用共同词表，每细胞按parameters.json的top_k选正归因基因；四种图共用节点和特征。已生成可加载的数据包及逐细胞图统计，未训练分类模型。','',
           '| GSE | 细胞数 | 合格基因数 | 选中归因覆盖中位数 | 数据包 MiB |',
           '|---|---:|---:|---:|---:|']
    for row in table:
        lines.append(f"| {row['dataset']} | {row['cells']:,} | {row['eligible_genes']:,} | {row['median_selected_importance_fraction']:.1%} | {row['size_bytes']/2**20:.1f} |")
    lines += ['', '## 四个共享基因图','', '| 关系 | 可识别基因 | 无向连接 |','|---|---:|---:|']
    for relation,record in global_params['relations'].items():
        lines.append(f"| {relation} | {record['known_genes']:,} | {record['undirected_edges']:,} |")
    lines += ['', '蛋白KNN按HGNC折叠为基因边，重复边取最大cosine；PPI保留conn≥0.5。图保留双向边记录，无向连接数为记录数的一半。未知映射及规范化后重复基因有独立记录。','',
              '## 文件与统计解释','',
              '- `gene_vocab.tsv`、`global_graphs/*.npz`：共享基因索引、四个CSR邻接矩阵；`coverage.npz`记录各图是否识别该节点。',
              '- `<GSE>/signatures.npz`：ptr、gene_id、features。features依次为log-normalized expression、IG/1000、排名百分位（0最重要）。不重复保存边和高维蛋白向量。',
              '- `<GSE>/cells.tsv`：细胞ID、原始行号、y及标签状态；`source_metadata.tsv`保留来源中的元数据。目前所有y=-1，尚未核实耐药/敏感标签。',
              '- `<GSE>/eligible_genes.tsv`：实测、SCimilarity可评分、四图共同识别、无HGNC映射碰撞的候选基因。只将正归因基因纳入各细胞signature；不足指定K按实际数量保留，不补零。节点数范围及不足指定K的细胞数见dataset_summary.tsv。',
              '- `<GSE>/signature_qc.tsv`：每细胞四图的边数、孤立率、最大连通分量比例及归因覆盖；数据集汇总见`dataset_summary.tsv`。',
              '- 归因覆盖：选中基因归因和 / 该细胞原始全部模型基因归因和；未在筛选后重新归一。合格基因覆盖另报。总分为0时比例缺失；已知节点无边计0；空图比例缺失。',
              '- 图结构指标均以当前细胞选中的节点为背景；孤立率=度数为0的节点数/节点数，最大连通分量比例=最大分量节点数/节点数。各数据集的中位数是在细胞层面取中位数，不把细胞视为独立生物重复。',
              '- 全局图的一条连接只表示先验关系；诱导子图不是实验测量的细胞内物理互作。上游IG解释SCimilarity表示，不能直接赋予耐药/敏感方向。','',
              '## 加载与复现','',
              '[完整命令与Python加载示例]('+str((root.parents[1]/'code/emb_KNN_graph_G/exp.md').as_posix())+')。加载器按signature切片全局CSR，返回局部节点编号的edge_index、edge_attr、x、gene_id和cell_id，保留孤立节点；不重算KNN、不补外部节点。','',
              'GSE112274、GSE117872没有可用归因，保留dataset_status.json并跳过。本次使用建议K；逐细胞累计质量自适应K和native词表未运行。']
    validation=root/'validation.json'
    if validation.exists():
        v=json.loads(validation.read_text(encoding='utf-8'))
        lines += ['',f"核验：{sum(r['cells_checked'] for r in v['datasets']):,}个细胞、{sum(r['graph_views_checked'] for r in v['datasets']):,}个图视图通过检查；另对每GSE最多16个细胞独立重算表达归一化，检查加载接口返回的图。详见validation.json。"]
    (root/'README.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


if __name__=='__main__': main()
