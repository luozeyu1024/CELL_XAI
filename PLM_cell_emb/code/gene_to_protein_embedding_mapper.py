#!/usr/bin/env python3
"""
基因ID到蛋白质embedding匹配脚本
功能：
1. 将RNA-seq基因ID转换为UniProt ID
2. 匹配蛋白质embedding
3. 多个蛋白质embedding取平均
4. 统计未映射情况
"""

import argparse
import json
import os
import random
import pandas as pd
import numpy as np
import scanpy as sc
from tqdm import tqdm
import glob


def load_hgnc_data(hgnc_file):
    """加载HGNC数据并建立基因symbol到UniProt ID的映射"""
    print("正在加载HGNC数据...")
    
    # 读取HGNC数据
    hgnc_df = pd.read_csv(hgnc_file, sep='\t', low_memory=False)
    
    # 建立symbol到uniprot_ids的映射
    gene_to_uniprot = {}
    
    for _, row in tqdm(hgnc_df.iterrows(), total=len(hgnc_df), desc="处理HGNC映射"):
        symbol = row['symbol']
        uniprot_ids = row['uniprot_ids']
        
        if pd.notna(symbol) and pd.notna(uniprot_ids):
            # 处理多个UniProt ID（用|分隔）
            uniprot_list = [uid.strip() for uid in str(uniprot_ids).split('|') if uid.strip()]
            if uniprot_list:
                gene_to_uniprot[symbol] = uniprot_list
    
    print(f"成功建立 {len(gene_to_uniprot)} 个基因的UniProt映射")
    return gene_to_uniprot


def load_rna_seq_data(h5ad_file):
    """加载RNA-seq数据"""
    print("正在加载RNA-seq数据...")
    adata = sc.read_h5ad(h5ad_file)
    gene_names = adata.var.index.tolist()
    print(f"RNA-seq数据包含 {len(gene_names)} 个基因, {adata.n_obs} 个细胞")
    return adata, gene_names


def map_genes_to_proteins(gene_names, gene_to_uniprot, max_proteins=10):
    """将基因名称映射到蛋白质ID"""
    print("正在映射基因到蛋白质ID...")
    
    gene_protein_mapping = []
    unmapped_genes = []
    
    for gene in tqdm(gene_names, desc="基因映射"):
        mapping_entry = {"gene_id": gene}
        
        if gene in gene_to_uniprot:
            protein_ids = gene_to_uniprot[gene]
            
            # 如果蛋白质数量超过最大值，随机选择
            if len(protein_ids) > max_proteins:
                protein_ids = random.sample(protein_ids, max_proteins)
            
            # 添加蛋白质ID列
            for i, protein_id in enumerate(protein_ids):
                mapping_entry[f"protein_id_{i+1}"] = protein_id
        else:
            # 记录未映射的基因
            unmapped_genes.append(gene)
        
        gene_protein_mapping.append(mapping_entry)
    
    mapped_count = sum(1 for entry in gene_protein_mapping if len(entry) > 1)
    print(f"成功映射 {mapped_count} 个基因到蛋白质")
    print(f"未映射基因数量: {len(unmapped_genes)}")
    
    return gene_protein_mapping, unmapped_genes


def extract_uniprot_id(full_id):
    """从完整的UniProt ID中提取核心ID
    例如: sp|A0A023I7E1|ENG1_RHIMI -> A0A023I7E1
    或: A0A023I7E1 -> A0A023I7E1
    """
    if '|' in full_id:
        parts = full_id.split('|')
        if len(parts) >= 2:
            return parts[1]
    return full_id.strip()


def find_embedding_files(embedding_dir, id_column="ID"):
    """在指定目录中查找npy和xlsx文件"""
    npy_files = glob.glob(os.path.join(embedding_dir, "*.npy"))
    xlsx_files = glob.glob(os.path.join(embedding_dir, "*.xlsx"))
    
    if not npy_files:
        raise FileNotFoundError(f"在 {embedding_dir} 中未找到.npy文件")
    if not xlsx_files:
        raise FileNotFoundError(f"在 {embedding_dir} 中未找到.xlsx文件")
    
    selected_xlsx = None
    if len(xlsx_files) > 1:
        print(f"找到 {len(xlsx_files)} 个xlsx文件，正在筛选含有 '{id_column}' 列的文件...")
        for xlsx_file in xlsx_files:
            try:
                df = pd.read_excel(xlsx_file, nrows=0)
                if id_column in df.columns:
                    selected_xlsx = xlsx_file
                    print(f"选择文件: {xlsx_file}")
                    break
            except Exception as e:
                print(f"读取 {xlsx_file} 时出错: {e}")
                continue
        
        if selected_xlsx is None:
            print(f"警告：未找到含有 '{id_column}' 列的xlsx文件，使用第一个: {xlsx_files[0]}")
            selected_xlsx = xlsx_files[0]
    else:
        selected_xlsx = xlsx_files[0]
    
    return npy_files[0], selected_xlsx


def load_protein_embeddings(embedding_dir, id_column):
    """加载蛋白质embedding和对应的ID"""
    print("正在加载蛋白质embedding...")
    
    # 查找文件
    npy_file, xlsx_file = find_embedding_files(embedding_dir, id_column)
    
    print(f"使用embedding文件: {npy_file}")
    print(f"使用ID文件: {xlsx_file}")
    
    # 加载embedding
    embeddings = np.load(npy_file)
    print(f"Embedding形状: {embeddings.shape}")
    
    # 加载ID文件
    id_df = pd.read_excel(xlsx_file)
    if id_column not in id_df.columns:
        raise ValueError(f"ID列 '{id_column}' 不存在于 {xlsx_file} 中。可用列: {id_df.columns.tolist()}")
    
    # 提取UniProt ID
    protein_ids = [extract_uniprot_id(str(id_val)) for id_val in id_df[id_column]]
    
    print(f"加载了 {len(protein_ids)} 个蛋白质embedding")
    
    # 建立ID到embedding的映射
    protein_to_embedding = {}
    for i, protein_id in enumerate(protein_ids):
        protein_to_embedding[protein_id] = embeddings[i]
    
    return protein_to_embedding


def get_top_genes_per_cell(adata, top_k=None):
    """获取每个细胞的top k基因"""
    if top_k is None:
        return None
        
    print(f"计算每个细胞的top {top_k} 基因...")
    
    # 如果是稀疏矩阵，转换为密集矩阵
    if hasattr(adata.X, 'toarray'):
        expression_matrix = adata.X.toarray()
    else:
        expression_matrix = adata.X
    
    # 获取基因名称
    gene_names = adata.var.index.tolist()
    
    # 为每个细胞找到top k基因
    top_genes_per_cell = []
    
    for cell_idx in tqdm(range(expression_matrix.shape[0]), desc="处理细胞"):
        cell_expression = expression_matrix[cell_idx, :]
        # 获取top k基因的索引
        top_gene_indices = np.argsort(cell_expression)[-top_k:][::-1]  # 降序排列
        # 获取基因名称
        top_genes = [gene_names[idx] for idx in top_gene_indices]
        top_genes_per_cell.append(top_genes)
    
    return top_genes_per_cell


def filter_genes_by_top_selection(adata, gene_embeddings, top_genes_per_cell):
    """根据top基因选择过滤基因embedding"""
    if top_genes_per_cell is None:
        return gene_embeddings
    
    print("根据top基因选择过滤基因embedding...")
    
    # 收集所有被选中的基因
    selected_genes = set()
    for cell_top_genes in top_genes_per_cell:
        selected_genes.update(cell_top_genes)
    
    # 过滤基因embedding，只保留被选中的基因
    filtered_gene_embeddings = {
        gene: embedding for gene, embedding in gene_embeddings.items() 
        if gene in selected_genes
    }
    
    print(f"原始基因embedding数: {len(gene_embeddings)}")
    print(f"过滤后基因embedding数: {len(filtered_gene_embeddings)}")
    
    return filtered_gene_embeddings


def normalize_rna_seq_data(adata, normalize=True):
    """对RNA-seq数据进行CP10K+log1P归一化"""
    if not normalize:
        print("跳过归一化，直接使用原始表达数据")
        return adata.copy()
        
    print("正在对RNA-seq数据进行CP10K+log1P归一化...")
    
    # 复制数据以避免修改原始数据
    adata_norm = adata.copy()
    
    # CP10K归一化 (每个细胞总计数归一化到10000)
    sc.pp.normalize_total(adata_norm, target_sum=1e4)
    
    # log1p转换
    sc.pp.log1p(adata_norm)
    
    print("归一化完成")
    return adata_norm


def calculate_cell_embeddings(adata_norm, gene_embeddings, output_dir):
    """计算细胞embedding"""
    print("正在计算细胞embedding...")
    
    # 获取有embedding的基因在adata中的索引
    available_genes = []
    gene_indices = []
    embedding_list = []
    
    for i, gene in enumerate(adata_norm.var.index):
        if gene in gene_embeddings:
            available_genes.append(gene)
            gene_indices.append(i)
            embedding_list.append(gene_embeddings[gene])
    
    print(f"在RNA-seq数据中找到 {len(available_genes)} 个有embedding的基因")
    
    if len(available_genes) == 0:
        print("警告：没有找到匹配的基因，无法计算细胞embedding")
        return None
    
    # 构建基因embedding矩阵 (基因数 x embedding维度)
    gene_emb_matrix = np.array(embedding_list)
    
    # 提取对应基因的表达矩阵 (细胞数 x 基因数)
    expression_matrix = adata_norm.X[:, gene_indices]
    
    # 如果是稀疏矩阵，转换为密集矩阵
    if hasattr(expression_matrix, 'toarray'):
        expression_matrix = expression_matrix.toarray()
    
    # 矩阵乘法: (细胞数 x 基因数) @ (基因数 x embedding维度) = (细胞数 x embedding维度)
    cell_embeddings = expression_matrix @ gene_emb_matrix
    
    print(f"细胞embedding形状: {cell_embeddings.shape}")
    
    # 保存细胞embedding
    os.makedirs(output_dir, exist_ok=True)
    cell_emb_file = os.path.join(output_dir, "cell_embedding.npy")
    np.save(cell_emb_file, cell_embeddings)
    
    print(f"细胞embedding已保存到: {cell_emb_file}")
    
    return cell_embeddings, available_genes


def analyze_multi_protein_genes(gene_protein_mapping):
    """分析有多个蛋白质的基因"""
    multi_protein_stats = {
        "多蛋白质基因统计": {},
        "多蛋白质基因详情": []
    }
    
    protein_count_distribution = {}
    multi_protein_genes = []
    
    for gene_mapping in gene_protein_mapping:
        gene_id = gene_mapping["gene_id"]
        
        # 计算该基因对应的蛋白质数量
        protein_count = len([key for key in gene_mapping.keys() if key.startswith("protein_id_")])
        
        if protein_count > 0:
            if protein_count not in protein_count_distribution:
                protein_count_distribution[protein_count] = 0
            protein_count_distribution[protein_count] += 1
            
            if protein_count > 1:
                proteins = [gene_mapping[key] for key in gene_mapping.keys() if key.startswith("protein_id_")]
                multi_protein_genes.append({
                    "gene_id": gene_id,
                    "protein_count": protein_count,
                    "proteins": proteins
                })
    
    multi_protein_stats["多蛋白质基因统计"] = {
        "蛋白质数量分布": protein_count_distribution,
        "总基因数": len(gene_protein_mapping),
        "有蛋白质映射的基因数": sum(protein_count_distribution.values()),
        "多蛋白质基因数": len(multi_protein_genes),
        "多蛋白质基因比例": f"{len(multi_protein_genes)/sum(protein_count_distribution.values())*100:.2f}%" if sum(protein_count_distribution.values()) > 0 else "0%"
    }
    
    multi_protein_stats["多蛋白质基因详情"] = multi_protein_genes
    
    return multi_protein_stats


def match_gene_protein_embeddings(gene_protein_mapping, protein_to_embedding):
    """匹配基因-蛋白质映射与embedding"""
    print("正在匹配基因与蛋白质embedding...")
    
    gene_embeddings = {}
    matched_genes = []
    unmatched_proteins = []
    genes_without_embedding = []
    
    for gene_mapping in tqdm(gene_protein_mapping, desc="匹配embedding"):
        gene_id = gene_mapping["gene_id"]
        
        # 收集该基因对应的所有蛋白质embedding
        embeddings_for_gene = []
        found_proteins = []
        missing_proteins = []
        
        # 检查该基因是否有蛋白质映射
        has_protein_mapping = any(key.startswith("protein_id_") for key in gene_mapping.keys())
        
        if has_protein_mapping:
            for key, protein_id in gene_mapping.items():
                if key.startswith("protein_id_"):
                    if protein_id in protein_to_embedding:
                        embeddings_for_gene.append(protein_to_embedding[protein_id])
                        found_proteins.append(protein_id)
                    else:
                        missing_proteins.append(protein_id)
            
            if embeddings_for_gene:
                # 如果有多个embedding，取平均
                if len(embeddings_for_gene) == 1:
                    gene_embeddings[gene_id] = embeddings_for_gene[0]
                else:
                    # 多个蛋白质embedding取平均
                    avg_embedding = np.mean(embeddings_for_gene, axis=0)
                    gene_embeddings[gene_id] = avg_embedding
                
                matched_genes.append({
                    "gene_id": gene_id,
                    "found_proteins": found_proteins,
                    "missing_proteins": missing_proteins,
                    "embedding_count": len(embeddings_for_gene)
                })
            else:
                # 有蛋白质映射但没有embedding
                genes_without_embedding.append({
                    "gene_id": gene_id,
                    "proteins": missing_proteins
                })
                unmatched_proteins.extend(missing_proteins)
        else:
            # 没有蛋白质映射的基因在之前已经记录
            pass
    
    print(f"成功匹配 {len(gene_embeddings)} 个基因到embedding")
    print(f"有蛋白质映射但无embedding的基因数: {len(genes_without_embedding)}")
    print(f"未找到embedding的蛋白质数: {len(set(unmatched_proteins))}")
    
    return gene_embeddings, matched_genes, genes_without_embedding, list(set(unmatched_proteins))


def create_mapping_statistics(gene_names, unmapped_genes, matched_genes, genes_without_embedding, unmatched_proteins, multi_protein_stats):
    """创建映射统计信息"""
    total_genes = len(gene_names)
    unmapped_gene_count = len(unmapped_genes)
    matched_gene_count = len(matched_genes)
    genes_without_embedding_count = len(genes_without_embedding)
    
    # 计算有蛋白质映射的基因总数
    genes_with_protein_mapping = matched_gene_count + genes_without_embedding_count
    
    statistics = {
        "总览": {
            "总基因数": total_genes,
            "有蛋白质映射的基因数": genes_with_protein_mapping,
            "无蛋白质映射的基因数": unmapped_gene_count,
            "有embedding的基因数": matched_gene_count,
            "有蛋白质映射但无embedding的基因数": genes_without_embedding_count
        },
        "比例": {
            "蛋白质映射成功率": f"{genes_with_protein_mapping/total_genes*100:.2f}%",
            "embedding匹配成功率": f"{matched_gene_count/total_genes*100:.2f}%",
            "在有蛋白质映射中embedding匹配率": f"{matched_gene_count/genes_with_protein_mapping*100:.2f}%" if genes_with_protein_mapping > 0 else "0%"
        },
        "详细信息": {
            "未找到蛋白质映射的基因总数": unmapped_gene_count,
            "未找到embedding的唯一蛋白质数": len(unmatched_proteins)
        },
        "多蛋白质基因分析": multi_protein_stats["多蛋白质基因统计"]
    }
    
    return statistics


def save_results(gene_protein_mapping, gene_embeddings, unmapped_genes, matched_genes, 
                genes_without_embedding, unmatched_proteins, gene_names, multi_protein_stats, adata_norm, output_dir):
    """保存结果"""
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存基因到蛋白质映射（JSON格式）
    mapping_file = os.path.join(output_dir, "gene_to_protein_mapping.json")
    with open(mapping_file, 'w', encoding='utf-8') as f:
        json.dump(gene_protein_mapping, f, indent=2, ensure_ascii=False)
    print(f"基因到蛋白质映射已保存到: {mapping_file}")
    
    # 保存基因embedding（h5ad格式）
    if gene_embeddings:
        # 按照基因名称排序以确保一致性
        sorted_genes = sorted(gene_embeddings.keys())
        embeddings_matrix = np.array([gene_embeddings[gene] for gene in sorted_genes])
        
        # 创建AnnData对象，基因名称作为观测值(obs)，embedding维度作为变量(var)
        import anndata as adata_pkg
        
        # 创建观测值索引（基因名称）
        obs_names = sorted_genes
        
        # 创建变量索引（embedding维度）
        var_names = [f"emb_dim_{i}" for i in range(embeddings_matrix.shape[1])]
        
        # 创建AnnData对象
        adata = adata_pkg.AnnData(
            X=embeddings_matrix,
            obs=pd.DataFrame(index=obs_names),
            var=pd.DataFrame(index=var_names)
        )
        
        # 保存为h5ad文件
        embedding_file = os.path.join(output_dir, "gene_embeddings.h5ad")
        adata.write(embedding_file)
        
        print(f"基因embedding已保存到: {embedding_file}")
        print(f"数据形状: {adata.shape} (基因数 x embedding维度)")
        print(f"基因数: {len(sorted_genes)}, Embedding维度: {embeddings_matrix.shape[1]}")
    
    # 保存未映射情况统计
    unmapped_stats = {
        "未映射到蛋白质的基因": unmapped_genes,
        "有蛋白质映射但无embedding的基因": genes_without_embedding,
        "未找到embedding的蛋白质": unmatched_proteins,
        "成功匹配的基因详情": matched_genes
    }
    
    unmapped_file = os.path.join(output_dir, "unmapped_analysis.json")
    with open(unmapped_file, 'w', encoding='utf-8') as f:
        json.dump(unmapped_stats, f, indent=2, ensure_ascii=False)
    print(f"未映射情况分析已保存到: {unmapped_file}")
    
    # 保存统计摘要
    statistics = create_mapping_statistics(gene_names, unmapped_genes, matched_genes, 
                                         genes_without_embedding, unmatched_proteins, multi_protein_stats)
    
    stats_file = os.path.join(output_dir, "mapping_statistics.json")
    with open(stats_file, 'w', encoding='utf-8') as f:
        json.dump(statistics, f, indent=2, ensure_ascii=False)
    print(f"映射统计信息已保存到: {stats_file}")
    
    # 注释：这里不再额外保存到固定路径，统一使用save_path参数控制输出位置
    
    # 保存预处理后的RNA-seq数据
    if adata_norm is not None:
        # 创建预处理后的h5ad文件，确保细胞顺序与cell_embedding.npy一致
        preprocessed_file = os.path.join(output_dir, "preprocessed.h5ad")
        
        # 复制预处理后的数据
        adata_save = adata_norm.copy()
        
        # 确保有cell ID和gene ID信息
        if adata_save.obs.index.name is None:
            adata_save.obs.index.name = 'cell_id'
        if adata_save.var.index.name is None:
            adata_save.var.index.name = 'gene_id'
            
        # 保存预处理后的数据
        adata_save.write(preprocessed_file)
        print(f"预处理后的RNA-seq数据已保存到: {preprocessed_file}")
        print(f"数据形状: {adata_save.shape} (细胞数 x 基因数)")
        print(f"细胞顺序与cell_embedding.npy一致")
        
        # 注释：移除额外保存，统一使用save_path控制
    
    # 打印统计摘要
    print("\n=== 映射统计摘要 ===")
    for category, stats in statistics.items():
        print(f"\n{category}:")
        for key, value in stats.items():
            print(f"  {key}: {value}")


def main():
    parser = argparse.ArgumentParser(description="基因ID到蛋白质embedding匹配")
    parser.add_argument("--hgnc_file", type=str, required=True,
                       help="HGNC数据文件路径")
    parser.add_argument("--h5ad_file", type=str, required=True,
                       help="RNA-seq h5ad文件路径")
    parser.add_argument("--embedding_dir", type=str, required=True,
                       help="蛋白质embedding文件目录（包含.npy和.xlsx文件）")
    parser.add_argument("--id_column", type=str, default="ID",
                       help="蛋白质ID文件中的ID列名称")
    parser.add_argument("--save_path", type=str, required=True,
                       help="输出目录")
    parser.add_argument("--max_proteins", type=int, default=10,
                       help="每个基因最多保留的蛋白质数量")
    parser.add_argument("--seed", type=int, default=42,
                       help="随机种子")
    parser.add_argument("--top_k", type=int, default=None,
                       help="每个细胞选择的top基因数量（默认不进行筛选）")
    parser.add_argument("--no_normalize", action="store_true",
                       help="不进行CP10K+log1P归一化（默认进行归一化）")
    
    args = parser.parse_args()
    
    # 设置随机种子
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    print("=== 基因ID到蛋白质embedding匹配 ===")
    print(f"HGNC文件: {args.hgnc_file}")
    print(f"RNA-seq文件: {args.h5ad_file}")
    print(f"Embedding目录: {args.embedding_dir}")
    print(f"ID列名: {args.id_column}")
    print(f"输出目录: {args.save_path}")
    print(f"最大蛋白质数: {args.max_proteins}")
    print(f"Top基因数: {args.top_k if args.top_k else '不筛选'}")
    print(f"归一化: {'否' if args.no_normalize else '是'}")
    
    try:
        # 步骤1：加载HGNC数据
        gene_to_uniprot = load_hgnc_data(args.hgnc_file)
        
        # 步骤2：加载RNA-seq数据
        adata, gene_names = load_rna_seq_data(args.h5ad_file)
        
        # 步骤3：获取每个细胞的top基因（如果指定）
        top_genes_per_cell = get_top_genes_per_cell(adata, args.top_k)
        
        # 步骤4：映射基因到蛋白质
        gene_protein_mapping, unmapped_genes = map_genes_to_proteins(gene_names, gene_to_uniprot, args.max_proteins)
        
        # 步骤5：分析多蛋白质基因
        multi_protein_stats = analyze_multi_protein_genes(gene_protein_mapping)
        
        # 步骤6：加载蛋白质embedding
        protein_to_embedding = load_protein_embeddings(args.embedding_dir, args.id_column)
        
        # 步骤7：匹配基因与embedding
        gene_embeddings, matched_genes, genes_without_embedding, unmatched_proteins = \
            match_gene_protein_embeddings(gene_protein_mapping, protein_to_embedding)
        
        # 步骤8：根据top基因选择过滤基因embedding
        filtered_gene_embeddings = filter_genes_by_top_selection(adata, gene_embeddings, top_genes_per_cell)
        
        # 步骤9：RNA-seq数据归一化
        adata_norm = normalize_rna_seq_data(adata, normalize=not args.no_normalize)
        
        # 步骤10：计算细胞embedding
        cell_emb_result = calculate_cell_embeddings(adata_norm, filtered_gene_embeddings, args.save_path)
        
        # 步骤11：保存结果
        save_results(gene_protein_mapping, filtered_gene_embeddings, unmapped_genes, matched_genes,
                    genes_without_embedding, unmatched_proteins, gene_names, multi_protein_stats, adata_norm, args.save_path)
        
        print("\n=== 处理完成 ===")
        print(f"总基因数: {len(gene_names)}")
        print(f"有蛋白质映射的基因数: {sum(1 for entry in gene_protein_mapping if len(entry) > 1)}")
        print(f"有embedding的基因数: {len(gene_embeddings)}")
        print(f"过滤后基因数: {len(filtered_gene_embeddings)}")
        if cell_emb_result:
            cell_embeddings, available_genes = cell_emb_result
            print(f"细胞embedding形状: {cell_embeddings.shape}")
            print(f"用于计算细胞embedding的基因数: {len(available_genes)}")
        
        # 输出多蛋白质基因统计
        multi_protein_count = len(multi_protein_stats["多蛋白质基因详情"])
        print(f"多蛋白质基因数: {multi_protein_count}")
        if multi_protein_count > 0:
            print("蛋白质数量分布:", multi_protein_stats["多蛋白质基因统计"]["蛋白质数量分布"])
        
    except Exception as e:
        print(f"错误: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())