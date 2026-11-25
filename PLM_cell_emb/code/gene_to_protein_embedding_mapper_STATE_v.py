#!/usr/bin/env python3
"""
基于STATE基因embedding的蛋白质embedding匹配和细胞embedding计算脚本
功能：
1. 将RNA-seq基因ID转换为UniProt ID
2. 匹配蛋白质embedding
3. 使用STATE基因embedding作为辅助
4. 支持top基因选择和归一化控制
5. 多个蛋白质embedding取平均
6. 统计未映射情况
"""

import argparse
import json
import os
import random
import pandas as pd
import numpy as np
import scanpy as sc
import anndata as ad
from tqdm import tqdm
import glob
import logging

# 从现有脚本导入重用函数
from gene_to_protein_embedding_mapper import (
    load_hgnc_data,
    load_rna_seq_data,
    map_genes_to_proteins,
    extract_uniprot_id,
    find_embedding_files,
    load_protein_embeddings,
    get_top_genes_per_cell,
    filter_genes_by_top_selection,
    normalize_rna_seq_data,
    analyze_multi_protein_genes,
    match_gene_protein_embeddings,
    create_mapping_statistics,
    save_results
)

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_state_gene_embeddings(state_h5ad_path):
    """
    加载STATE模型的基因嵌入
    
    Args:
        state_h5ad_path (str): STATE_tok_5120_Emb.h5ad文件路径
        
    Returns:
        dict: 基因名到嵌入向量的映射
        int: 嵌入维度
    """
    logger.info(f"正在加载STATE基因嵌入: {state_h5ad_path}")
    
    if not os.path.exists(state_h5ad_path):
        raise FileNotFoundError(f"STATE嵌入文件不存在: {state_h5ad_path}")
    
    # 加载AnnData对象
    state_adata = ad.read_h5ad(state_h5ad_path)
    
    logger.info(f"STATE嵌入数据形状: {state_adata.shape}")
    logger.info(f"基因数量: {state_adata.n_obs}")
    logger.info(f"嵌入维度: {state_adata.n_vars}")
    
    # 创建基因名到嵌入的映射
    gene_to_embedding = {}
    gene_names = state_adata.obs.index.tolist()
    embeddings = state_adata.X
    
    for i, gene_name in enumerate(gene_names):
        if hasattr(embeddings, 'toarray'):
            # 如果是稀疏矩阵
            gene_embedding = embeddings[i, :].toarray().flatten()
        else:
            # 如果是密集矩阵
            gene_embedding = embeddings[i, :].flatten()
        gene_to_embedding[gene_name] = gene_embedding
    
    embedding_dim = state_adata.n_vars
    
    logger.info(f"成功加载 {len(gene_to_embedding)} 个基因的STATE嵌入")
    
    return gene_to_embedding, embedding_dim


def combine_protein_and_state_embeddings(gene_embeddings, state_gene_embeddings, combination_method="protein_priority"):
    """
    组合蛋白质embedding和STATE基因embedding
    
    Args:
        gene_embeddings (dict): 从蛋白质获得的基因embedding
        state_gene_embeddings (dict): STATE基因embedding
        combination_method (str): 组合方法
            - "protein_priority": 优先使用蛋白质embedding，缺失时用STATE
            - "state_priority": 优先使用STATE embedding，缺失时用蛋白质
            - "average": 对于有两种embedding的基因，取平均值
    
    Returns:
        dict: 组合后的基因embedding
    """
    logger.info(f"正在组合蛋白质和STATE基因embedding，方法: {combination_method}")
    
    combined_embeddings = {}
    protein_only = 0
    state_only = 0
    both_available = 0
    
    # 获取所有基因
    all_genes = set(gene_embeddings.keys()) | set(state_gene_embeddings.keys())
    
    for gene in all_genes:
        has_protein = gene in gene_embeddings
        has_state = gene in state_gene_embeddings
        
        if has_protein and has_state:
            both_available += 1
            if combination_method == "protein_priority":
                combined_embeddings[gene] = gene_embeddings[gene]
            elif combination_method == "state_priority":
                combined_embeddings[gene] = state_gene_embeddings[gene]
            elif combination_method == "average":
                # 需要处理维度不同的情况
                protein_emb = gene_embeddings[gene]
                state_emb = state_gene_embeddings[gene]
                if len(protein_emb) == len(state_emb):
                    combined_embeddings[gene] = (protein_emb + state_emb) / 2
                else:
                    logger.warning(f"基因 {gene} 的蛋白质embedding和STATE embedding维度不同，使用蛋白质embedding")
                    combined_embeddings[gene] = protein_emb
        elif has_protein:
            protein_only += 1
            combined_embeddings[gene] = gene_embeddings[gene]
        elif has_state:
            state_only += 1
            combined_embeddings[gene] = state_gene_embeddings[gene]
    
    logger.info(f"组合结果 - 仅蛋白质: {protein_only}, 仅STATE: {state_only}, 两者都有: {both_available}")
    logger.info(f"组合后总基因数: {len(combined_embeddings)}")
    
    return combined_embeddings


def calculate_state_cell_embeddings(adata_norm, combined_gene_embeddings, output_dir):
    """
    使用组合的基因embedding计算细胞embedding
    """
    logger.info("正在计算基于组合基因embedding的细胞embedding...")
    
    # 获取有embedding的基因在adata中的索引
    available_genes = []
    gene_indices = []
    embedding_list = []
    
    for i, gene in enumerate(adata_norm.var.index):
        if gene in combined_gene_embeddings:
            available_genes.append(gene)
            gene_indices.append(i)
            embedding_list.append(combined_gene_embeddings[gene])
    
    logger.info(f"在RNA-seq数据中找到 {len(available_genes)} 个有组合embedding的基因")
    
    if len(available_genes) == 0:
        logger.warning("没有找到匹配的基因，无法计算细胞embedding")
        return None
    
    # 构建基因embedding矩阵 (基因数 x embedding维度)
    gene_emb_matrix = np.array(embedding_list)
    logger.info(f"基因embedding矩阵形状: {gene_emb_matrix.shape}")
    
    # 提取对应基因的表达矩阵 (细胞数 x 基因数)
    expression_matrix = adata_norm.X[:, gene_indices]
    
    # 如果是稀疏矩阵，转换为密集矩阵
    if hasattr(expression_matrix, 'toarray'):
        expression_matrix = expression_matrix.toarray()
    
    logger.info(f"表达矩阵形状: {expression_matrix.shape}")
    
    # 矩阵乘法: (细胞数 x 基因数) @ (基因数 x embedding维度) = (细胞数 x embedding维度)
    cell_embeddings = expression_matrix @ gene_emb_matrix
    
    logger.info(f"细胞embedding形状: {cell_embeddings.shape}")
    
    # 保存细胞embedding
    os.makedirs(output_dir, exist_ok=True)
    cell_emb_file = os.path.join(output_dir, "state_combined_cell_embedding.npy")
    np.save(cell_emb_file, cell_embeddings)
    
    logger.info(f"组合细胞embedding已保存到: {cell_emb_file}")
    
    return cell_embeddings, available_genes


def analyze_gene_matching_state(rna_seq_genes, state_genes, protein_genes, output_dir):
    """
    分析RNA-seq基因与STATE基因、蛋白质基因的匹配情况
    """
    logger.info("分析基因匹配情况...")
    
    os.makedirs(output_dir, exist_ok=True)
    
    rna_seq_set = set(rna_seq_genes)
    state_set = set(state_genes)
    protein_set = set(protein_genes)
    
    # 计算各种交集
    state_matched = rna_seq_set & state_set
    protein_matched = rna_seq_set & protein_set
    both_matched = state_matched & protein_matched
    only_state = state_matched - protein_matched
    only_protein = protein_matched - state_matched
    no_match = rna_seq_set - (state_matched | protein_matched)
    
    matching_stats = {
        "RNA-seq总基因数": len(rna_seq_genes),
        "STATE总基因数": len(state_genes),
        "蛋白质总基因数": len(protein_genes),
        "STATE匹配基因数": len(state_matched),
        "蛋白质匹配基因数": len(protein_matched),
        "两者都匹配基因数": len(both_matched),
        "仅STATE匹配基因数": len(only_state),
        "仅蛋白质匹配基因数": len(only_protein),
        "完全无匹配基因数": len(no_match),
        "STATE匹配率": f"{len(state_matched)/len(rna_seq_genes)*100:.2f}%",
        "蛋白质匹配率": f"{len(protein_matched)/len(rna_seq_genes)*100:.2f}%",
        "总体覆盖率": f"{len(state_matched | protein_matched)/len(rna_seq_genes)*100:.2f}%"
    }
    
    # 保存详细匹配信息
    matching_details = {
        "两者都匹配的基因": sorted(list(both_matched)),
        "仅STATE匹配的基因": sorted(list(only_state))[:100],
        "仅蛋白质匹配的基因": sorted(list(only_protein))[:100],
        "完全无匹配的基因": sorted(list(no_match))[:100],
        "统计信息": matching_stats
    }
    
    matching_file = os.path.join(output_dir, "state_protein_gene_matching_analysis.json")
    with open(matching_file, 'w', encoding='utf-8') as f:
        json.dump(matching_details, f, indent=2, ensure_ascii=False)
    
    logger.info(f"基因匹配分析已保存到: {matching_file}")
    
    # 打印匹配统计
    print(f"\n=== STATE与蛋白质基因匹配分析 ===")
    for key, value in matching_stats.items():
        print(f"{key}: {value}")
    
    return matching_stats


def save_state_results(gene_protein_mapping, combined_gene_embeddings, unmapped_genes, matched_genes,
                      genes_without_embedding, unmatched_proteins, gene_names, multi_protein_stats, 
                      adata_norm, state_matching_stats, output_dir):
    """保存STATE版本的结果"""
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存基因到蛋白质映射（JSON格式）
    mapping_file = os.path.join(output_dir, "state_gene_to_protein_mapping.json")
    with open(mapping_file, 'w', encoding='utf-8') as f:
        json.dump(gene_protein_mapping, f, indent=2, ensure_ascii=False)
    logger.info(f"基因到蛋白质映射已保存到: {mapping_file}")
    
    # 保存组合基因embedding
    if combined_gene_embeddings:
        sorted_genes = sorted(combined_gene_embeddings.keys())
        embeddings_matrix = np.array([combined_gene_embeddings[gene] for gene in sorted_genes])
        
        # 保存为npy格式
        combined_emb_file = os.path.join(output_dir, "state_combined_gene_embeddings.npy")
        np.save(combined_emb_file, embeddings_matrix)
        
        # 保存基因名列表
        genes_file = os.path.join(output_dir, "state_combined_gene_names.txt")
        with open(genes_file, 'w', encoding='utf-8') as f:
            for gene in sorted_genes:
                f.write(f"{gene}\n")
        
        logger.info(f"组合基因embedding已保存到: {combined_emb_file}")
        logger.info(f"基因名列表已保存到: {genes_file}")
        logger.info(f"数据形状: {embeddings_matrix.shape} (基因数 x embedding维度)")
    
    # 保存未映射情况统计
    unmapped_stats = {
        "未映射到蛋白质的基因": unmapped_genes,
        "有蛋白质映射但无embedding的基因": genes_without_embedding,
        "未找到embedding的蛋白质": unmatched_proteins,
        "成功匹配的基因详情": matched_genes,
        "STATE匹配统计": state_matching_stats
    }
    
    unmapped_file = os.path.join(output_dir, "state_unmapped_analysis.json")
    with open(unmapped_file, 'w', encoding='utf-8') as f:
        json.dump(unmapped_stats, f, indent=2, ensure_ascii=False)
    logger.info(f"未映射情况分析已保存到: {unmapped_file}")
    
    # 保存统计摘要
    statistics = create_mapping_statistics(gene_names, unmapped_genes, matched_genes, 
                                         genes_without_embedding, unmatched_proteins, multi_protein_stats)
    
    # 添加STATE特定统计
    statistics["STATE分析"] = state_matching_stats
    
    stats_file = os.path.join(output_dir, "state_mapping_statistics.json")
    with open(stats_file, 'w', encoding='utf-8') as f:
        json.dump(statistics, f, indent=2, ensure_ascii=False)
    logger.info(f"映射统计信息已保存到: {stats_file}")
    
    # 保存预处理后的RNA-seq数据
    if adata_norm is not None:
        preprocessed_file = os.path.join(output_dir, "state_preprocessed.h5ad")
        adata_save = adata_norm.copy()
        
        if adata_save.obs.index.name is None:
            adata_save.obs.index.name = 'cell_id'
        if adata_save.var.index.name is None:
            adata_save.var.index.name = 'gene_id'
            
        adata_save.write(preprocessed_file)
        logger.info(f"预处理后的RNA-seq数据已保存到: {preprocessed_file}")
        logger.info(f"数据形状: {adata_save.shape} (细胞数 x 基因数)")
    
    # 打印统计摘要
    print(f"\n=== STATE版本映射统计摘要 ===")
    for category, stats in statistics.items():
        print(f"\n{category}:")
        if isinstance(stats, dict):
            for key, value in stats.items():
                print(f"  {key}: {value}")
        else:
            print(f"  {stats}")


def main():
    parser = argparse.ArgumentParser(description="基于STATE基因embedding的蛋白质embedding匹配和细胞embedding计算")
    parser.add_argument("--hgnc_file", type=str, required=True,
                       help="HGNC数据文件路径")
    parser.add_argument("--h5ad_file", type=str, required=True,
                       help="RNA-seq h5ad文件路径")
    parser.add_argument("--embedding_dir", type=str, required=True,
                       help="蛋白质embedding文件目录（包含.npy和.xlsx文件）")
    parser.add_argument("--state_embedding", type=str, required=True,
                       help="STATE基因embedding h5ad文件路径")
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
    parser.add_argument("--combination_method", type=str, default="protein_priority",
                       choices=["protein_priority", "state_priority", "average"],
                       help="蛋白质和STATE embedding组合方法")
    
    args = parser.parse_args()
    
    # 设置随机种子
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    logger.info("=== 基于STATE的基因ID到蛋白质embedding匹配 ===")
    logger.info(f"HGNC文件: {args.hgnc_file}")
    logger.info(f"RNA-seq文件: {args.h5ad_file}")
    logger.info(f"蛋白质Embedding目录: {args.embedding_dir}")
    logger.info(f"STATE Embedding文件: {args.state_embedding}")
    logger.info(f"ID列名: {args.id_column}")
    logger.info(f"输出目录: {args.save_path}")
    logger.info(f"最大蛋白质数: {args.max_proteins}")
    logger.info(f"Top基因数: {args.top_k if args.top_k else '不筛选'}")
    logger.info(f"归一化: {'否' if args.no_normalize else '是'}")
    logger.info(f"组合方法: {args.combination_method}")
    
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
        
        # 步骤7：匹配基因与蛋白质embedding
        gene_embeddings, matched_genes, genes_without_embedding, unmatched_proteins = \
            match_gene_protein_embeddings(gene_protein_mapping, protein_to_embedding)
        
        # 步骤8：加载STATE基因embedding
        state_gene_embeddings, state_embedding_dim = load_state_gene_embeddings(args.state_embedding)
        
        # 步骤9：分析基因匹配情况
        state_matching_stats = analyze_gene_matching_state(
            gene_names, list(state_gene_embeddings.keys()), list(gene_embeddings.keys()), args.save_path)
        
        # 步骤10：组合蛋白质和STATE embedding
        combined_gene_embeddings = combine_protein_and_state_embeddings(
            gene_embeddings, state_gene_embeddings, args.combination_method)
        
        # 步骤11：根据top基因选择过滤基因embedding
        filtered_combined_embeddings = filter_genes_by_top_selection(adata, combined_gene_embeddings, top_genes_per_cell)
        
        # 步骤12：RNA-seq数据归一化
        adata_norm = normalize_rna_seq_data(adata, normalize=not args.no_normalize)
        
        # 步骤13：计算细胞embedding
        cell_emb_result = calculate_state_cell_embeddings(adata_norm, filtered_combined_embeddings, args.save_path)
        
        # 步骤14：保存结果
        save_state_results(gene_protein_mapping, filtered_combined_embeddings, unmapped_genes, matched_genes,
                          genes_without_embedding, unmatched_proteins, gene_names, multi_protein_stats, 
                          adata_norm, state_matching_stats, args.save_path)
        
        print(f"\n=== STATE版本处理完成 ===")
        print(f"总基因数: {len(gene_names)}")
        print(f"有蛋白质映射的基因数: {sum(1 for entry in gene_protein_mapping if len(entry) > 1)}")
        print(f"有蛋白质embedding的基因数: {len(gene_embeddings)}")
        print(f"有STATE embedding的基因数: {len(state_gene_embeddings)}")
        print(f"组合后基因数: {len(combined_gene_embeddings)}")
        print(f"过滤后基因数: {len(filtered_combined_embeddings)}")
        
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
        logger.error(f"错误: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())