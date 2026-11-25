#!/usr/bin/env python3
"""
GenePT基因embedding到细胞embedding计算脚本
功能：
1. 读取GenePT预训练基因embedding（pickle格式）
2. 与RNA-seq表达矩阵进行加权矩阵乘法
3. 计算细胞embedding
4. 保存结果和统计信息
"""

import argparse
import json
import os
import pickle
import pandas as pd
import numpy as np
import scanpy as sc
from tqdm import tqdm

# 可以直接import原文件中的函数
from gene_to_protein_embedding_mapper import (
    load_rna_seq_data,
    normalize_rna_seq_data,
    calculate_cell_embeddings,
    get_top_genes_per_cell
)


def load_genept_embeddings(pickle_file_path):
    """加载GenePT基因embedding"""
    print("正在加载GenePT基因embedding...")
    
    try:
        with open(pickle_file_path, 'rb') as f:
            gene_embeddings = pickle.load(f)
    except Exception as e:
        print(f"加载GenePT embedding文件失败: {e}")
        return None
    
    print(f"数据类型: {type(gene_embeddings)}")
    
    if not isinstance(gene_embeddings, dict):
        print(f"错误：期望字典格式，但得到 {type(gene_embeddings)}")
        return None
    
    print(f"成功加载 {len(gene_embeddings)} 个基因的embedding")
    
    # 分析基因名格式
    gene_names = list(gene_embeddings.keys())
    ensg_count = sum(1 for name in gene_names if name.startswith('ENSG'))
    hgnc_count = len(gene_names) - ensg_count
    
    print(f"ENSG格式基因: {ensg_count}")
    print(f"HGNC格式基因: {hgnc_count}")
    
    # 检查embedding维度
    if gene_names:
        first_embedding = gene_embeddings[gene_names[0]]
        if isinstance(first_embedding, (list, np.ndarray)):
            embedding_dim = len(first_embedding)
            print(f"Embedding维度: {embedding_dim}")
            
            # 根据维度推断模型类型
            if embedding_dim == 1536:
                model_type = 'text-embedding-ada-002'
            elif embedding_dim == 3072:
                model_type = 'text-embedding-3-large'
            else:
                model_type = f'未知模型 (维度: {embedding_dim})'
            print(f"推断模型类型: {model_type}")
        
        # 显示示例基因名
        print(f"基因名示例: {gene_names[:5]}")
    
    return gene_embeddings


def match_genes_with_rna_seq(gene_embeddings, rna_seq_genes):
    """匹配GenePT基因embedding与RNA-seq数据中的基因"""
    print("正在匹配GenePT基因与RNA-seq数据...")
    
    # 统计匹配情况
    matched_genes = []
    unmatched_rna_genes = []
    unmatched_genept_genes = []
    
    # RNA-seq中的基因集合
    rna_gene_set = set(rna_seq_genes)
    genept_gene_set = set(gene_embeddings.keys())
    
    # 找到匹配的基因
    matched_gene_set = rna_gene_set.intersection(genept_gene_set)
    matched_genes = list(matched_gene_set)
    
    # 找到未匹配的基因
    unmatched_rna_genes = list(rna_gene_set - genept_gene_set)
    unmatched_genept_genes = list(genept_gene_set - rna_gene_set)
    
    print(f"RNA-seq总基因数: {len(rna_seq_genes)}")
    print(f"GenePT总基因数: {len(gene_embeddings)}")
    print(f"匹配基因数: {len(matched_genes)}")
    print(f"RNA-seq中未匹配基因数: {len(unmatched_rna_genes)}")
    print(f"GenePT中未匹配基因数: {len(unmatched_genept_genes)}")
    print(f"匹配率: {len(matched_genes)/len(rna_seq_genes)*100:.2f}%")
    
    # 创建匹配的基因embedding字典
    matched_embeddings = {gene: gene_embeddings[gene] for gene in matched_genes}
    
    return matched_embeddings, matched_genes, unmatched_rna_genes, unmatched_genept_genes


def filter_genept_genes_by_top_selection(gene_embeddings, top_genes_per_cell):
    """根据top基因选择过滤GenePT基因embedding"""
    if top_genes_per_cell is None:
        return gene_embeddings
    
    print("根据top基因选择过滤GenePT基因embedding...")
    
    # 收集所有被选中的基因
    selected_genes = set()
    for cell_top_genes in top_genes_per_cell:
        selected_genes.update(cell_top_genes)
    
    # 过滤基因embedding，只保留被选中的基因
    filtered_gene_embeddings = {
        gene: embedding for gene, embedding in gene_embeddings.items() 
        if gene in selected_genes
    }
    
    print(f"原始GenePT基因embedding数: {len(gene_embeddings)}")
    print(f"过滤后GenePT基因embedding数: {len(filtered_gene_embeddings)}")
    
    return filtered_gene_embeddings


def calculate_genept_cell_embeddings(adata_norm, gene_embeddings, output_dir):
    """
    使用GenePT基因embedding计算细胞embedding
    这是原calculate_cell_embeddings函数的GenePT特化版本
    """
    print("正在计算GenePT细胞embedding...")
    
    # 获取有embedding的基因在adata中的索引
    available_genes = []
    gene_indices = []
    embedding_list = []
    
    for i, gene in enumerate(adata_norm.var.index):
        if gene in gene_embeddings:
            available_genes.append(gene)
            gene_indices.append(i)
            embedding_list.append(gene_embeddings[gene])
    
    print(f"在RNA-seq数据中找到 {len(available_genes)} 个有GenePT embedding的基因")
    
    if len(available_genes) == 0:
        print("警告：没有找到匹配的基因，无法计算细胞embedding")
        return None
    
    # 构建基因embedding矩阵 (基因数 x embedding维度)
    gene_emb_matrix = np.array(embedding_list)
    print(f"基因embedding矩阵形状: {gene_emb_matrix.shape}")
    
    # 提取对应基因的表达矩阵 (细胞数 x 基因数)
    expression_matrix = adata_norm.X[:, gene_indices]
    
    # 如果是稀疏矩阵，转换为密集矩阵
    if hasattr(expression_matrix, 'toarray'):
        expression_matrix = expression_matrix.toarray()
    
    print(f"表达矩阵形状: {expression_matrix.shape}")
    
    # GenePT方法：表达量加权的基因embedding平均
    # 矩阵乘法: (细胞数 x 基因数) @ (基因数 x embedding维度) = (细胞数 x embedding维度)
    cell_embeddings = expression_matrix @ gene_emb_matrix
    
    print(f"细胞embedding形状: {cell_embeddings.shape}")
    
    # 保存细胞embedding
    os.makedirs(output_dir, exist_ok=True)
    cell_emb_file = os.path.join(output_dir, "genept_cell_embedding.npy")
    np.save(cell_emb_file, cell_embeddings)
    
    print(f"GenePT细胞embedding已保存到: {cell_emb_file}")
    
    return cell_embeddings, available_genes


def create_genept_statistics(rna_seq_genes, matched_genes, unmatched_rna_genes, unmatched_genept_genes, gene_embeddings):
    """创建GenePT匹配统计信息"""
    total_rna_genes = len(rna_seq_genes)
    total_genept_genes = len(gene_embeddings)
    matched_count = len(matched_genes)
    
    statistics = {
        "总览": {
            "RNA-seq总基因数": total_rna_genes,
            "GenePT总基因数": total_genept_genes,
            "匹配基因数": matched_count,
            "RNA-seq中未匹配基因数": len(unmatched_rna_genes),
            "GenePT中未匹配基因数": len(unmatched_genept_genes)
        },
        "匹配率": {
            "RNA-seq基因匹配率": f"{matched_count/total_rna_genes*100:.2f}%",
            "GenePT基因利用率": f"{matched_count/total_genept_genes*100:.2f}%"
        },
        "基因名格式分析": {
            "匹配基因中ENSG格式": sum(1 for g in matched_genes if g.startswith('ENSG')),
            "匹配基因中HGNC格式": sum(1 for g in matched_genes if not g.startswith('ENSG'))
        }
    }
    
    # 添加embedding信息
    if matched_genes:
        first_emb = gene_embeddings[matched_genes[0]]
        if isinstance(first_emb, (list, np.ndarray)):
            embedding_dim = len(first_emb)
            if embedding_dim == 1536:
                model_type = 'text-embedding-ada-002'
            elif embedding_dim == 3072:
                model_type = 'text-embedding-3-large'
            else:
                model_type = f'未知模型 (维度: {embedding_dim})'
            
            statistics["Embedding信息"] = {
                "维度": embedding_dim,
                "推断模型": model_type
            }
    
    return statistics


def save_genept_results(matched_embeddings, matched_genes, unmatched_rna_genes, 
                       unmatched_genept_genes, rna_seq_genes, gene_embeddings, 
                       adata_norm, output_dir):
    """保存GenePT结果"""
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存匹配的基因embedding
    if matched_embeddings:
        # 按照基因名称排序以确保一致性
        sorted_genes = sorted(matched_embeddings.keys())
        embeddings_matrix = np.array([matched_embeddings[gene] for gene in sorted_genes])
        
        # 保存为npy格式
        matched_emb_file = os.path.join(output_dir, "genept_matched_gene_embeddings.npy")
        np.save(matched_emb_file, embeddings_matrix)
        
        # 保存基因名列表
        genes_file = os.path.join(output_dir, "genept_matched_gene_names.txt")
        with open(genes_file, 'w', encoding='utf-8') as f:
            for gene in sorted_genes:
                f.write(f"{gene}\n")
        
        print(f"匹配的基因embedding已保存到: {matched_emb_file}")
        print(f"匹配的基因名列表已保存到: {genes_file}")
        print(f"数据形状: {embeddings_matrix.shape} (基因数 x embedding维度)")
    
    # 保存匹配统计信息
    statistics = create_genept_statistics(rna_seq_genes, matched_genes, unmatched_rna_genes, 
                                        unmatched_genept_genes, gene_embeddings)
    
    stats_file = os.path.join(output_dir, "genept_matching_statistics.json")
    with open(stats_file, 'w', encoding='utf-8') as f:
        json.dump(statistics, f, indent=2, ensure_ascii=False)
    print(f"GenePT匹配统计信息已保存到: {stats_file}")
    
    # 保存未匹配基因列表
    unmapped_info = {
        "RNA-seq中未匹配的基因": unmatched_rna_genes[:100],  # 只保存前100个以避免文件过大
        "GenePT中未匹配的基因": unmatched_genept_genes[:100],
        "未匹配基因总数": {
            "RNA-seq": len(unmatched_rna_genes),
            "GenePT": len(unmatched_genept_genes)
        }
    }
    
    unmapped_file = os.path.join(output_dir, "genept_unmapped_genes.json")
    with open(unmapped_file, 'w', encoding='utf-8') as f:
        json.dump(unmapped_info, f, indent=2, ensure_ascii=False)
    print(f"GenePT未匹配基因信息已保存到: {unmapped_file}")
    
    # 保存预处理后的RNA-seq数据
    if adata_norm is not None:
        preprocessed_file = os.path.join(output_dir, "genept_preprocessed.h5ad")
        adata_save = adata_norm.copy()
        
        # 处理obs：检查并删除可能的冲突列
        target_obs_name = 'cell_id'
        if target_obs_name in adata_save.obs.columns:
            if not adata_save.obs[target_obs_name].equals(adata_save.obs.index):
                adata_save.obs.drop(columns=[target_obs_name], inplace=True)
        adata_save.obs.index.name = target_obs_name
        
        # 处理var：检查并删除可能的冲突列
        target_var_name = 'gene_id'
        if target_var_name in adata_save.var.columns:
            if not adata_save.var[target_var_name].equals(adata_save.var.index):
                adata_save.var.drop(columns=[target_var_name], inplace=True)
        adata_save.var.index.name = target_var_name
            
        adata_save.write(preprocessed_file)
        print(f"预处理后的RNA-seq数据已保存到: {preprocessed_file}")
        print(f"数据形状: {adata_save.shape} (细胞数 x 基因数)")
    
    # 打印统计摘要
    print("\n=== GenePT匹配统计摘要 ===")
    for category, stats in statistics.items():
        print(f"\n{category}:")
        for key, value in stats.items():
            print(f"  {key}: {value}")


def main():
    parser = argparse.ArgumentParser(description="GenePT基因embedding到细胞embedding计算")
    parser.add_argument("--genept_pickle", type=str, required=True,
                       help="GenePT基因embedding pickle文件路径")
    parser.add_argument("--h5ad_file", type=str, required=True,
                       help="RNA-seq h5ad文件路径")
    parser.add_argument("--save_path", type=str, required=True,
                       help="输出目录")
    parser.add_argument("--seed", type=int, default=42,
                       help="随机种子")
    parser.add_argument("--top_k", type=int, default=None,
                       help="每个细胞选择的top基因数量（默认不进行筛选）")
    parser.add_argument("--no_normalize", action="store_true",
                       help="不进行CP10K+log1P归一化（默认进行归一化）")
    
    args = parser.parse_args()
    
    # 设置随机种子
    np.random.seed(args.seed)
    
    print("=== GenePT基因embedding到细胞embedding计算 ===")
    print(f"GenePT pickle文件: {args.genept_pickle}")
    print(f"RNA-seq文件: {args.h5ad_file}")
    print(f"输出目录: {args.save_path}")
    print(f"Top基因数: {args.top_k if args.top_k else '不筛选'}")
    print(f"归一化: {'否' if args.no_normalize else '是'}")
    
    try:
        # 步骤1：加载GenePT基因embedding
        gene_embeddings = load_genept_embeddings(args.genept_pickle)
        if gene_embeddings is None:
            return 1
        
        # 步骤2：加载RNA-seq数据
        adata, rna_seq_genes = load_rna_seq_data(args.h5ad_file)
        
        # 步骤3：获取每个细胞的top基因（如果指定）
        top_genes_per_cell = get_top_genes_per_cell(adata, args.top_k)
        
        # 步骤4：匹配基因
        matched_embeddings, matched_genes, unmatched_rna_genes, unmatched_genept_genes = \
            match_genes_with_rna_seq(gene_embeddings, rna_seq_genes)
        
        if not matched_embeddings:
            print("错误：没有找到匹配的基因，无法继续")
            return 1
        
        # 步骤5：根据top基因选择过滤基因embedding
        filtered_matched_embeddings = filter_genept_genes_by_top_selection(matched_embeddings, top_genes_per_cell)
        
        # 步骤6：RNA-seq数据归一化
        adata_norm = normalize_rna_seq_data(adata, normalize=not args.no_normalize)
        
        # 步骤7：计算细胞embedding
        cell_emb_result = calculate_genept_cell_embeddings(adata_norm, filtered_matched_embeddings, args.save_path)
        
        # 步骤8：保存结果
        save_genept_results(filtered_matched_embeddings, matched_genes, unmatched_rna_genes,
                          unmatched_genept_genes, rna_seq_genes, gene_embeddings,
                          adata_norm, args.save_path)
        
        print("\n=== GenePT处理完成 ===")
        print(f"RNA-seq总基因数: {len(rna_seq_genes)}")
        print(f"GenePT总基因数: {len(gene_embeddings)}")
        print(f"匹配基因数: {len(matched_genes)}")
        print(f"过滤后基因数: {len(filtered_matched_embeddings)}")
        print(f"匹配率: {len(matched_genes)/len(rna_seq_genes)*100:.2f}%")
        
        if cell_emb_result:
            cell_embeddings, available_genes = cell_emb_result
            print(f"细胞embedding形状: {cell_embeddings.shape}")
            print(f"用于计算的基因数: {len(available_genes)}")
        
    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())