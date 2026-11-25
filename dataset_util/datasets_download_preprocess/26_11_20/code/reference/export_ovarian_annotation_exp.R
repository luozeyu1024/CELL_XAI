args <- commandArgs(trailingOnly=TRUE)
if (length(args)!=2) stop('Expected author SCE RDS and output TSV')
suppressPackageStartupMessages(library(SummarizedExperiment))
if (!requireNamespace('SingleCellExperiment',quietly=TRUE)) {
  # Read inherited ordinary metadata slots only; no SCE computation is performed.
  namespace <- new.env(parent=baseenv()); info <- new.env(parent=baseenv())
  assign('spec',c(name='SingleCellExperiment',version='1.10.1'),envir=info)
  assign('.__NAMESPACE__.',info,envir=namespace)
  .Internal(registerNamespace('SingleCellExperiment',namespace))
}
object <- readRDS(args[1])
metadata <- as.data.frame(attr(object,'colData'))
stopifnot(all(c('Barcode','hpca.celltype','encode.celltype') %in% names(metadata)))
# Exact ordered decision sets from the author's R/tenx.R .decideCellType.
groups <- list(
  EPI=c('Epithelial_cells','Epithelial cells','Embryonic_stem_cells','MSC','iPS_cells','Keratinocytes','Neuroepithelial_cell','Neurons'),
  MYE=c('Macrophage','Monocyte','Macrophages','Monocytes','DC','Erythrocytes','Neutrophils'),
  LYMPH=c('T_cells','B_cell','Pre-B_cell_CD34-','NK_cell','CD8+ T-cells','CD4+ T-cells','HSC','B-cells','NK cells','Pro-B_cell_CD34+'),
  ENDO=c('Endothelial_cells','Endothelial cells'),
  STROM=c('Fibroblasts','Tissue_stem_cells','Smooth_muscle_cells','Osteoblasts','Chondrocytes','Myocytes','Pericytes','Smooth muscle','Adipocytes'))
metadata$rule_celltype <- vapply(seq_len(nrow(metadata)),function(i) {
  for (name in names(groups)) {
    if (metadata$hpca.celltype[i] %in% groups[[name]] || metadata$encode.celltype[i] %in% groups[[name]]) return(name)
  }
  return('unknown')
},character(1))
if ('celltype' %in% names(metadata)) {
  metadata$author_celltype <- as.character(metadata$celltype)
  metadata$author_celltype[is.na(metadata$author_celltype) | metadata$author_celltype==''] <- 'unknown'
  metadata$annotation_source <- 'saved author celltype; missing labels retained'
} else {
  metadata$author_celltype <- metadata$rule_celltype
  metadata$annotation_source <- 'author ordered hpca/encode decision rule'
}
dir.create(dirname(args[2]),recursive=TRUE,showWarnings=FALSE)
write.table(metadata,args[2],sep='\t',row.names=FALSE,quote=FALSE,na='')
print(table(metadata$author_celltype)); cat('EXPORTED',nrow(metadata),'\n')
