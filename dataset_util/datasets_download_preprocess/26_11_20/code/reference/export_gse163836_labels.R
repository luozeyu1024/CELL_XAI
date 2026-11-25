args <- commandArgs(trailingOnly=TRUE)
library(Matrix)
load(args[1])
stopifnot(ncol(rawCounts)==length(rawCounts.grp),!anyDuplicated(colnames(rawCounts)))
if(!is.null(names(rawCounts.grp))) stopifnot(identical(names(rawCounts.grp),colnames(rawCounts)))
stopifnot(all(rawCounts@x>=0 & rawCounts@x==floor(rawCounts@x)))
d <- data.frame(Cell_ID=colnames(rawCounts),label=ifelse(rawCounts.grp=='FCIBC02','sensitive','resistant'),
                original_group=as.character(rawCounts.grp),
                cell_line=ifelse(rawCounts.grp=='SUM149PR','SUM149','FCIBC02'),
                drug='paclitaxel',label_scope='cell_line_derived',
                matched_parental_available=rawCounts.grp!='SUM149PR',
                source='GEO GSM4988075/GSM4988076/GSM4988077; rawCounts.grp',check.names=FALSE)
d$sample_id <- unname(c(FCIBC02='GSM4988075',SUM149PR='GSM4988076',FCIBC02PR='GSM4988077')[d$original_group])
d$batch <- ifelse(d$original_group=='FCIBC02','Batch 1','Batch 2')
stopifnot(!anyNA(d$sample_id))
write.table(d,file=file.path(args[2],'cell_labels.tsv'),sep='\t',quote=FALSE,row.names=FALSE)
write.table(data.frame(n_genes=nrow(rawCounts),n_cells=ncol(rawCounts),nnz=length(rawCounts@x),min_nonzero=min(rawCounts@x),max_count=max(rawCounts@x),all_nonnegative_integer=TRUE,cell_label_alignment=TRUE),file=file.path(args[2],'validation.tsv'),sep='\t',quote=FALSE,row.names=FALSE)
print(table(d$original_group,d$label));print(dim(rawCounts))
