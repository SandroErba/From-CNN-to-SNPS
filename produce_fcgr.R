##convert genomic sequences in fcgr 100x100
setwd('/mnt/d/UniVr/ProgettoNatComp_CGR/From-CNN-to-SNPS')

library(kaos)
library(seqinr)
library(foreach)
library(doParallel)

cl <- makeCluster(10)
registerDoParallel(cl)

A = list.files('ncbi_dataset_Helicobacter/ncbi_dataset/data',pattern='GCF',full.names=T)
print(head(A))
foreach (i=1:length(A),.packages = c('kaos','seqinr')) %dopar%{
  if (paste0('fcgr_',tail(unlist(strsplit(A[i],split='/')),n=1),'.txt') %in% list.files(path='dataset/Helicobacter/')){}
  else{         
  seq = read.fasta(list.files(paste0(A[i],'/'),full.names = T)[1],as.string=T)
  seq = unlist(strsplit(seq[[1]],split=''))
  cgr_list = cgr(seq,res = 100)
  write.table(cgr_list$matrix,file=paste0('dataset/Helicobacter/fcgr_',tail(unlist(strsplit(A[i],split='/')),n=1),'.txt'),
              row.names = F,quote = F,col.names = F)
}
  }

stopCluster(cl)
