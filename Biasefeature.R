.csPrediction <- function(W,Y0,method){
  ###This function implements the label propagation to predict the label(subtype) for new patients.	
  ### note method is an indicator of which semi-supervised method to use
  # method == 0 indicates to use the local and global consistency method
  # method >0 indicates to use label propagation method.
  
  alpha=0.9;
  P= W/rowSums(W)
  if(method==0){
    Y= (1-alpha)* solve( diag(dim(P)[1])- alpha*P)%*%Y0;
  } else {
    NLabel=which(rowSums(Y0)==0)[1]-1;
    Y=Y0;
    for (i in 1:1000){
      Y=P%*%Y;
      Y[1:NLabel,]=Y0[1:NLabel,];
    }
  }
  return(Y);
}

.discretisation <- function(eigenVectors) {
  
  normalize <- function(x) x / sqrt(sum(x^2))
  eigenVectors = t(apply(eigenVectors,1,normalize))
  
  n = nrow(eigenVectors)
  k = ncol(eigenVectors)
  
  R = matrix(0,k,k)
  R[,1] = t(eigenVectors[round(n/2),])
  
  mini <- function(x) {
    i = which(x == min(x))
    return(i[1])
  }
  
  c = matrix(0,n,1)
  for (j in 2:k) {
    c = c + abs(eigenVectors %*% matrix(R[,j-1],k,1))
    i = mini(c)
    R[,j] = t(eigenVectors[i,])
  }
  
  lastObjectiveValue = 0
  for (i in 1:20) {
    eigenDiscrete = .discretisationEigenVectorData(eigenVectors %*% R)
    
    svde = svd(t(eigenDiscrete) %*% eigenVectors)
    U = svde[['u']]
    V = svde[['v']]
    S = svde[['d']]
    
    NcutValue = 2 * (n-sum(S))
    if(abs(NcutValue - lastObjectiveValue) < .Machine$double.eps) 
      break
    
    lastObjectiveValue = NcutValue
    R = V %*% t(U)
    
  }
  
  return(list(discrete=eigenDiscrete,continuous =eigenVectors))
}

.discretisationEigenVectorData <- function(eigenVector) {
  
  Y = matrix(0,nrow(eigenVector),ncol(eigenVector))
  maxi <- function(x) {
    i = which(x == max(x))
    return(i[1])
  }
  j = apply(eigenVector,1,maxi)
  Y[cbind(1:nrow(eigenVector),j)] = 1
  
  return(Y)
  
}

.dominateset <- function(xx,KK=20) {
  ###This function outputs the top KK neighbors.	
  
  zero <- function(x) {
    s = sort(x, index.return=TRUE)
    x[s$ix[1:(length(x)-KK)]] = 0
    return(x)
  }
  normalize <- function(X) X / rowSums(X)
  A = matrix(0,nrow(xx),ncol(xx));
  for(i in 1:nrow(xx)){
    A[i,] = zero(xx[i,]);
    
  }
  
  
  return(normalize(A))
}

# Calculate the mutual information between vectors x and y.
.mutualInformation <- function(x, y) {
  classx <- unique(x)
  classy <- unique(y)
  nx <- length(x)
  ncx <- length(classx)
  ncy <- length(classy)
  
  probxy <- matrix(NA, ncx, ncy)
  for (i in 1:ncx) {
    for (j in 1:ncy) {
      probxy[i, j] <- sum((x == classx[i]) & (y == classy[j])) / nx
    }
  }
  
  probx <- matrix(rowSums(probxy), ncx, ncy)
  proby <- matrix(colSums(probxy), ncx, ncy, byrow=TRUE)
  result <- sum(probxy * log(probxy / (probx * proby), 2), na.rm=TRUE)
  return(result)
}

# Calculate the entropy of vector x.
.entropy <- function(x) {
  class <- unique(x)
  nx <- length(x)
  nc <- length(class)
  
  prob <- rep.int(NA, nc)
  for (i in 1:nc) {
    prob[i] <- sum(x == class[i])/nx
  }
  
  result <- -sum(prob * log(prob, 2))
  return(result)
}

.repmat = function(X,m,n){
  ##R equivalent of repmat (matlab)
  if (is.null(dim(X))) {
    mx = length(X)
    nx = 1
  } else {
    mx = dim(X)[1]
    nx = dim(X)[2]
  }
  matrix(t(matrix(X,mx,nx*n)),mx*m,nx*n,byrow=T)
}

KNN_SMI <- function(W,K=20,t=20) {
  ###This function is the main function of our software. The inputs are as follows:
  # W: an affinity matrix
  # K : number of neighbors
  # t : number of iterations for fusion
  
  ###The output is a unified similarity graph. It contains both complementary information and common structures from all individual network. 
  ###You can do various applications on this graph, such as clustering(subtyping), classification, prediction.
  
  
  normalize <- function(X) X / rowSums(X)
  # makes elements other than largest K zero
  
  
  newW =matrix(0,dim(W)[1], dim(W)[2])
  nextW =matrix(0,dim(W)[1], dim(W)[2])
  ###First, normalize different networks to avoid scale problems.
  W = normalize(W);
  W = (W+t(W))/2;
  
  
  ### Calculate the local transition matrix.
  newW = (.dominateset(W,K))
  
  nextW=W;
  # perform the diffusion for t iterations
  for (i in 1:t) {
    nextW=newW %*% (nextW) %*% t(newW);
  }
  ###Normalize each new obtained networks.
  
  W = nextW + diag(nrow(W));
  W = (W + t(W))/2;
  
  
  return(W)    
}

library(SNFtool)

Express=read.table("/Users/cyx/SNF_gai/R/newfea2000/Biase/Biase3celltypes.txt",header = T,
                   comment.char='!',stringsAsFactors = FALSE,quote = "",sep='\t')
#View(Express)

dim(Express)
Express=Express[,-1]
len=ncol(Express)
Express=Express[apply(Express,1,function(x) sum(x>1)>len*0.05),]
Express=Express[apply(Express,1,function(x) sum(x>1)<len*0.95),]
Express=apply(Express,2,as.numeric)
num=apply(Express, 1, sd)

Express=as.array(Express)
num=as.array(num)
#View(sort(apply(Express, 1, sd),decreasing = TRUE))
Total<-cbind(num,Express)
Total1=Total[order(Total[,1],decreasing = TRUE),]
Total1=Total1[1:2000,]
Total2=Total1[,-1]
#View(Total2)
Data1=Total2
Data1=t(Data1)
#library("FactoMineR")
#library("factoextra")
#res.pca <- PCA(Data1, graph = FALSE)
#eig.val <- get_eigenvalue(res.pca)
#eig.val
#fviz_eig(res.pca, addlabels = TRUE, ylim = c(0, 50))
Exp.pca=prcomp(Data1,center = TRUE,scale. = TRUE)
plot(Exp.pca, type = "l")
names(Exp.pca)
summary(Exp.pca)
#View(summary(Exp.pca))
plot(Exp.pca$x, main="after PCA")
new.Exp<-as.data.frame(Exp.pca$x[,1:22])
#View(new.Exp)
new.Exp=apply(new.Exp,2,as.numeric)
new.Exp=t(new.Exp)
#View(new.Exp)
new.Exp=t(new.Exp)
Dist1 = dist2(as.matrix(new.Exp),as.matrix(new.Exp))
K=10
alpha = 0.5;
T=100
W1 = affinityMatrix(Dist1, K, alpha)
W = KNN_SMI(W1, K, T)
#View(W)
library(apcluster)
apresla<-apcluster(negDistMat(r=7),W)
s1<-negDistMat(W,r=7)
apreslb<-apcluster(s1)
apreslb
b = spectralClustering(W, 3)
a<-c(1,1,1,1,1,1,1,1,1,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3)
group = spectralClustering(W, 3)
b=group
#b=t(b)
adjustedRandIndex(a,b)


# 使用 UMAP 进行降维
umap_result <- umap(Data1)

# 提取 UMAP 的二维坐标
umap_coords <- as.data.frame(umap_result$layout)
colnames(umap_coords) <- c("UMAP1", "UMAP2")
umap_coords$label <- b

# 绘制散点图
ggplot(umap_coords, aes(x = UMAP1, y = UMAP2, color = as.factor(label))) +
  geom_point(size = 3, alpha = 0.8) +
  theme_minimal() +
  labs(title = "UMAP Dimensionality Reduction",
       x = "UMAP1",
       y = "UMAP2",
       color = "Label") +
  theme(plot.title = element_text(hjust = 0.5, size = 16))














aver_gene<-function(X){
  summa <- matrix(rep(1:ncol(Total2),1*ncol(Total2)),1,ncol(Total2))
  Total3=rbind(summa,Total2)
  p <<- matrix(rep(1:(nrow(Total3)),1*(nrow(Total3))),(nrow(Total3)),1)
  #View(p)
  for (i in 1:ncol(Total3)) {
    for (j in 1:ncol(X)) {
      if(Total3[1,i]==X[1,j]){
        p<<-cbind(p,Total3[,i])
      }
    }
  }
  #View(p)
}

clu=apreslb@clusters
qq=t(unlist(clu[1]))
aver_gene(qq)
pp1=p
qq=t(unlist(clu[2]))
aver_gene(qq)
pp2=p
qq=t(unlist(clu[3]))
aver_gene(qq)
pp3=p
#View(pp2)
#View(Total2)

pp1=pp1[-1,-1]
pp2=pp2[-1,-1]
pp3=pp3[-1,-1]

kstotal1 <- matrix(rep(0,2000*1),2000,1)
for (i in 1:2000) {
  zx=t.test(pp1[i,],Total2[i,])
  kstotal1[i]=zx$p.value
}
#View(kstotal1)
kstotal2 <- matrix(rep(0,2000*1),2000,1)
for (i in 1:2000) {
  zx=t.test(pp2[i,],Total2[i,])
  kstotal2[i]=zx$p.value
}

kstotal3 <- matrix(rep(0,2000*1),2000,1)
for (i in 1:2000) {
  zx=t.test(pp3[i,],Total2[i,])
  kstotal3[i]=zx$p.value
}

toks=cbind(kstotal1,kstotal2,kstotal3)

mintoks=apply(toks,1,mad)
#View(mintoks)
ksid <- matrix(rep(1:2000,2000*1),2000,1)

kstotal=cbind(ksid,mintoks)
kstotal=kstotal[order(kstotal[,2],decreasing = FALSE),]
kstota<-kstotal[kstotal[,2]<=0.05,]
View(kstota)

library("fdrtool")
zzz=t(kstota[,-1])
zzz=as.vector(zzz)
fdrtool(zzz, statistic="pvalue")
fdr<-fdrtool(zzz, statistic="pvalue")
fdr$qval

ss=kstota[,1]
ss=as.array(ss)
#View(ss)

perval <- matrix(rep(1:nrow(Total2),nrow(Total2)*1),nrow(Total2),1)
Total4=cbind(perval,Total2)
#View(Total4)
#View(pp)
pp <- matrix(rep(1:ncol(Total4)),1*ncol(Total4),1,ncol(Total4))
pp=t(pp)
for (i in 1:nrow(Total4)) {
  for (j in 1:nrow(ss)) {
    if(Total4[i,1]==ss[j]){
      pp=rbind(pp,Total4[i,])
    }
  }
}




pp=pp[-1,-1]
pp=t(pp)
#View(pp)
K = 10;
alpha = 0.5;
T = 100;
Data1 = standardNormalization(pp)
Dist1 = dist2(as.matrix(Data1),as.matrix(Data1))
W1 = affinityMatrix(Dist1, K, alpha)
W = KNN_SMI(W1, K, T)
#View(W)
library(apcluster)
apresla<-apcluster(negDistMat(r=7),W)
s1<-negDistMat(W,r=7)
apreslb<-apcluster(s1)
apreslb
group = spectralClustering(W, 3)
library(mclust)
#yandata
a<-c(1,1,1,1,1,1,1,1,1,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3)

b=group
#b=t(b)
adjustedRandIndex(a,b)
b=apreslb@idx
