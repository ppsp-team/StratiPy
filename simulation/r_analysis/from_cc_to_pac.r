###########################################################
#  from consensus matrix, get PAC index
###########################################################

# Ref: Critical limitations of consensus clustering in class discovery
# Yasin Senbabaoglu, George Michailidis & Jun Z. Li
# Scientific Reports 4, Article number: 6207 (2014)
# doi:10.1038/srep06207

# "the "proportion of ambigous clustering" (PAC),
# defined as the fraction of sample pairs with 
# consensus index values falloing in the intermediate
# sub-interval (x1, x2), element of [0, 1]. A low value
# of PAC indicates a flat middle segment, allowing inference
# of the optimal K by the lowest PAC"

setwd(dir="github/stratipy/simulation/r_analysis/")
# install.packages("R.matlab")

# file.path() dosen't work on Windows
#file.path('..', 'output', 'nbs', 'consensus_clustering', 'diff', 'gnmf', 'consensus_weight=min_simp=True_alpha=0.7_tol=0.01_singletons=False_ngh=3_minMut=0_maxMut=100_comp=2_permut=1000_lambd=200_tolNMF=0.001.mat')

filename <- "../output/nbs/consensus_clustering/diff/gnmf/consensus_weight=min_simp=True_alpha=0.7_tol=0.01_singletons=False_ngh=3_minMut=0_maxMut=100_comp=2_permut=1000_lambd=200_tolNMF=0.001.mat"
library(R.matlab)
data <- readMat(filename)

cm <- data$distance.patients 

# install.packages("diceR")
library(diceR)

x1 <- 0.05
x2 <- 0.95
PAC(cm, lower=x1, upper=x2)

###########################################################

base_file <- "../output/nbs/consensus_clustering"
mut_type <- c("raw", "diff", "mean_qn", "median_qn")
nmf_type <- c("nmf", "gnmf")
influence_weight <- "min"
simplification <- "True"
alpha <- c(0, 0.7)
tol <- 10e-3
keep_singletons <- "False"
ngh_max <- 3
min_mutation <- 0
max_mutation <- 100
n_components <- 2:11
n_permutations <- 1000
lambd <- c(0, 200, 1800)
tol_nmf <- 1e-3

a_greek <- intToUtf8(0x03B1)
l_greek <- intToUtf8(0x03BB)

df <- data.frame(type=character(),
                 mut_type=character(),
                 nmf_type=character(),
                 influence_weight=character(),
                 simplification=logical(),
                 alpha=numeric(),
                 tol=numeric(),
                 keep_singletons=logical(),
                 ngh_max=integer(),
                 min_mutation=integer(),
                 max_mutation=integer(),
                 n_components=integer(),
                 n_permutations=integer(),
                 lambd=integer(),
                 tol_nmf=numeric(),
                 pac=numeric(),
                 stringsAsFactors = FALSE)

# iterate all consensus matrix file names
for (i0 in mut_type){
  f0 <- paste(base_file, i0, sep="/")
  
  for (i1 in nmf_type){
    f1 <- paste(f0, i1, sep='/')
    
    for (i2 in influence_weight){
      f2 <- paste(f1, i2, sep='/consensus_weight=')
      
      for (i3 in simplification){
        f3 <- paste(f2, i3, sep="_simp=")
        
        for (i4 in alpha){
          if(i0!="raw" & i4==0) next
          if(i0=="raw" & i4!=0) next
          f4 <- paste(f3, i4, sep="_alpha=")
          
          for (i5 in tol){
            f5 <- paste(f4, i5, sep="_tol=")
            
            for (i6 in keep_singletons){
              f6 <- paste(f5, i6, sep="_singletons=")
              
              for (i7 in ngh_max){
                f7 <- paste(f6, i7, sep="_ngh=")
                
                for (i8 in min_mutation){
                  f8 <- paste(f7, i8, sep="_minMut=")
                  
                  for (i9 in max_mutation){
                    f9 <- paste(f8, i9, sep="_maxMut=")
                    
                    for (i10 in n_components){
                      f10 <- paste(f9, i10, sep="_comp=")
                    
                      for (i11 in n_permutations){
                        f11 <- paste(f10, i11, sep="_permut=")
                      
                        for (i12 in lambd){
                          if(i1=="nmf" & i12!=0) next
                          if(i1=="gnmf" & i12==0) next
                          f12 <- paste(f11, i12, sep="_lambd=")
                        
                          for (i13 in tol_nmf){
                            f13 <- paste(f12, i13, sep="_tolNMF=")
                            filename <- paste(f13, ".mat", sep="")
                            
                            if (file.exists(filename)){
                              ia <- paste0(a_greek, "=", i4)
                              il <- paste0(l_greek, "=", i12)
                              # ingh <- paste0(i7, "ngh")
                              type_name <- paste(toupper(i1), i0, ia, il)
                              
                              # PCA function and save as a file
                              data <- readMat(filename)
                              cons_mat <- data$distance.patients
                              pac <- PAC(cons_mat, lower=x1, upper=x2)

                              # add all values to data frame
                              df[nrow(df)+1,] = c(type_name, i0,i1,i2,i3,i4,i5,i6,i7,i8,i9,i10,i11,i12,i13,pac)
                            }
                            else print(paste0("No such file: ", filename))
                          }
                        }
                      }
                    }
                  }
                }
              }
            }
          }
        }
      }
    }
  }
}


# convert some colums' mode (because all mode is character)
logic_col <- c("simplification","keep_singletons")
numeric_col <- c("alpha","tol","tol_nmf","pac")
int_col <- c("ngh_max","min_mutation","max_mutation","n_components","n_permutations","lambd")

for (i in logic_col) df[, i] <- as.logical(df[, i])
for (i in numeric_col) df[, i] <- as.numeric(df[, i])
for (i in int_col) df[, i] <- as.integer(df[, i])

# save data frame as csv file
# write.csv(df, file="all_param_pca.csv", row.names=FALSE, sep="\t")

# a <- read.csv(file="test.csv")
# b <- df[, c("n_components", "pac")]



###########################################################
library(ggplot2)
# install.packages('svglite')
require("ggplot2")
# plot saving function
# savePlot <- function(myPlot) {
#   pdf("myPlot.pdf")
#   print(myPlot)
#   dev.off()
# }

# global plot
ggplot(data=df, aes(x=n_components, y=pac, group=type)) +
  geom_line(aes(color=type)) +
  geom_point(aes(color=type)) +
  scale_x_continuous(breaks=n_components) +
  labs(x="Component number", y="PAC") +
  theme_bw() +
  theme(legend.title=element_blank(), panel.grid.minor.x = element_blank())

ggsave("all_data.png", plot = last_plot(), path = "plot",
       width = 7, height = 6, dpi = 300)

# get some rows for plot

# select_rows <- function(data, x1, x2, x3, x4, x5){
#   subset(data,
#          mut_type==x1 & 
#            nmf_type==x2 & 
#            alpha==x3 &
#            ngh_max==x4 &
#            lambd==x5)
# }
# 
# nmf_raw <- select_rows(df, "raw", "nmf", alpha[1], ngh_max, lambd[1])
# nmf_diff <- select_rows(df, "diff", "nmf", alpha[2], ngh_max, lambd[1])
# nmf_mean <- select_rows(df, "mean_qn", "nmf", alpha[2], ngh_max, lambd[1])
# nmf_median <- select_rows(df, "median_qn", "nmf", alpha[2], ngh_max, lambd[1])

nmf_df <- subset(df, nmf_type=="nmf")
gnmf200_df <- subset(df, nmf_type=="gnmf" & lambd==200)
gnmf1800_df <- subset(df, nmf_type=="gnmf"& lambd==1800)

# NMF plot
ggplot(data=nmf_df, aes(x=n_components, y=pac, group=type)) +
  geom_line(aes(color=type)) +
  geom_point(aes(color=type)) +
  scale_x_continuous(breaks=n_components) +
  labs(x="Component number", y="PAC") +
  theme_bw() +
  theme(legend.title=element_blank(), panel.grid.minor.x = element_blank())

ggsave("nmf.png", plot = last_plot(), path = "plot",
       width = 7, height = 6, dpi = 300)

# GNMF (lambd=200) plot
ggplot(data=gnmf200_df, aes(x=n_components, y=pac, group=type)) +
  geom_line(aes(color=type)) +
  geom_point(aes(color=type)) +
  scale_x_continuous(breaks=n_components) +
  labs(x="Component number", y="PAC") +
  theme_bw() +
  theme(legend.title=element_blank(), panel.grid.minor.x = element_blank())

ggsave("gnmf_200.png", plot = last_plot(), path = "plot",
       width = 7, height = 6, dpi = 300)

# GNMF (lambd=1800) plot
ggplot(data=gnmf1800_df, aes(x=n_components, y=pac, group=type)) +
  geom_line(aes(color=type)) +
  geom_point(aes(color=type)) +
  scale_x_continuous(breaks=n_components) +
  labs(x="Component number", y="PAC") +
  theme_bw() +
  theme(legend.title=element_blank(), panel.grid.minor.x = element_blank())

ggsave("gnmf_1800.png", plot = last_plot(), path = "plot",
       width = 7, height = 6, dpi = 300)


###########################################################


