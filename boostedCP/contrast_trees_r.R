library(conTree)
library(rjson)

contrast_rfun=function(x_input,y_input,upper,lower,miscov_rate,treesize,output_dir_node,output_dir_tree,min.node,print_tree="FALSE"){
  mydiscr = function(y,z,w){
    # input(vectors):
    # y = y-values in node
    # w = z-values in node
    # w = observation weights in node
    #
    # let y be the index of x wrt the original X
    cond_cov = abs(1-mean((y_input[y]<=upper[y])*(y_input[y]>=lower[y]))-miscov_rate)
    return(cond_cov)
    #
    #output(scalar):
    # corresponding computed discrepancy
  }
  tree = contrast(x_input, seq_len(length(y_input)), y_input,type=mydiscr,tree.size = treesize,min.node=min.node)

  node_sum = nodesum(tree,x_input, seq_len(length(y_input)), y_input)
  nx = getnodes(tree,x_input)
  node_sum$nx=nx
  jsonData <- toJSON(node_sum)
  write(jsonData, output_dir_node)
  
  if(print_tree!="FALSE"){
  cat(print_tree, " contrast tree:\n")
  tree_sum=treesum(tree)
  jsonData <- toJSON(tree_sum)
  write(jsonData, output_dir_tree)
  }   
}



seed=Sys.getenv("seed")
ratio=Sys.getenv("ratio")
data_name=Sys.getenv("data_name")
print_tree=Sys.getenv("print_tree")

if(print_tree!="FALSE"){
cat("\n[R] data_name:",data_name,"\n")
cat("[R] seed:",seed,"\n")
cat("[R] ratio:",ratio,"\n\n")
}

cache_dir=paste("cache/",data_name,"/seed_",seed,"_ratio_",ratio,"_contrast_rfun.json",sep="")
cache_dir_tree=paste("cache/",data_name,"/seed_",seed,"_ratio_",ratio,"_contrast_rfun_",print_tree,".json",sep="")

inputs <- fromJSON(file = cache_dir)
x_input =  do.call(rbind,inputs$x_input)
y_input = inputs$y_input
upper = inputs$upper
lower =  inputs$lower
treesize =  inputs$treesize
miscov_rate=inputs$miscov_rate

n_base=500
if(length(y_input)<n_base){
    min.node=max(50,min(n_base,nrow(x_input)/treesize))
}else{
    min.node=max(100,min(n_base,nrow(x_input)/treesize))
}

contrast_rfun(x_input,y_input,upper,lower,miscov_rate,treesize,cache_dir,cache_dir_tree,min.node,print_tree)




