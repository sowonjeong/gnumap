if(!require('fastRG')) {
  install.packages('fastRG')
  library('fastRG')
}

set.seed(1)

lazy_dcsbm <- dcsbm(n = 500, k = 5, expected_density = 0.01)
lazy_dcsbm

# sometimes you gotta let the world burn and
# sample a wildly dense graph

dense_lazy_dcsbm <- dcsbm(n = 500, k = 5, expected_density = 0.8)
dense_lazy_dcsbm

# explicitly setting the degree heterogeneity parameter,
# mixing matrix, and relative community sizes rather
# than using randomly generated defaults

k <- 5
n <- 500
B <- matrix(stats::runif(k * k), nrow = k, ncol = k)

theta <- round(stats::rlnorm(n, 2))

pi <- c(1, 2, 4, 1, 1)

custom_dcsbm <- dcsbm(
  theta = theta,
  B = B,
  pi = pi,
  expected_degree = 30
)

custom_dcsbm

# efficient eigendecompostion that leverages low-rank structure in
# E(A) so that you don't have to form E(A) to find eigenvectors,
# as E(A) is typically dense. computation is
# handled via RSpectra

population_eigs <- eigs_sym(custom_dcsbm)

## visualize

library(nett)
library(igraph)

original = par("mar")

# gr = igraph::graph_from_adjacency_matrix(edgelist, "undirected") # convert to igraph object 
par(mar = c(0,0,0,0))
#out = nett::plot_net(fastRG::sample_igraph(custom_dcsbm), community = z)
#fastRG::sample_igraph()


customedgelist <- sample_edgelist(custom_dcsbm)
lazyedgelist <- sample_edgelist(lazy_dcsbm)
denseedgelist <- sample_edgelist(dense_lazy_dcsbm)

lazy_y = lazy_dcsbm[['z']]
levels(lazy_y) = c(1:5)

custom_y = custom_dcsbm[['z']]
levels(custom_y) = c(1:5)

dense_y = dense_lazy_dcsbm[['z']]
levels(dense_y) = c(1:5)


setwd('/Users/sowonjeong/Documents/GitHub/gnumap/SBM/')
write.csv(customedgelist, "customSBM.csv", row.names = FALSE)
write.csv(lazyedgelist, "lazySBM.csv", row.names = FALSE)
write.csv(denseedgelist,"denseSBM.csv", row.names = FALSE)

write.csv(custom_y, "custom_y.csv", row.names = FALSE)
write.csv(lazy_y, "lazy_y.csv", row.names = FALSE)
write.csv(dense_y, "dense_y.csv", row.names = FALSE)