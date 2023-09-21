import rpy2.robjects as robjects

# Define R script
r_script = '''
setwd("~/Desktop/emb_images/peanutcsv")

library(tidyverse)
# library(conflicted)
library(data.table)
library(ggplot2)

blob <-
  list.files(pattern = "^Blobs_25023_[0-9]+\\.csv$", full.names=TRUE) %>%
  map_df(~read.csv(.))

sphere <-
  list.files(pattern = "^Sphere_24978_[0-9]+\\.csv$", full.names=TRUE) %>%
  map_df(~read.csv(.))

circles <-
  list.files(pattern = "^Circles_24878_[0-9]+\\.csv$", full.names=TRUE) %>%
  map_df(~read.csv(.))

moons <-
  list.files(pattern = "^Moons_24928_[0-9]+\\.csv$", full.names=TRUE) %>%
  map_df(~read.csv(.))

swissroll <-
  list.files(pattern = "^Swissroll_25024_[0-9]+\\.csv$", full.names=TRUE) %>%
  map_df(~read.csv(.))



df = sphere

df %>%
  select(model, alpha, gnn_type, sp,acc, local, density)%>%
  group_by(model, alpha, gnn_type)%>%
  summarise_all(list(mean = ~mean(.), sd = ~sd(.)))
  #write.csv(.,file = "sphere.csv")

df %>%
  select(model, alpha, gnn_type, sp,acc, local, density)%>%
  group_by(model, alpha, gnn_type)%>%
  #filter(gnn_type=="symmetric")%>%
  summarise_all(list(mean = ~mean(.), sd = ~sd(.)))%>%
  ggplot(aes(x=alpha, y = sp_mean, group=model))+
  #geom_errorbar(aes(ymin = sp_mean-sp_sd, ymax = sp_mean+sp_sd), width=1.0, position = position_dodge(0.05))+
  geom_line(aes(color=model))+
  #scale_color_brewer(palette="Paired")+theme_minimal()+
  facet_grid(cols = vars(gnn_type))+
  xlab(expression(alpha)) + ylab("global structure(unsupervised)")


df %>%
  select(model, alpha, gnn_type, sp,acc, local, density)%>%
  group_by(model, alpha, gnn_type)%>%
  #filter(gnn_type=="symmetric")%>%
  summarise_all(list(mean = ~mean(.), sd = ~sd(.)))%>%
  ggplot(aes(x=alpha, y = acc_mean, group=model))+
  #geom_errorbar(aes(ymin = sp_mean-sp_sd, ymax = sp_mean+sp_sd), width=1.0, position = position_dodge(0.05))+
  geom_line(aes(color=model))+
  #geom_point()+
  #scale_color_brewer(palette="Paired")+theme_minimal()+
  facet_grid(cols = vars(gnn_type))+
  xlab(expression(alpha)) + ylab("global structure(supervised)")

df %>%
  select(model, alpha, gnn_type, sp,acc, local, density)%>%
  group_by(model, alpha, gnn_type)%>%
  #filter(gnn_type=="symmetric")%>%
  summarise_all(list(mean = ~mean(.), sd = ~sd(.)))%>%
  ggplot(aes(x=alpha, y = local_mean, group=model))+
  #geom_errorbar(aes(ymin = sp_mean-sp_sd, ymax = sp_mean+sp_sd), width=1.0, position = position_dodge(0.05))+
  geom_line(aes(color=model))+
  #geom_point()+
  #scale_color_brewer(palette="Paired")+theme_minimal()+
  facet_grid(cols = vars(gnn_type))+
  xlab(expression(alpha)) + ylab("local structure")



df %>%
  select(model, alpha, gnn_type, sp,acc, local, density)%>%
  group_by(model, alpha, gnn_type)%>%
  #filter(gnn_type=="symmetric")%>%
  summarise_all(list(mean = ~mean(.), sd = ~sd(.)))%>%
  ggplot(aes(x=alpha, y = density_mean, group=model))+
  #geom_errorbar(aes(ymin = sp_mean-sp_sd, ymax = sp_mean+sp_sd), width=1.0, position = position_dodge(0.05))+
  geom_line(aes(color=model))+
  #geom_point()+
  #scale_color_brewer(palette="Paired")+theme_minimal()+
  facet_grid(cols = vars(gnn_type))
'''

# Run R script
robjects.r(r_script)

