setwd("~/gnumap/result_cluster/results")

library(tidyverse)
# library(conflicted)
library(data.table)
library(ggplot2)

blob <-
  list.files(pattern = "^Blob_False_gnn_results_4487761_[0-9]+\\.csv$", full.names=TRUE) %>% 
  map_df(~read.csv(.))

sphere <-
  list.files(pattern = "^Sphere_False_gnn_results_4481288_[0-9]+\\.csv$", full.names=TRUE) %>% 
  map_df(~read.csv(.))

circles <-
  list.files(pattern = "^Circles_False_gnn_results_4487866_[0-9]+\\.csv$", full.names=TRUE) %>% 
  map_df(~read.csv(.))

moons <-
  list.files(pattern = "^Moons_False_gnn_results_4487811_[0-9]+\\.csv$", full.names=TRUE) %>% 
  map_df(~read.csv(.))

swissroll <-
  list.files(pattern = "^Swissroll_False_gnn_results_4504910_[0-9]+\\.csv$", full.names=TRUE) %>% 
  map_df(~read.csv(.))

scurve <-
  list.files(pattern = "^Scurve_False_gnn_results_4481340_[0-9]+\\.csv$", full.names=TRUE) %>% 
  map_df(~read.csv(.))



df = blob

df %>%
  select(model, alpha, gnn_type, sp,acc, local, density)%>%
  group_by(model, alpha, gnn_type)%>%
  summarise_all(list(mean = ~mean(.), sd = ~sd(.)))

df %>%
  select(model, alpha, gnn_type, sp,acc, local, density)%>%
  group_by(model, alpha, gnn_type)%>%
  #filter(gnn_type=="symmetric")%>%
  summarise_all(list(mean = ~mean(.), sd = ~sd(.)))%>%
  ggplot(aes(x=alpha, y = sp_mean, group=model))+
  #geom_errorbar(aes(ymin = sp_mean-sp_sd, ymax = sp_mean+sp_sd), width=1.0, position = position_dodge(0.05))+
  geom_line(aes(color=model))+
  geom_point()+
  #scale_color_brewer(palette="Paired")+theme_minimal()+
  facet_grid(cols = vars(gnn_type))


df %>%
  select(model, alpha, gnn_type, sp,acc, local, density)%>%
  group_by(model, alpha, gnn_type)%>%
  #filter(gnn_type=="symmetric")%>%
  summarise_all(list(mean = ~mean(.), sd = ~sd(.)))%>%
  ggplot(aes(x=alpha, y = acc_mean, group=model))+
  #geom_errorbar(aes(ymin = sp_mean-sp_sd, ymax = sp_mean+sp_sd), width=1.0, position = position_dodge(0.05))+
  geom_line(aes(color=model))+
  geom_point()+
  #scale_color_brewer(palette="Paired")+theme_minimal()+
  facet_grid(cols = vars(gnn_type))


df %>%
  select(model, alpha, gnn_type, sp,acc, local, density)%>%
  group_by(model, alpha, gnn_type)%>%
  #filter(gnn_type=="symmetric")%>%
  summarise_all(list(mean = ~mean(.), sd = ~sd(.)))%>%
  ggplot(aes(x=alpha, y = local_mean, group=model))+
  #geom_errorbar(aes(ymin = sp_mean-sp_sd, ymax = sp_mean+sp_sd), width=1.0, position = position_dodge(0.05))+
  geom_line(aes(color=model))+
  geom_point()+
  #scale_color_brewer(palette="Paired")+theme_minimal()+
  facet_grid(cols = vars(gnn_type))



df %>%
  select(model, alpha, gnn_type, sp,acc, local, density)%>%
  group_by(model, alpha, gnn_type)%>%
  #filter(gnn_type=="symmetric")%>%
  summarise_all(list(mean = ~mean(.), sd = ~sd(.)))%>%
  ggplot(aes(x=alpha, y = density_mean, group=model))+
  #geom_errorbar(aes(ymin = sp_mean-sp_sd, ymax = sp_mean+sp_sd), width=1.0, position = position_dodge(0.05))+
  geom_line(aes(color=model))+
  geom_point()+
  #scale_color_brewer(palette="Paired")+theme_minimal()+
  facet_grid(cols = vars(gnn_type))

