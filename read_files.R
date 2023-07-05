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

cora <-
  list.files(pattern = "^Cora_False_gnn_results_4715015_[0-9]+\\.csv$", full.names=TRUE) %>% 
  map_df(~read.csv(.))

citeseer <-
  list.files(pattern = "^Citeseer_False_gnn_results_4860351_[0-9]+\\.csv$", full.names=TRUE) %>% 
  map_df(~read.csv(.))

df = blob

df %>%
  select(model, alpha, gnn_type, sp,acc, local, density)%>%
  group_by(model, alpha, gnn_type)%>%
  summarise_all(list(mean = ~mean(.), sd = ~sd(.)))%>%
  write.csv(.,file = "scurve.csv")

df %>%
  select(model, alpha, gnn_type, sp,acc, local, density)%>%
  mutate_at(vars(alpha),list(factor)) %>%
  #filter(gnn_type=="symmetric")%>%
  #summarise_all(list(mean = ~mean(.), sd = ~sd(.)))%>%
  ggplot(aes(x=alpha, y = sp, color = gnn_type))+
  geom_boxplot(notch = TRUE)+
  #geom_errorbar(aes(ymin = sp_mean-sp_sd, ymax = sp_mean+sp_sd), width=1.0, position = position_dodge(0.05))+
  #geom_line(aes(color=model))+
  #geom_point(aes(color=model), size = 0.5)+
  theme_bw()+
  #geom_smooth(aes(color=model))+
  #scale_color_brewer(palette="Paired")+theme_minimal()+
  facet_grid(cols = vars(model))+
  xlab(expression(alpha)) + ylab("global structure(unsupervised)")


df %>%
  select(model, alpha, gnn_type, sp,acc, local, density)%>%
  mutate_at(vars(alpha),list(factor)) %>%
  filter(model=="CCA-SSG")%>%
  #summarise_all(list(mean = ~mean(.), sd = ~sd(.)))%>%
  ggplot(aes(x=alpha, y = sp, color = gnn_type))+
  geom_boxplot(notch = TRUE)+
  #geom_errorbar(aes(ymin = sp_mean-sp_sd, ymax = sp_mean+sp_sd), width=1.0, position = position_dodge(0.05))+
  #geom_line(aes(color=model))+
  #geom_point(aes(color=model), size = 0.5)+
  theme_bw()+
  #geom_smooth(aes(color=model))+
  #scale_color_brewer(palette="Paired")+theme_minimal()+
  facet_grid(cols = vars(model))+
  xlab(expression(alpha)) + ylab("global structure(supervised)")

a = df %>%
  select(model, alpha, gnn_type, sp,acc, local, density)%>%
  mutate_at(vars(alpha),list(factor))%>%
  filter(model=="CCA-SSG", gnn_type=="symmetric")
  
df %>%
  select(model, alpha, gnn_type, sp,acc, local, density)%>%
  mutate_at(vars(alpha),list(factor)) %>%
  #filter(gnn_type=="symmetric")%>%
  #summarise_all(list(mean = ~mean(.), sd = ~sd(.)))%>%
  ggplot(aes(x=alpha, y = local, color = gnn_type))+
  geom_boxplot(notch = TRUE)+
  #geom_errorbar(aes(ymin = sp_mean-sp_sd, ymax = sp_mean+sp_sd), width=1.0, position = position_dodge(0.05))+
  #geom_line(aes(color=model))+
  #geom_point(aes(color=model), size = 0.5)+
  theme_bw()+
  #geom_smooth(aes(color=model))+
  #scale_color_brewer(palette="Paired")+theme_minimal()+
  facet_grid(cols = vars(model))+
  xlab(expression(alpha)) + ylab("local structure")



df %>%
  select(model, alpha, gnn_type, sp,acc, local, density)%>%
  #group_by(model, alpha, gnn_type)%>%
  #filter(gnn_type=="symmetric")%>%
  #summarise_all(list(mean = ~mean(.), sd = ~sd(.)))%>%
  ggplot(aes(x=alpha, y = density, group=model))+
  #geom_errorbar(aes(ymin = sp_mean-sp_sd, ymax = sp_mean+sp_sd), width=1.0, position = position_dodge(0.05))+
  #geom_line(aes(color=model))+
  #geom_point(aes(color=model), size = 0.5)+
  geom_boxplot()+
  theme_bw()+
  #geom_smooth(aes(color=model))+
  #geom_jitter(aes(color=model))+
  scale_color_brewer(palette="Paired")+theme_minimal()+
  facet_grid(cols = vars(gnn_type))

