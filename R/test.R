

library(tidyverse)

results_test <- read_delim("C:/Users/joost/IdeaProjects/evolving-foodweb/results_test.csv",
                           ";", escape_double = FALSE, trim_ws = TRUE)


dsp <- 0.1
est <- 0.00

results_test %>%
  filter(m == dsp,
         e_step == est) %>% 
  distinct(time, patch, environment) %>% 
  mutate(patch = ordered(patch)) %>%
  ggplot(aes(time, environment, color = patch)) +
  geom_line()

results_test %>%
  filter(m == dsp,
         e_step == est) %>% 
  mutate(species = ordered(species),
         patch = ordered(patch),
         bodymass = ordered(bodymass)) %>%
  ggplot(aes(time, N, color = species)) +
  geom_line() +
  scale_y_log10() +
  # scale_colour_viridis_d() +
  # scale_color_brewer(palette = "PuBuGn") +
  facet_grid(patch~bodymass) +
  theme(legend.position = "none")

results_test %>%
  filter(m == dsp,
         e_step == est) %>% 
  distinct(time, X, Y, patch, environment) %>% 
  mutate(patch = ordered(patch)) %>%
  ggplot(aes(time, environment, color = patch)) +
  geom_line() +
  facet_grid(Y~X, labeller = "label_both") +
  theme(legend.position = "none")

results_test %>%
  filter(m == dsp,
         e_step == est) %>% 
  filter(bodymass == 1) %>% 
  mutate(species = ordered(species),
         patch = ordered(patch),
         bodymass = ordered(bodymass)) %>%
  ggplot(aes(time, N, color = species)) +
  geom_line() +
  facet_grid(Y~X, labeller = "label_both") +
  theme(legend.position = "none")

results_test %>%
  filter(m == dsp,
         e_step == est) %>% 
  filter(bodymass == 10) %>% 
  mutate(species = ordered(species),
         patch = ordered(patch),
         bodymass = ordered(bodymass)) %>%
  ggplot(aes(time, N, color = species)) +
  geom_line() +
  facet_grid(Y~X, labeller = "label_both") +
  theme(legend.position = "none")

results_test %>%
  filter(m == dsp,
         e_step == est) %>% 
  filter(bodymass == 100) %>% 
  mutate(species = ordered(species),
         patch = ordered(patch),
         bodymass = ordered(bodymass)) %>%
  ggplot(aes(time, N, color = species)) +
  geom_line() +
  facet_grid(Y~X, labeller = "label_both") +
  theme(legend.position = "none")


results_test %>%
  mutate(species = ordered(species),
         patch = ordered(patch),
         bodymass = ordered(bodymass)) %>%
  group_by(time, species, bodymass) %>% 
  summarise(N = sum(N, na.rm = TRUE)) %>% 
  ungroup() %>% 
  ggplot(aes(time, N, color = species)) +
  geom_line() +
  # scale_y_log10() +
  # scale_colour_viridis_d() +
  # scale_color_brewer(palette = "PuBuGn") +
  facet_wrap(~bodymass, ncol = 1, scales = "free_y")

results_test %>%
  mutate(species = ordered(species),
         patch = ordered(patch),
         bodymass = ordered(bodymass)) %>%
  group_by(time, species, bodymass) %>% 
  summarise(biomass = sum(biomass, na.rm = TRUE)) %>% 
  ungroup() %>% 
  ggplot(aes(time, biomass, color = species)) +
  geom_line() +
  # scale_y_log10() +
  # scale_colour_viridis_d() +
  # scale_color_brewer(palette = "PuBuGn") +
  facet_wrap(~bodymass, ncol = 1, scales = "free_y")


# x_0 <- 0.1
# pow_x <- -0.25
# x_0*c(1,10,100,1000)^pow_x
# 1/(x_0*c(1,10,100,1000)^pow_x)

# n <- 40
# e <- 0.7
# (g <- round(runif(n) * 0.5 * ifelse(rbernoulli(n), -1, 1) + e))
# mean(g)


# D <- data.frame(e = seq(-5,5,0.1),
#                 g = 0,
#                 o = 4)
# D <-
#   D %>%
#   mutate(f = exp(-((e-g)^2)/(2*o^2)))
# 
# D %>%
#   ggplot(aes(e, f)) +
#   geom_line() +
#   geom_vline(aes(xintercept = -1)) +
#   geom_vline(aes(xintercept = 1)) +
#   geom_vline(aes(xintercept = -2)) +
#   geom_vline(aes(xintercept = 2))

# bm <- c(1,10,100)
# m <- 0.1
# mPow <- 0.25
# 
# m*(bm^mPow)

