

library(tidyverse)

results_test <- read_delim("C:/Users/joost/IdeaProjects/evolving-foodweb/results_test.csv",
                           ";", escape_double = FALSE, trim_ws = TRUE)

results_test %>%
  mutate(species = ordered(species),
         patch = ordered(patch),
         bodymass = ordered(bodymass)) %>%
  ggplot(aes(time, N, color = species)) +
  geom_line() +
  scale_y_log10() +
  # scale_colour_viridis_d() +
  # scale_color_brewer(palette = "PuBuGn") +
  facet_grid(patch~bodymass)

results_test %>%
  mutate(species = ordered(species),
         patch = ordered(patch),
         bodymass = ordered(bodymass)) %>%
  group_by(time, species, bodymass) %>% 
  summarise(N = sum(N, na.rm = TRUE)) %>% 
  ungroup() %>% 
  ggplot(aes(time, N, color = species)) +
  geom_line() +
  scale_y_log10() +
  # scale_colour_viridis_d() +
  # scale_color_brewer(palette = "PuBuGn") +
  facet_wrap(~bodymass, ncol = 1)

results_test %>%
  mutate(species = ordered(species),
         patch = ordered(patch),
         bodymass = ordered(bodymass)) %>%
  group_by(time, species, bodymass) %>% 
  summarise(biomass = sum(biomass, na.rm = TRUE)) %>% 
  ungroup() %>% 
  ggplot(aes(time, biomass, color = species)) +
  geom_line() +
  scale_y_log10() +
  # scale_colour_viridis_d() +
  # scale_color_brewer(palette = "PuBuGn") +
  facet_wrap(~bodymass, ncol = 1)


# x_0 <- 0.1
# pow_x <- -0.25
# x_0*c(1,10,100,1000)^pow_x
# 1/(x_0*c(1,10,100,1000)^pow_x)


