# Modular Framework for Large-Scale Micro-Mobility Data Analysis

library(dplyr)
library(ggplot2)
library(geosphere)

# 1. Data Ingestion / Generator Component
generate_synthetic_mobility_data <- function(n = 10000) {
  data.frame(
    tripduration = runif(n, 60, 5000),
    start.station.latitude = runif(n, 40.70, 40.80),
    start.station.longitude = runif(n, -74.02, -73.93),
    end.station.latitude = runif(n, 40.70, 40.80),
    end.station.longitude = runif(n, -73.93, -73.93),
    birth.year = sample(1950:2005, n, replace = TRUE),
    gender = sample(c(0, 1, 2), n, replace = TRUE)
  )
}

# 2. Pipeline Execution Component
process_mobility_data <- function(df, max_duration = 1800, max_age = 80) {
  df %>%
    filter(tripduration <= max_duration) %>%
    mutate(
      age = 2020 - birth.year,
      distance = distHaversine(
        cbind(start.station.longitude, start.station.latitude),
        cbind(end.station.longitude, end.station.latitude)
      ),
      speed = distance / tripduration
    ) %>%
    filter(age < max_age, gender %in% c(1, 2)) %>%
    mutate(gender = factor(gender, labels = c("Male", "Female")))
}

# 3. Execution Template
raw_data <- generate_synthetic_mobility_data(15000)
processed_data <- process_mobility_data(raw_data)

# Summary check
summary(processed_data$speed)