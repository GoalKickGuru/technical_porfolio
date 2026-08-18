# =============================================================================
# FetchMaker Dog Data Analysis — Comprehensive Statistical Report (R Version)
# Version 2.0 | Generated: 2026-07-25
# =============================================================================
#
# REQUIRED R PACKAGES
# -----------------------------------------------------------------------------
#   install.packages(c(
#     "tidyverse",      # data manipulation + ggplot2 visualization
#     "rstatix",        # tidy statistical tests (pipe-friendly)
#     "broom",          # tidy model outputs
#     "multcomp",       # Tukey HSD post-hoc tests
#     " DescTools",      # Cramer's V effect size
#     "randomForest",   # machine learning — breed prediction
#     "caret",          # ML workflow (train/test split, evaluation)
#     "patchwork",      # multi-panel plot assembly
#     "knitr",          # formatted table output
#     " DescTools"       # Cramer's V
#   ))
#
# DATA REQUIREMENTS
# -----------------------------------------------------------------------------
#   Place 'dog_data.csv' in the working directory:
#   setwd("path/to/your/data")
#
# ANALYSIS SECTIONS
# -----------------------------------------------------------------------------
#   Part 1: Data Loading & Exploratory Inspection
#   Part 2: Rescue Rate Analysis (All Breeds vs. 8% Baseline)
#   Part 3: Weight Distribution Analysis & Visualization
#   Part 4: Comprehensive ANOVA & Post-Hoc Testing
#   Part 5: Color Association Analysis (Chi-Square & Cramer's V)
#   Part 6: Extended Analyses (Age, Hypoallergenic, Child-Friendly)
#   Part 7: Machine Learning — Breed Prediction Model
#   Part 8: Executive Summary Dashboard
#   Part 9: Next Steps Recommendations
#
# =============================================================================

# =============================================================================
# SETUP: LOAD LIBRARIES
# =============================================================================
suppressPackageStartupMessages({
  library(tidyverse)
  library(rstatix)
  library(broom)
  library(multcomp)
  library(DescTools)
  library(randomForest)
  library(caret)
  library(patchwork)
  library(knitr)
})

# Helper: formatted section headers
print_section <- function(title, subtitle = "") {
  border <- strrep("=", 72)
  cat("\n", border, "\n", sep = "")
  cat("  ", title, "\n", sep = "")
  if (subtitle != "") cat("  ", subtitle, "\n", sep = "")
  cat(border, "\n")
}

print_subsection <- function(title) {
  line <- strrep("-", 72)
  cat("\n", line, "\n", sep = "")
  cat("  ", title, "\n", sep = "")
  cat(line, "\n")
}

cat("=========================================================================\n")
cat("  FETCHMAKER DOG DATA ANALYSIS — R VERSION 2.0\n")
cat("=========================================================================\n")

# =============================================================================
# PART 1: DATA LOADING & EXPLORATORY INSPECTION
# =============================================================================
print_section(
  "PART 1: DATA LOADING & EXPLORATORY INSPECTION",
  "read_csv with column type optimization"
)

# R optimizes memory via col_types specification (equivalent to Python dtypes)
col_types <- cols(
  is_rescue         = col_integer(),
  weight           = col_integer(),
  tail_length      = col_double(),
  age              = col_integer(),
  color            = col_character(),
  likes_children   = col_integer(),
  is_hypoallergenic = col_integer(),
  name             = col_character(),
  breed            = col_character()
)

dogs <- read_csv("dog_data.csv", col_types = col_types, show_col_types = FALSE)

cat(sprintf("\n  Total records loaded : %s", format(nrow(dogs), big.mark = ",")))
cat(sprintf("\n  Breeds detected      : %d", n_distinct(dogs$breed)))
cat(sprintf("\n  Breed list           : %s\n",
            paste(sort(unique(dogs$breed)), collapse = ", ")))

cat("\n  Column data types:\n")
for (col in names(dogs)) {
  cat(sprintf("    %-22s %s\n", col, class(dogs[[col]])[1]))
}

# Missing values
missing_vals <- sapply(dogs, function(x) sum(is.na(x)))
if (any(missing_vals > 0)) {
  cat("\n  Missing values detected:\n")
  for (col in names(missing_vals[missing_vals > 0])) {
    cat(sprintf("    %-22s %d missing\n", col, missing_vals[col]))
  }
} else {
  cat("\n  Missing values       : None detected\n")
}

# Breed frequency table
breed_counts <- dogs %>%
  count(breed, sort = TRUE)

cat("\n  Breed Frequency Distribution:\n")
cat(sprintf("  %-15s %6s  %10s\n", "Breed", "Count", "Percentage"))
cat(sprintf("  %s %s  %s\n", strrep("-", 15), strrep("-", 6), strrep("-", 10)))
for (i in seq_len(nrow(breed_counts))) {
  pct <- breed_counts$n[i] / nrow(dogs) * 100
  cat(sprintf("  %-15s %6d  %9.1f%%\n",
              breed_counts$breed[i], breed_counts$n[i], pct))
}

# =============================================================================
# PART 2: RESCUE RATE ANALYSIS (ALL BREEDS VS. 8% BASELINE)
# =============================================================================
print_section(
  "PART 2: RESCUE RATE ANALYSIS",
  "Binomial test: Is each breed's rescue rate significantly different from 8%?"
)

POPULATION_RESCUE_RATE <- 0.08

# Tidy approach: group + summarize + map binomial test
rescue_analysis <- dogs %>%
  group_by(breed) %>%
  summarise(
    total_dogs         = n(),
    num_rescues        = sum(is_rescue == 1),
    rescue_rate        = num_rescues / total_dogs,
    diff_from_expected = rescue_rate - POPULATION_RESCUE_RATE,
    p_value = map2_dbl(
      num_rescues, total_dogs,
      ~ binom.test(.x, .y, p = POPULATION_RESCUE_RATE,
                   alternative = "two.sided")$p.value
    ),
    significant = p_value < 0.05
  ) %>%
  arrange(desc(rescue_rate))

cat(sprintf("\n  Baseline rescue rate : %.0f%%", POPULATION_RESCUE_RATE * 100))
cat(sprintf("\n  Significance level   : alpha = 0.05 (two-sided)\n"))
cat(sprintf("\n  %-15s %5s %7s %7s %8s %10s %6s\n",
            "Breed", "Total", "Rescues", "Rate", "Diff", "p-value", "Sig.?"))
cat(sprintf("  %s %s %s %s %s %s %s\n",
            strrep("-", 15), strrep("-", 5), strrep("-", 7),
            strrep("-", 7), strrep("-", 8), strrep("-", 10), strrep("-", 6)))
for (i in seq_len(nrow(rescue_analysis))) {
  row <- rescue_analysis[i, ]
  sig <- ifelse(row$significant, "Yes *", "No")
  cat(sprintf("  %-15s %5d %7d %6.1f%% %+7.1f%% %10.4f %6s\n",
              row$breed, row$total_dogs, row$num_rescues,
              row$rescue_rate * 100, row$diff_from_expected * 100,
              row$p_value, sig))
}
cat("\n  * Significant at alpha = 0.05\n")

# =============================================================================
# PART 3: WEIGHT DISTRIBUTION ANALYSIS & VISUALIZATION
# =============================================================================
print_section(
  "PART 3: WEIGHT DISTRIBUTION ANALYSIS",
  "Tidyverse group_by + summarise (vectorized, fast)"
)

weight_stats <- dogs %>%
  group_by(breed) %>%
  summarise(
    N      = n(),
    Mean   = round(mean(weight), 2),
    Std    = round(sd(weight), 2),
    Min    = min(weight),
    Median = round(median(weight), 2),
    Max    = max(weight)
  ) %>%
  arrange(desc(Mean))

cat("\n  Weight Statistics by Breed (lbs):\n")
cat(sprintf("  %-15s %5s %7s %7s %5s %7s %5s\n",
            "Breed", "N", "Mean", "Std", "Min", "Median", "Max"))
cat(sprintf("  %s %s %s %s %s %s %s\n",
            strrep("-", 15), strrep("-", 5), strrep("-", 7),
            strrep("-", 7), strrep("-", 5), strrep("-", 7), strrep("-", 5)))
for (i in seq_len(nrow(weight_stats))) {
  row <- weight_stats[i, ]
  cat(sprintf("  %-15s %5d %7.1f %7.1f %5d %7.1f %5d\n",
              row$breed, row$N, row$Mean, row$Std,
              row$Min, row$Median, row$Max))
}

# --- Visualization with ggplot2 + patchwork ---

# Panel 1: Boxplot — all breeds
p1 <- ggplot(dogs, aes(x = reorder(breed, weight, median), y = weight)) +
  geom_boxplot(fill = "#6d4aff", alpha = 0.5) +
  geom_hline(yintercept = mean(dogs$weight),
             color = "red", linetype = "dashed",
             linewidth = 0.8) +
  annotate("text", x = 1, y = mean(dogs$weight) + 3,
           label = sprintf("Mean: %.1f lbs", mean(dogs$weight)),
           color = "red", size = 3, hjust = 0) +
  coord_flip() +
  labs(title = "Boxplot: Weight by Breed",
       x = NULL, y = "Weight (lbs)") +
  theme_bw(base_size = 11) +
  theme(plot.title = element_text(face = "bold"))

# Panel 2: Boxplot — mid-sized breeds
mid_breeds <- c("whippet", "terrier", "pitbull")
dogs_mid <- dogs %>% filter(breed %in% mid_breeds)

p2 <- ggplot(dogs_mid, aes(x = breed, y = weight, fill = breed)) +
  geom_boxplot(alpha = 0.6) +
  scale_fill_manual(values = c("whippet" = "#6d4aff",
                               "terrier" = "#ff6d4a",
                               "pitbull" = "#4a6dff")) +
  labs(title = "Mid-Sized: Whippet vs Terrier vs Pitbull",
       x = NULL, y = "Weight (lbs)") +
  theme_bw(base_size = 11) +
  theme(plot.title = element_text(face = "bold"),
        legend.position = "none")

# Panel 3: Histogram overlay — mid-sized breeds
p3 <- ggplot(dogs_mid, aes(x = weight, fill = breed)) +
  geom_histogram(alpha = 0.5, position = "identity",
                 bins = 15, aes(y = after_stat(density))) +
  scale_fill_manual(values = c("whippet" = "#6d4aff",
                               "terrier" = "#ff6d4a",
                               "pitbull" = "#4a6dff")) +
  geom_vline(xintercept = mean(dogs_mid$weight),
             color = "gray40", linetype = "dotted") +
  labs(title = "Weight Density Comparison (Mid-Sized)",
       x = "Weight (lbs)", y = "Density") +
  theme_bw(base_size = 11) +
  theme(plot.title = element_text(face = "bold"))

# Panel 4: Correlation matrix
numeric_cols <- dogs %>%
  select(weight, tail_length, age) %>%
  cor()

corr_df <- as.data.frame(as.table(numeric_cols))
colnames(corr_df) <- c("Var1", "Var2", "Correlation")

p4 <- ggplot(corr_df, aes(x = Var1, y = Var2, fill = Correlation)) +
  geom_tile() +
  geom_text(aes(label = sprintf("%.2f", Correlation)), size = 4) +
  scale_fill_gradient2(low = "#FFFFFF", mid = "#90A4EF",
                        high = "#6d4aff", midpoint = 0.5) +
  labs(title = "Correlation Matrix",
       x = NULL, y = NULL) +
  theme_bw(base_size = 11) +
  theme(plot.title = element_text(face = "bold"),
        legend.position = "none")

# Combine with patchwork
combined_plot <- (p1 | p2) / (p3 | p4) +
  plot_annotation(
    title = "FetchMaker: Weight Distribution Across All Breeds",
    theme = theme(plot.title = element_text(size = 16, face = "bold"))
  )

ggsave("fetchmaker_weight_analysis.png", combined_plot,
       width = 14, height = 10, dpi = 150, bg = "white")
cat("\n  [Saved] fetchmaker_weight_analysis.png\n")

# =============================================================================
# PART 4: COMPREHENSIVE ANOVA & POST-HOC TESTING
# =============================================================================
print_section(
  "PART 4: COMPREHENSIVE ANOVA & POST-HOC TESTING",
  "Assumption checks -> ANOVA -> Kruskal-Wallis -> Tukey HSD"
)

dogs_mid_weight <- dogs %>% filter(breed %in% mid_breeds)

# --- Assumption: Normality (Shapiro-Wilk) ---
print_subsection("Assumption Check: Normality (Shapiro-Wilk)")
shapiro_results <- dogs_mid_weight %>%
  group_by(breed) %>%
  summarise(
    statistic = shapiro.test(weight)$statistic,
    p_value   = shapiro.test(weight)$p.value
  )
for (i in seq_len(nrow(shapiro_results))) {
  row <- shapiro_results[i, ]
  verdict <- ifelse(row$p_value > 0.05, "PASS (normal)", "FAIL (non-normal)")
  cat(sprintf("    %-12s W=%.3f  p=%.3f  -> %s\n",
              capitalize(row$breed), row$statistic, row$p_value, verdict))
}

# --- Assumption: Homogeneity of Variance (Levene's) ---
print_subsection("Assumption Check: Homogeneity of Variance (Levene's)")
levene_result <- leveneTest(weight ~ breed, data = dogs_mid_weight)
levene_p <- levene_result$`Pr(>F)`[1]
verdict <- ifelse(levene_p > 0.05, "PASS (equal)", "FAIL (unequal)")
cat(sprintf("    Statistic=%.3f  p=%.3f  -> %s\n",
            levene_result$`F value`[1], levene_p, verdict))

# --- Parametric: One-way ANOVA ---
print_subsection("Test 1: One-way ANOVA (Parametric)")
aov_result <- aov(weight ~ breed, data = dogs_mid_weight)
aov_summary <- summary(aov_result)[[1]]
F_stat <- aov_summary$`F value`[1]
p_anova <- aov_summary$`Pr(>F)`[1]
verdict <- ifelse(p_anova < 0.05, "SIGNIFICANT", "NOT SIGNIFICANT")
cat(sprintf("    F-statistic : %.3f\n", F_stat))
cat(sprintf("    p-value     : %.6f\n", p_anova))
cat(sprintf("    Result      : %s (alpha=0.05)\n", verdict))

# --- Non-parametric: Kruskal-Wallis ---
print_subsection("Test 2: Kruskal-Wallis (Non-parametric)")
kw_result <- kruskal.test(weight ~ breed, data = dogs_mid_weight)
verdict <- ifelse(kw_result$p.value < 0.05, "SIGNIFICANT", "NOT SIGNIFICANT")
cat(sprintf("    H-statistic : %.3f\n", kw_result$statistic))
cat(sprintf("    p-value     : %.6f\n", kw_result$p.value))
cat(sprintf("    Result      : %s (alpha=0.05)\n", verdict))

# --- Post-hoc: Tukey HSD ---
print_subsection("Post-hoc: Tukey HSD (Pairwise Comparisons)")
tukey_result <- TukeyHSD(aov_result)
tukey_df <- as.data.frame(tukey_result$breed)
tukey_df$comparison <- rownames(tukey_df)
tukey_df$sig <- ifelse(tukey_df$`p adj` < 0.05, "*", "")
cat(sprintf("    %-25s %8s %8s %8s %8s %5s\n",
            "Comparison", "Diff", "Lower", "Upper", "p adj", "Sig"))
cat(sprintf("    %s %s %s %s %s %s\n",
            strrep("-", 25), strrep("-", 8), strrep("-", 8),
            strrep("-", 8), strrep("-", 8), strrep("-", 5)))
for (i in seq_len(nrow(tukey_df))) {
  row <- tukey_df[i, ]
  cat(sprintf("    %-25s %8.2f %8.2f %8.2f %8.4f %5s\n",
              row$comparison, row$diff, row$lwr,
              row$upr, row$`p adj`, row$sig))
}

# =============================================================================
# PART 5: COLOR ASSOCIATION ANALYSIS (CHI-SQUARE & CRAMER'S V)
# =============================================================================
print_section(
  "PART 5: COLOR ASSOCIATION ANALYSIS",
  "Chi-square test + Cramer's V effect size"
)

analyze_color_breed <- function(df, target_breeds = NULL, label = "") {
  if (!is.null(target_breeds)) {
    df_subset <- df %>% filter(breed %in% target_breeds)
  } else {
    df_subset <- df
  }

  contingency <- table(df_subset$color, df_subset$breed)
  chi2_result <- chisq.test(contingency)
  cv <- CramerV(contingency)

  if (cv > 0.3) {
    strength <- "Strong"
  } else if (cv > 0.1) {
    strength <- "Moderate"
  } else {
    strength <- "Weak"
  }

  cat(sprintf("\n  Analysis scope     : %s\n", label))
  cat(sprintf("  Contingency shape  : %d x %d\n", nrow(contingency), ncol(contingency)))
  cat(sprintf("  Chi-square stat    : %.3f\n", chi2_result$statistic))
  cat(sprintf("  Degrees of freedom : %d\n", chi2_result$parameter))
  cat(sprintf("  P-value            : %.6f\n", chi2_result$p.value))
  cat(sprintf("  Cramer's V         : %.3f (%s association)\n", cv, strength))
  conclusion <- ifelse(chi2_result$p.value < 0.05,
                       "SIGNIFICANT — color differs by breed",
                       "NOT significant — no color-breed association")
  cat(sprintf("  Conclusion         : %s (alpha=0.05)\n", conclusion))

  return(list(
    table = contingency,
    chi2  = chi2_result$statistic,
    p_val = chi2_result$p.value,
    cramer_v = cv
  ))
}

print_subsection("5a: Original Analysis (Poodle vs. Shihtzu)")
result_ps <- analyze_color_breed(dogs, c("poodle", "shihtzu"), "Poodle vs. Shihtzu")

print_subsection("5b: Extended Analysis (All Breeds)")
result_all <- analyze_color_breed(dogs, label = "All 8 breeds")

print_subsection("5c: Color Distribution Within Each Breed (%)")
color_pct <- prop.table(result_all$table, margin = 2) * 100
color_pct_df <- as.data.frame(round(color_pct, 1))
colnames(color_pct_df) <- c("Color", "Breed", "Percent")
color_pct_wide <- color_pct_df %>%
  pivot_wider(names_from = Breed, values_from = Percent)
print(color_pct_wide, n = Inf)

# Store key results for executive summary
p_all <- result_all$p_val
chi2_all <- result_all$chi2

# =============================================================================
# PART 6: EXTENDED ANALYSES
# =============================================================================

# ---- 6A: Age Distribution Analysis ----
print_section(
  "PART 6A: AGE DISTRIBUTION ANALYSIS",
  "Kruskal-Wallis test + ggplot2 visualization"
)

p_age_hist <- ggplot(dogs, aes(x = age, fill = breed)) +
  geom_histogram(position = "identity", alpha = 0.5, bins = 14) +
  labs(title = "Age Distribution by Breed",
       x = "Age (years)", y = "Count") +
  theme_bw(base_size = 11) +
  theme(plot.title = element_text(face = "bold"),
        legend.position = "bottom")

p_age_box <- ggplot(dogs, aes(x = reorder(breed, age, median), y = age, fill = breed)) +
  geom_boxplot(alpha = 0.6) +
  coord_flip() +
  labs(title = "Age Statistics by Breed",
       x = NULL, y = "Age (years)") +
  theme_bw(base_size = 11) +
  theme(plot.title = element_text(face = "bold"),
        legend.position = "none")

age_plot <- p_age_hist + p_age_box +
  plot_annotation(title = "FetchMaker: Age Distribution Analysis",
                   theme = theme(plot.title = element_text(size = 14, face = "bold")))

ggsave("fetchmaker_age_analysis.png", age_plot,
       width = 13, height = 5, dpi = 150, bg = "white")
cat("\n  [Saved] fetchmaker_age_analysis.png\n")

# Kruskal-Wallis test across all breeds
kw_age_result <- kruskal.test(age ~ breed, data = dogs)
cat(sprintf("\n  Kruskal-Wallis H-statistic : %.3f\n", kw_age_result$statistic))
cat(sprintf("  p-value                    : %.6f\n", kw_age_result$p.value))
verdict <- ifelse(kw_age_result$p.value < 0.05,
                  "Different age distributions across breeds",
                  "Similar age distributions across breeds")
cat(sprintf("  Result                     : %s\n", verdict))

# ---- 6B: Hypoallergenic Analysis ----
print_section(
  "PART 6B: HYPOALLERGENIC ANALYSIS",
  "Which breeds are most hypoallergenic? + rescue association test"
)

hypo_stats <- dogs %>%
  group_by(breed) %>%
  summarise(
    total     = n(),
    hypo_count = sum(is_hypoallergenic == 1),
    hypo_pct   = round(hypo_count / total * 100, 1)
  ) %>%
  arrange(desc(hypo_pct))

cat(sprintf("\n  %-15s %5s %6s %8s\n", "Breed", "Total", "Hypo.", "Percent"))
cat(sprintf("  %s %s %s %s\n",
            strrep("-", 15), strrep("-", 5), strrep("-", 6), strrep("-", 8)))
for (i in seq_len(nrow(hypo_stats))) {
  row <- hypo_stats[i, ]
  cat(sprintf("  %-15s %5d %6d %7.1f%%\n",
              row$breed, row$total, row$hypo_count, row$hypo_pct))
}

# Chi-square: hypoallergenic vs rescue
ct_hyp_rescue <- table(dogs$is_hypoallergenic, dogs$is_rescue)
chi2_hyp_result <- chisq.test(ct_hyp_rescue)
cat(sprintf("\n  Hypoallergenic vs Rescue Association:\n"))
cat(sprintf("    Chi-square : %.3f\n", chi2_hyp_result$statistic))
cat(sprintf("    p-value    : %.4f\n", chi2_hyp_result$p.value))
verdict <- ifelse(chi2_hyp_result$p.value < 0.05,
                 "ASSOCIATION EXISTS", "NO association")
cat(sprintf("    Result     : %s (alpha=0.05)\n", verdict))

# ---- 6C: Child-Friendly Analysis ----
print_section(
  "PART 6C: CHILD-FRIENDLINESS ANALYSIS",
  "Proportion of dogs that like children, by breed"
)

cf_stats <- dogs %>%
  group_by(breed) %>%
  summarise(score = mean(likes_children == 1)) %>%
  arrange(desc(score))

cat(sprintf("\n  %-15s %20s\n", "Breed", "Child-Friendly Score"))
cat(sprintf("  %s %s\n", strrep("-", 15), strrep("-", 20)))
for (i in seq_len(nrow(cf_stats))) {
  row <- cf_stats[i, ]
  bar <- strrep("#", as.integer(row$score * 20))
  cat(sprintf("  %-15s %10.3f  %s\n", row$breed, row$score, bar))
}

# =============================================================================
# PART 7: MACHINE LEARNING — BREED PREDICTION MODEL
# =============================================================================
print_section(
  "PART 7: MACHINE LEARNING — BREED PREDICTION",
  "Random Forest classifier: Can physical traits predict breed?"
)

# Encode color as factor (R handles factors natively)
dogs$color_factor <- as.factor(dogs$color)

# Prepare features
ml_data <- dogs %>%
  select(weight, tail_length, age, color_factor, breed) %>%
  mutate(breed = as.factor(breed))

# 80/20 train-test split with stratified sampling
set.seed(42)
train_indices <- createDataPartition(ml_data$breed, p = 0.8, list = FALSE)
train_data <- ml_data[train_indices, ]
test_data  <- ml_data[-train_indices, ]

# Train Random Forest
set.seed(42)
rf_model <- randomForest(
  breed ~ .,
  data = train_data,
  ntree = 100,
  importance = TRUE
)

# Evaluate
predictions <- predict(rf_model, newdata = test_data)
accuracy <- mean(predictions == test_data$breed)

cat(sprintf("\n  Training samples : %d", nrow(train_data)))
cat(sprintf("\n  Testing samples  : %d", nrow(test_data)))
cat(sprintf("\n  Model accuracy    : %.2f%%\n", accuracy * 100))

# Feature importance
imp <- importance(rf_model)
imp_df <- data.frame(
  feature = rownames(imp),
  importance = imp[, "MeanDecreaseGini"]
) %>%
  arrange(desc(importance))

cat("\n  Feature Importance (higher = more predictive):\n")
cat(sprintf("  %-18s %10s %20s\n", "Feature", "Importance", "Bar"))
cat(sprintf("  %s %s %s\n", strrep("-", 18), strrep("-", 10), strrep("-", 20)))
for (i in seq_len(nrow(imp_df))) {
  row <- imp_df[i, ]
  bar <- strrep("#", as.integer(row$importance / max(imp_df$importance) * 40))
  cat(sprintf("  %-18s %10.4f %20s\n", row$feature, row$importance, bar))
}

# Confusion matrix
cat("\n  Confusion Matrix & Classification Report:\n")
cm <- confusionMatrix(predictions, test_data$breed)
cat(sprintf("    Overall accuracy : %.2f%%\n", cm$overall["Accuracy"] * 100))
cat(sprintf("\n  Per-class breakdown:\n"))
cat(sprintf("    %-15s %8s %8s %8s\n", "Class", "Precision", "Recall", "F1"))
cat(sprintf("    %s %s %s %s\n", strrep("-", 15), strrep("-", 8),
            strrep("-", 8), strrep("-", 8)))
by_class <- cm$byClass
for (cls in rownames(by_class)) {
  breed_name <- gsub("^Class: ", "", cls)
  cat(sprintf("    %-15s %8.3f %8.3f %8.3f\n",
              breed_name,
              by_class[cls, "Precision"],
              by_class[cls, "Recall"],
              by_class[cls, "F1"]))
}

# =============================================================================
# PART 8: EXECUTIVE SUMMARY DASHBOARD
# =============================================================================
print_section(
  "PART 8: EXECUTIVE SUMMARY",
  "Key findings for stakeholder presentation"
)

findings <- list(
  list("Dataset Size", sprintf("%s dogs across %d breeds",
        format(nrow(dogs), big.mark = ","), n_distinct(dogs$breed))),
  list("Most Common Breed", sprintf("%s (%d dogs)",
        capitalize(breed_counts$breed[1]), breed_counts$n[1])),
  list("Average Weight", sprintf("%.1f lbs (+/- %.1f)",
        mean(dogs$weight), sd(dogs$weight))),
  list("Overall Rescue Rate", sprintf("%.1f%% (vs 8%% baseline)",
        mean(dogs$is_rescue == 1) * 100)),
  list("Hypoallergenic Dogs", sprintf("%.1f%% of all dogs",
        mean(dogs$is_hypoallergenic == 1) * 100)),
  list("Child-Friendly Dogs", sprintf("%.1f%% like children",
        mean(dogs$likes_children == 1) * 100)),
  list("Color-Breed Link",
       sprintf("%s (chi2=%.1f, p=%.4f)",
               ifelse(p_all < 0.05, "SIGNIFICANT", "Not significant"),
               chi2_all, p_all)),
  list("Weight Differences",
       sprintf("%s (ANOVA p=%.4f)",
               ifelse(p_anova < 0.05, "SIGNIFICANT", "Not significant"),
               p_anova)),
  list("ML Breed Prediction",
       sprintf("Random Forest accuracy: %.1f%%", accuracy * 100))
)

cat("\n")
for (f in findings) {
  cat(sprintf("  - %-24s %s\n", f[[1]], f[[2]))
}

# =============================================================================
# PART 9: NEXT STEPS RECOMMENDATIONS
# =============================================================================
print_section(
  "PART 9: NEXT STEPS RECOMMENDATIONS",
  "Actionable items for FetchMaker leadership"
)

recommendations <- list(
  list("HIGH", "Collect More Data",
       "Ensure balanced representation across all 8 breeds for more reliable conclusions."),
  list("HIGH", "Track Adoption Outcomes",
       "Correlate physical traits with successful matches and time-to-adoption metrics."),
  list("MEDIUM", "A/B Testing",
       "Compare algorithm-based recommendations vs. traditional search methods."),
  list("MEDIUM", "Customer Surveys",
       "Gather owner satisfaction data post-adoption to refine matching criteria."),
  list("LOW", "Seasonal Analysis",
       "Track adoption patterns throughout the year to optimize inventory and marketing.")
)

cat("\n")
for (r in recommendations) {
  cat(sprintf("  [%s] %s\n", r[[1]], r[[2]]))
  cat(sprintf("        %s\n\n", r[[3]]))
}

# =============================================================================
# SCRIPT COMPLETION
# =============================================================================
cat("=========================================================================\n")
cat("  ANALYSIS COMPLETE\n")
cat("=========================================================================\n")
cat("
  Generated Files:
    1. fetchmaker_weight_analysis.png - Weight distributions & correlations
    2. fetchmaker_age_analysis.png  - Age distributions across breeds

  Console Output:
    - Rescue rate binomial tests (all breeds)
    - ANOVA + Tukey HSD weight comparison
    - Chi-square color-breed association
    - Age distribution (Kruskal-Wallis)
    - Hypoallergenic & child-friendliness analysis
    - Random Forest breed prediction model
    - Executive summary dashboard
")
cat("=========================================================================\n")


# =============================================================================
# APPENDIX: PYTHON vs R COMPARISON TABLE
# =============================================================================
# This section prints a comparison table to the console for reference.
# It covers what each language excels at for THIS analysis and general DS work.
# =============================================================================

print_section(
  "APPENDIX: PYTHON vs R COMPARISON",
  "Strengths and limitations for this analysis and general data science"
)

cat("
  ┌──────────────────────────┬─────────────────────────────┬─────────────────────────────┐
  │ Analysis / Task          │ Python                     │ R                           │
  ├──────────────────────────┼─────────────────────────────┼─────────────────────────────┤
  │                          │                             │                             │
  │ Data Loading             │ pd.read_csv() + dtype spec  │ read_csv() + col_types spec │
  │                          │ Flexible, fast on large     │ Equally fast, type-safe     │
  │                          │ datasets                    │ col_* specification         │
  ├──────────────────────────┼─────────────────────────────┼─────────────────────────────┤
  │                          │                             │                             │
  │ Data Manipulation        │ pandas groupby().agg()      │ dplyr: group_by() %>%       │
  │                          │ Verbose syntax,             │ summarise() — pipe operator │
  │                          │ flexible chaining           │ is highly readable           │
  ├──────────────────────────┼─────────────────────────────┼─────────────────────────────┤
  │                          │                             │                             │
  │ Binomial Test            │ scipy.stats.binomtest()     │ binom.test() built-in to    │
  │                          │ Requires SciPy import       │ base R — no package needed  │
  ├──────────────────────────┼─────────────────────────────┼─────────────────────────────┤
  │                          │                             │                             │
  │ ANOVA + Tukey HSD        │ scipy.f_oneway() +          │ aov() + TukeyHSD() — both   │
  │                          │ statsmodels.pairwise_tukey-  │ in base R, seamless formula│
  │                          │ hsd() (separate packages)   │ syntax: weight ~ breed      │
  ├──────────────────────────┼─────────────────────────────┼─────────────────────────────┤
  │                          │                             │                             │
  │ Chi-Square + Cramer's V  │ scipy.chi2_contingency()    │ chisq.test() + DescTools::  │
  │                          │ + manual Cramer's V calc    │ CramerV() — dedicated fn    │
  ├──────────────────────────┼─────────────────────────────┼─────────────────────────────┤
  │                          │                             │                             │
  │ Assumption Tests         │ scipy.stats.shapiro(),      │ shapiro.test(),             │
  │ (Shapiro, Levene)        │ scipy.stats.levene() —      │ car::leveneTest() —         │
  │                          │ separate calls needed       │ formula interface           │
  ├──────────────────────────┼─────────────────────────────┼─────────────────────────────┤
  │                          │                             │                             │
  │ Visualization            │ matplotlib + seaborn        │ ggplot2 + patchwork         │
  │                          │ More control over axes,    │ Grammar of Graphics is      │
  │                          │ but verbose syntax          │ elegant & layered           │
  ├──────────────────────────┼─────────────────────────────┼─────────────────────────────┤
  │                          │                             │                             │
  │ Machine Learning         │ scikit-learn: fit/predict   │ randomForest + caret        │
  │ (Random Forest)          │ Unified API across models,  │ Caret provides unified      │
  │                          │ richer ecosystem            │ train() but slower tuning   │
  ├──────────────────────────┼─────────────────────────────┼─────────────────────────────┤
  │                          │                             │                             │
  │ Classification Report    │ classification_report()    │ confusionMatrix() from      │
  │                          │ prints to stdout directly   │ caret — rich object w/      │
  │                          │                             │ byClass stats               │
  ├──────────────────────────┼─────────────────────────────┼─────────────────────────────┤
  │                          │                             │                             │
  │ Statistical Formula      │ No native formula syntax    │ y ~ x formula notation      │
  │ Syntax                   │ Must construct arrays       │ is native to R: aov(y~x)   │
  ├──────────────────────────┼─────────────────────────────┼─────────────────────────────┤
  │                          │                             │                             │
  │ Pipe / Chaining          │ pandas .pipe() (limited)   │ %>% (magrittr) is deeply    │
  │                          │ Method chaining: df.method() │ integrated, very readable   │
  ├──────────────────────────┼─────────────────────────────┼─────────────────────────────┤
  │                          │                             │                             │
  │ Deployment / Production  │ EXCELLENT: Flask, FastAPI,  │ LIMITED: plumber package    │
  │                          │ Docker, cloud-native        │ exists but less adopted     │
  ├──────────────────────────┼─────────────────────────────┼─────────────────────────────┤
  │                          │                             │                             │
  │ Deep Learning            │ EXCELLENT: PyTorch,         │ LIMITED: keras/tensorflow   │
  │                          │ TensorFlow, JAX             │ wrapper exists but R is not  │
  │                          │                             │ a primary DL language       │
  ├──────────────────────────┼─────────────────────────────┼─────────────────────────────┤
  │                          │                             │                             │
  │ Statistical Reporting    │ GOOD: statsmodels summary   │ EXCELLENT: knitr, RMarkdown,│
  │ & Documents              │ tables, but plain text       │ Quarto — publish-ready PDFs,│
  │                          │                             │ HTML, Word with embedded code│
  ├──────────────────────────┼─────────────────────────────┼─────────────────────────────┤
  │                          │                             │                             │
  │ Bayesian Statistics      │ GOOD: PyMC, pymc3           │ EXCELLENT: brms, rstanarm   │
  │                          │                             │ formula interface, Stan      │
  ├──────────────────────────┼─────────────────────────────┼─────────────────────────────┤
  │                          │                             │                             │
  │ Survival Analysis        │ LIMITED: lifelines package  │ EXCELLENT: survival package  │
  │                          │                             │ is the gold standard         │
  ├──────────────────────────┼─────────────────────────────┼─────────────────────────────┤
  │                          │                             │                             │
  │ Time Series             │ GOOD: statsmodels, Prophet  │ EXCELLENT: forecast, fable,  │
  │                          │                             │ tseries packages             │
  ├──────────────────────────┼─────────────────────────────┼─────────────────────────────┤
  │                          │                             │                             │
  │ Big Data / Spark         │ EXCELLENT: PySpark,          │ GOOD: sparklyr wraps Spark  │
  │                          │ Dask, native integration     │ but lags behind Python       │
  ├──────────────────────────┼─────────────────────────────┼─────────────────────────────┤
  │                          │                             │                             │
  │ Web Scraping / APIs     │ EXCELLENT: requests,        │ MODERATE: httr, rvest       │
  │                          │ BeautifulSoup, Scrapy       │ functional but smaller ecosys│
  ├──────────────────────────┼─────────────────────────────┼─────────────────────────────┤
  │                          │                             │                             │
  │ Interactive Dashboards   │ GOOD: Streamlit, Dash       │ EXCELLENT: Shiny is         │
  │                          │                             │ purpose-built for stats     │
  ├──────────────────────────┼─────────────────────────────┼─────────────────────────────┤
  │                          │                             │                             │
  │ Community / Job Market   │ Larger dev community,       │ Dominates in academia,      │
  │                          │ broader industry adoption   │ biostatistics, pharma       │
  ├──────────────────────────┼─────────────────────────────┼─────────────────────────────┤
  │                          │                             │                             │
  │ THIS ANALYSIS VERDICT    │ Good for ML + deployment    │ Superior for stats + viz    │
  │                          │ Equal for data wrangling    │ Native formula syntax wins  │
  └──────────────────────────┴─────────────────────────────┴─────────────────────────────┘

  SUMMARY:
  --------
  For THIS SPECIFIC analysis (hypothesis testing, ANOVA, chi-square, RF):
    - R has a slight EDGE due to native formula syntax (y ~ x), built-in
      statistical tests (binom.test, aov, TukeyHSD), and ggplot2 elegance.
    - Python is EQUAL for data manipulation and ML, and SUPERIOR if you plan
      to deploy this analysis as a web API or production pipeline.

  GENERAL GUIDANCE:
  -----------------
    - Choose R if: statistics, research, publishing reports, clinical trials,
      bioinformatics, or interactive statistical dashboards are the priority.
    - Choose Python if: production deployment, deep learning, big data,
      web scraping, MLOps, or software engineering integration is needed.
    - Many teams use BOTH: R for exploration/reporting, Python for production.
")