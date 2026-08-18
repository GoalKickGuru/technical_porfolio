# =============================================================================
# Commute Times RDD – Pure R Solution (original + extensions)
# Requires: ggplot2, dplyr, rdd
# =============================================================================

library(ggplot2)
library(dplyr)
library(rdd)

cat("============================================================\n")
cat("       COMMUTE TIMES – SHARP RDD SOLUTION (R)\n")
cat("============================================================\n\n")

# 1-2 Load & inspect
snow_df <- read.csv("/home/workdir/artifacts/snow.csv")
cat("N =", nrow(snow_df), "\n")
print(head(snow_df))
print(summary(snow_df))
print(table(snow_df$emergency))

# Sharp check
cat("\nSharp design?\n")
print(all((snow_df$snowfall >= 4) == (snow_df$emergency == "Emergency")))

# 3-5 Visuals
scatter_base <- ggplot(snow_df, aes(x = snowfall, y = minutes,
                                    color = emergency, shape = emergency)) +
  geom_point(alpha = 0.7) +
  theme_minimal()
scatter_cutpoint <- scatter_base + geom_vline(xintercept = 4, linetype = "dashed")
scatter_lines <- scatter_cutpoint +
  geom_smooth(aes(group = emergency), method = "lm", se = FALSE, linewidth = 1.2)
print(scatter_lines)

# Density
print(ggplot(snow_df, aes(x = snowfall)) +
        geom_histogram(aes(y = after_stat(density)), bins = 30, fill = "steelblue", alpha = 0.7) +
        geom_density() +
        geom_vline(xintercept = 4, colour = "red", linetype = "dashed") +
        labs(title = "Forcing-variable density") + theme_minimal())

# 6-11 IK bandwidth + estimate
snow_ik_bw <- IKbandwidth(X = snow_df$snowfall, Y = snow_df$minutes, cutpoint = 4)
cat("\nIK bandwidth =", snow_ik_bw, "\n")
scatter_bw <- scatter_cutpoint +
  geom_vline(xintercept = 4 + c(-snow_ik_bw, snow_ik_bw), colour = "grey40")
print(scatter_bw)

snow_rdd <- RDestimate(formula = minutes ~ snowfall,
                       cutpoint = 4, bw = snow_ik_bw, data = snow_df)
print(snow_rdd)
cat("Observations used:\n"); print(snow_rdd$obs)
cat("Standard errors:\n"); print(snow_rdd$se)

# Sensitivity
cat("\n=== Bandwidth sensitivity ===\n")
for (h in c(1, 1.5, 2, 2.5, 3)) {
  est <- RDestimate(minutes ~ snowfall, cutpoint = 4, bw = h, data = snow_df)
  cat(sprintf("h=%.1f  est=%.3f  se=%.3f\n", h, est$est[1], est$se[1]))
}

# Placebo
cat("\n=== Placebo cut = 2 ===\n")
print(RDestimate(minutes ~ snowfall, cutpoint = 2, data = snow_df))

cat("\nDone.\n")
