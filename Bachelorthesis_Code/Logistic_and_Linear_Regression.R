setwd("/Users/romyl/OneDrive/Desktop/Bachelor Thesis")
df <- read.csv("processed_amazon_data.csv")

library(dplyr)
library(car)
library(broom)
library(ggplot2)
library(knitr)
library(kableExtra)


threshold <- median(df$rating_count, na.rm = TRUE)
df <- df %>% mutate(rating_category = ifelse(rating_count > threshold, "High", "Low"))
df$rating_category <- factor(df$rating_category, levels = c("Low", "High"))


logit_model_weighted <- glm(
  formula = rating_category ~  content_sentiment_score + 
    title_sentiment_score + weighted_rating + discounted_price + discount_percentage + actual_price,
  family = binomial,
  data = df
)

summary(logit_model_weighted)


logit_model_rating <- glm(
  formula = rating_category ~  content_sentiment_score + 
    title_sentiment_score + rating + discounted_price + discount_percentage + actual_price,
  family = binomial,
  data = df
)

summary(logit_model_rating)

lm_model_weighted <- lm(weighted_rating ~ content_sentiment_score + title_sentiment_score + rating_count + discounted_price + discount_percentage + actual_price, data = df)

summary(lm_model_weighted)

lm_model_rating <- lm(rating ~ content_sentiment_score + title_sentiment_score + rating_count + discounted_price + discount_percentage + actual_price, data = df)

summary(lm_model_rating)


logit_table_weighted <- tidy(logit_model_weighted) %>%
  mutate(
    Significance = case_when(
      p.value < 0.001 ~ "***",
      p.value < 0.01 ~ "**",
      p.value < 0.05 ~ "*",
      TRUE ~ ""
    )
  ) %>%
  dplyr::rename(
    Term = term,
    Estimate = estimate,
    `Std. Error` = std.error,
    Statistic = statistic,
    `P-value` = p.value
  )

logit_table_rating <- tidy(logit_model_rating) %>%
  mutate(
    Significance = case_when(
      p.value < 0.001 ~ "***",
      p.value < 0.01 ~ "**",
      p.value < 0.05 ~ "*",
      TRUE ~ ""
    )
  ) %>%
  dplyr::rename(
    Term = term,
    Estimate = estimate,
    `Std. Error` = std.error,
    Statistic = statistic,
    `P-value` = p.value
  )


lm_table_weighted <- tidy(lm_model_weighted) %>%
  mutate(
    Significance = case_when(
      p.value < 0.001 ~ "***",
      p.value < 0.01 ~ "**",
      p.value < 0.05 ~ "*",
      TRUE ~ ""
    )
  ) %>%
  dplyr::rename(
    Term = term,
    Estimate = estimate,
    `Std. Error` = std.error,
    Statistic = statistic,
    `P-value` = p.value
  )

lm_table_rating <- tidy(lm_model_rating) %>%
  mutate(
    Significance = case_when(
      p.value < 0.001 ~ "***",
      p.value < 0.01 ~ "**",
      p.value < 0.05 ~ "*",
      TRUE ~ ""
    )
  ) %>%
  dplyr::rename(
    Term = term,
    Estimate = estimate,
    `Std. Error` = std.error,
    Statistic = statistic,
    `P-value` = p.value
  )

logit_weighted_latex <- kable(
  logit_table_weighted,
  format = "latex",
  booktabs = TRUE,
  caption = "Coefficients and Significance for the Logistic Regression Model"
) %>%
  kable_styling(latex_options = c("hold_position"))

logit_rating_latex <- kable(
  logit_table_rating,
  format = "latex",
  booktabs = TRUE,
  caption = "Coefficients and Significance for the Logistic Regression Model"
) %>%
  kable_styling(latex_options = c("hold_position"))

lm_weighted_latex <- kable(
  lm_table_weighted,
  format = "latex",
  booktabs = TRUE,
  caption = "Coefficients and Significance for the Linear Regression Model"
) %>%
  kable_styling(latex_options = c("hold_position"))

lm_rating_latex <- kable(
  lm_table_rating,
  format = "latex",
  booktabs = TRUE,
  caption = "Coefficients and Significance for the Linear Regression Model"
) %>%
  kable_styling(latex_options = c("hold_position"))
cat(logit_weighted_latex)
cat(logit_rating_latex)

cat(lm_weighted_latex)
cat(lm_rating_latex)
