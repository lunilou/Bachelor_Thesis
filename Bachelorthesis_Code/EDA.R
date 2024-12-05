setwd("/Users/romyl/OneDrive/Desktop/Bachelor Thesis")
df <- read.csv("amazon.csv")

library(stringr)
library(dplyr)
library(tidytext)
library(sentimentr)
library(ggplot2)


# Data Cleaning
special_chars_pattern <- "[!@#$%^&*(),.?\":{}|<>]"
df$rating[str_detect(df$rating, special_chars_pattern) & df$product_id == "B08L12N5H1"] <- 4.2

df[df == ""] <- NA
df <- df[!is.na(df$rating_count), ]

df$discounted_price <- as.numeric(gsub("[₹,]", "", df$discounted_price))
df$actual_price <- as.numeric(gsub("[₹,]", "", df$actual_price))

df$rating <- as.numeric(gsub("[^0-9.]", "", df$rating))
df$rating_count <- as.numeric(gsub("[^0-9]", "", df$rating_count))

category_split <- strsplit(df$category, "\\|")
category_df <- do.call(rbind, lapply(category_split, function(x) c(x[1], x[2])))
colnames(category_df) <- c("main_category", "sub_category")
df$main_category <- category_df[, 1]
df$sub_category <- category_df[, 2]
df$category <- NULL  

df <- df %>% mutate(id = row_number())

# Feature Engineering
df$rating_score <- with(df, ifelse(rating < 2.0, "Poor",
                                   ifelse(rating >= 2.0 & rating <= 2.9, "Below Average",
                                          ifelse(rating >= 3.0 & rating <= 3.9, "Average",
                                                 ifelse(rating >= 4.0 & rating <= 4.9, "Good", "Excellent")))))

min_rating_count <- mean(df$rating_count, na.rm = TRUE)
df$weighted_rating <- with(df, ((rating * rating_count) + (min_rating_count * mean(rating, na.rm = TRUE))) / (rating_count + min_rating_count))

# Target Encoding for `product_name` and `product_id`
product_name_encoding <- df %>%
  group_by(product_name) %>%
  summarize(product_name_avg_rating = mean(rating, na.rm = TRUE))

product_id_encoding <- df %>%
  group_by(product_id) %>%
  summarize(product_id_avg_rating = mean(rating, na.rm = TRUE))

df <- left_join(df, product_name_encoding, by = "product_name")
df <- left_join(df, product_id_encoding, by = "product_id")

df$discount_percentage <- as.numeric(gsub("%", "", df$discount_percentage)) / 100
summary(df$discount_percentage)

# Sentiment Analysis
library(dplyr)
library(sentimentr)
library(tidytext)

sentiment_scores_content <- sentiment(get_sentences(df$review_content)) %>%
  group_by(element_id) %>%
  summarize(content_sentiment_score = mean(sentiment, na.rm = TRUE)) %>%
  mutate(id = row_number())

sentiment_scores_title <- sentiment(get_sentences(df$review_title)) %>%
  group_by(element_id) %>%
  summarize(title_sentiment_score = mean(sentiment, na.rm = TRUE)) %>%
  mutate(id = row_number())

df <- df %>%
  left_join(sentiment_scores_content %>% select(id, content_sentiment_score), by = "id") %>%
  left_join(sentiment_scores_title %>% select(id, title_sentiment_score), by = "id") %>%
  mutate(
    
    combined_sentiment_score = rowMeans(cbind(content_sentiment_score, title_sentiment_score), na.rm = TRUE),
    
    # Normalizing content, title, and combined sentiment scores to a 0-1 range
    normalized_content_sentiment_score = (content_sentiment_score - min(content_sentiment_score, na.rm = TRUE)) / 
      (max(content_sentiment_score, na.rm = TRUE) - min(content_sentiment_score, na.rm = TRUE)),
    normalized_title_sentiment_score = (title_sentiment_score - min(title_sentiment_score, na.rm = TRUE)) / 
      (max(title_sentiment_score, na.rm = TRUE) - min(title_sentiment_score, na.rm = TRUE)),
    normalized_combined_sentiment_score = (combined_sentiment_score - min(combined_sentiment_score, na.rm = TRUE)) / 
      (max(combined_sentiment_score, na.rm = TRUE) - min(combined_sentiment_score, na.rm = TRUE)),
    
    # Scaling normalized scores to a 1-5 range
    scaled_content_sentiment_score = normalized_content_sentiment_score * 4 + 1,
    scaled_title_sentiment_score = normalized_title_sentiment_score * 4 + 1,
    scaled_combined_sentiment_score = normalized_combined_sentiment_score * 4 + 1
  ) %>%
  # 5 sentiment categories
  mutate(
    content_sentiment_category = case_when(
      scaled_content_sentiment_score < 2 ~ "Strong Negative",
      scaled_content_sentiment_score < 3 ~ "Moderately Negative",
      scaled_content_sentiment_score < 4 ~ "Neutral",
      scaled_content_sentiment_score < 5 ~ "Moderately Positive",
      scaled_content_sentiment_score == 5 ~ "Strong Positive"
    ),
    title_sentiment_category = case_when(
      scaled_title_sentiment_score < 2 ~ "Strong Negative",
      scaled_title_sentiment_score < 3 ~ "Moderately Negative",
      scaled_title_sentiment_score < 4 ~ "Neutral",
      scaled_title_sentiment_score < 5 ~ "Moderately Positive",
      scaled_title_sentiment_score == 5 ~ "Strong Positive"
    ),
    combined_sentiment_category = case_when(
      scaled_combined_sentiment_score < 2 ~ "Strong Negative",
      scaled_combined_sentiment_score < 3 ~ "Moderately Negative",
      scaled_combined_sentiment_score < 4 ~ "Neutral",
      scaled_combined_sentiment_score < 5 ~ "Moderately Positive",
      scaled_combined_sentiment_score == 5 ~ "Strong Positive"
    )
  )

head(df %>% select(id, content_sentiment_score, content_sentiment_category, 
                   title_sentiment_score, title_sentiment_category, 
                   combined_sentiment_score, combined_sentiment_category))

# EDA
library(dplyr)
library(ggplot2)
library(tidyverse)
library(corrplot)

str(df)

summary(df)

colSums(is.na(df))

numeric_data <- df %>% select(where(is.numeric)) %>% select(-id, -normalized_combined_sentiment_score , -normalized_title_sentiment_score ,-normalized_content_sentiment_score, -scaled_combined_sentiment_score, -scaled_title_sentiment_score, -scaled_content_sentiment_score, -product_name_avg_rating, -product_id_avg_rating )  # Remove only the id column
correlation_matrix <- cor(numeric_data, use = "complete.obs")

corrplot::corrplot(correlation_matrix, method = "color", type = "upper", tl.col = "black", tl.srt = 45, na.label = " ")

# Distribution of Rating Count
ggplot(df, aes(x = rating_count)) +
  geom_histogram(bins = 30, fill = "skyblue", color = "black") +
  labs(title = "Distribution of Rating Count", x = "Rating Count", y = "Count") +
  theme_minimal()

# Distribution of Discounted Price
ggplot(df, aes(x = discounted_price)) +
  geom_histogram(bins = 30, fill = "blue", color = "black") +
  labs(title = "Distribution of Discounted Price", x = "Discounted Price", y = "Count") +
  theme_minimal()

# Distribution of Combined Sentiment Score
ggplot(df, aes(x = combined_sentiment_score)) +
  geom_histogram(bins = 30, fill = "darkblue", color = "black") +
  labs(title = "Distribution of Combined Sentiment Score", x = "Combined Sentiment Score", y = "Count") +
  theme_minimal()


# Average Combined Sentiment Score by Main Category (Barplot)
category_sentiment <- df %>%
  group_by(main_category) %>%
  summarize(avg_combined_sentiment = mean(combined_sentiment_score, na.rm = TRUE))

ggplot(category_sentiment, aes(x = reorder(main_category, -avg_combined_sentiment), y = avg_combined_sentiment)) +
  geom_bar(stat = "identity", fill = "lightblue") +
  coord_flip() +
  labs(title = "Distribution of Average Combined Sentiment Score by Main Category", x = "Main Category", y = "Average Combined Sentiment Score") +
  theme_minimal()

# Average Combined Sentiment Score by Sub-Category (Barplot)
sub_category_sentiment <- df %>%
  group_by(sub_category) %>%
  summarize(avg_combined_sentiment = mean(combined_sentiment_score, na.rm = TRUE))

ggplot(sub_category_sentiment, aes(x = reorder(sub_category, -avg_combined_sentiment), y = avg_combined_sentiment)) +
  geom_bar(stat = "identity", fill = "lightblue") +
  coord_flip() +
  labs(title = "Distribution of Combined Sentiment Scores by Sub-Category", x = "Sub-Category", y = "Average Combined Sentiment Score") +
  theme_minimal()

# Average Content Sentiment Score by Main Category (Barplot)
main_category_content_sentiment <- df %>%
  group_by(main_category) %>%
  summarize(avg_content_sentiment = mean(content_sentiment_score, na.rm = TRUE))

ggplot(main_category_content_sentiment, aes(x = reorder(main_category, -avg_content_sentiment), y = avg_content_sentiment)) +
  geom_bar(stat = "identity", fill = "orange") +
  coord_flip() +
  labs(title = "Distribution of Content Sentiment Scores by Main Category", x = "Main Category", y = "Average Content Sentiment Score") +
  theme_minimal()

# Average Content Sentiment Score by Sub-Category (Barplot)
sub_category_content_sentiment <- df %>%
  group_by(sub_category) %>%
  summarize(avg_content_sentiment = mean(content_sentiment_score, na.rm = TRUE))

ggplot(sub_category_content_sentiment, aes(x = reorder(sub_category, -avg_content_sentiment), y = avg_content_sentiment)) +
  geom_bar(stat = "identity", fill = "orange") +
  coord_flip() +
  labs(title = "Distribution of Content Sentiment Scores by Sub-Category", x = "Sub-Category", y = "Average Content Sentiment Score") +
  theme_minimal()

# Average Title Sentiment Score by Main Category (Barplot)
main_category_title_sentiment <- df %>%
  group_by(main_category) %>%
  summarize(avg_title_sentiment = mean(title_sentiment_score, na.rm = TRUE))

ggplot(main_category_title_sentiment, aes(x = reorder(main_category, -avg_title_sentiment), y = avg_title_sentiment)) +
  geom_bar(stat = "identity", fill = "skyblue") +
  coord_flip() +
  labs(title = "Distribution of Title Sentiment Scores by Main Category", x = "Main Category", y = "Average Title Sentiment Score") +
  theme_minimal()

# Average Title Sentiment Score by Sub-Category (Barplot)
sub_category_title_sentiment <- df %>%
  group_by(sub_category) %>%
  summarize(avg_title_sentiment = mean(title_sentiment_score, na.rm = TRUE))

ggplot(sub_category_title_sentiment, aes(x = reorder(sub_category, -avg_title_sentiment), y = avg_title_sentiment)) +
  geom_bar(stat = "identity", fill = "skyblue") +
  coord_flip() +
  labs(title = "Distribution of Title Sentiment Scores by Sub-Category", x = "Sub-Category", y = "Average Title Sentiment Score") +
  theme_minimal()

# Exploring Relationships
# Summary Statistics for Key Variables
summary_stats <- df %>%
  summarize(
    avg_rating_count = mean(rating_count, na.rm = TRUE),
    avg_discounted_price = mean(discounted_price, na.rm = TRUE),
    avg_combined_sentiment = mean(combined_sentiment_score, na.rm = TRUE),
    avg_content_sentiment = mean(content_sentiment_score, na.rm = TRUE),
    avg_title_sentiment = mean(title_sentiment_score, na.rm = TRUE)
  )
print(summary_stats)


sentiment_levels <- c("Strong Negative", "Moderately Negative", "Neutral", "Moderately Positive", "Strong Positive")

# Binning Sentiment Scores into Categories
df <- df %>%
  mutate(
    combined_sentiment_category = factor(case_when(
      combined_sentiment_score < -0.6 ~ "Strong Negative",
      combined_sentiment_score >= -0.6 & combined_sentiment_score < -0.1 ~ "Moderately Negative",
      combined_sentiment_score >= -0.1 & combined_sentiment_score <= 0.1 ~ "Neutral",
      combined_sentiment_score > 0.1 & combined_sentiment_score <= 0.6 ~ "Moderately Positive",
      combined_sentiment_score > 0.6 ~ "Strong Positive"
    ), levels = sentiment_levels),
    
    content_sentiment_category = factor(case_when(
      content_sentiment_score < -0.6 ~ "Strong Negative",
      content_sentiment_score >= -0.6 & content_sentiment_score < -0.1 ~ "Moderately Negative",
      content_sentiment_score >= -0.1 & content_sentiment_score <= 0.1 ~ "Neutral",
      content_sentiment_score > 0.1 & content_sentiment_score <= 0.6 ~ "Moderately Positive",
      content_sentiment_score > 0.6 ~ "Strong Positive"
    ), levels = sentiment_levels),
    
    title_sentiment_category = factor(case_when(
      title_sentiment_score < -0.6 ~ "Strong Negative",
      title_sentiment_score >= -0.6 & title_sentiment_score < -0.1 ~ "Moderately Negative",
      title_sentiment_score >= -0.1 & title_sentiment_score <= 0.1 ~ "Neutral",
      title_sentiment_score > 0.1 & title_sentiment_score <= 0.6 ~ "Moderately Positive",
      title_sentiment_score > 0.6 ~ "Strong Positive"
    ), levels = sentiment_levels)
  )

# Distribution of Sentiment Categories with Correct Ordering
ggplot(df, aes(x = combined_sentiment_category)) +
  geom_bar(fill = "lightblue", color = "black") +
  labs(title = "Distribution of Combined Sentiment Categories", x = "Sentiment Category", y = "Count") +
  theme_minimal()

ggplot(df, aes(x = content_sentiment_category)) +
  geom_bar(fill = "blue", color = "black") +
  labs(title = "Distribution of Content Sentiment Categories", x = "Sentiment Category", y = "Count") +
  theme_minimal()

ggplot(df, aes(x = title_sentiment_category)) +
  geom_bar(fill = "darkblue", color = "black") +
  labs(title = "Distribution of Title Sentiment Categories", x = "Sentiment Category", y = "Count") +
  theme_minimal()

# Sentiment Distribution by Rating
ggplot(df, aes(x = factor(rating), fill = combined_sentiment_category)) +
  geom_bar(position = "fill") +
  labs(title = "Sentiment Distribution by Product Rating", x = "Product Rating", y = "Proportion") +
  theme_minimal() +
  scale_fill_manual(values = c("red", "orange", "gray", "lightblue", "green")) +
  scale_x_discrete(drop = FALSE)  # Ensures all ratings are shown

# Price vs Sentiment Analysis
# Relationship between Combined Sentiment Score and Discounted Price
ggplot(df, aes(x = discounted_price, y = combined_sentiment_score)) +
  geom_point(alpha = 0.5, color = "purple") +
  labs(title = "Relationship between Discounted Price and Combined Sentiment Score", x = "Discounted Price", y = "Combined Sentiment Score") +
  theme_minimal()

# Relationship between Combined Sentiment Score and Actual Price
ggplot(df, aes(x = actual_price, y = combined_sentiment_score)) +
  geom_point(alpha = 0.5, color = "blue") +
  labs(title = "Relationship between Actual Price and Combined Sentiment Score", x = "Actual Price", y = "Combined Sentiment Score") +
  theme_minimal()

# Average Rating Count by Main Category
df %>%
  group_by(main_category) %>%
  summarize(avg_rating_count = mean(rating_count, na.rm = TRUE)) %>%
  ggplot(aes(x = reorder(main_category, -avg_rating_count), y = avg_rating_count)) +
  geom_bar(stat = "identity", fill = "coral") +
  coord_flip() +
  labs(title = "Average Rating Count by Main Category", x = "Main Category", y = "Average Rating Count") +
  theme_minimal()

# Average Rating Count by Sub Category
df %>%
  group_by(sub_category) %>%
  summarize(avg_rating_count = mean(rating_count, na.rm = TRUE)) %>%
  ggplot(aes(x = reorder(sub_category, -avg_rating_count), y = avg_rating_count)) +
  geom_bar(stat = "identity", fill = "lightcoral") +
  coord_flip() +
  labs(title = "Average Rating Count by Sub Category", x = "Sub Category", y = "Average Rating Count") +
  theme_minimal()

# Average Rating by Main Category
df %>%
  group_by(main_category) %>%
  summarize(avg_rating = mean(rating, na.rm = TRUE)) %>%
  ggplot(aes(x = reorder(main_category, -avg_rating), y = avg_rating)) +
  geom_bar(stat = "identity", fill = "coral") +
  coord_flip() +
  labs(title = "Average Rating by Main Category", x = "Main Category", y = "Average Rating") +
  theme_minimal()

# Average Rating by Sub Category
df %>%
  group_by(sub_category) %>%
  summarize(avg_rating = mean(rating, na.rm = TRUE)) %>%
  ggplot(aes(x = reorder(sub_category, -avg_rating), y = avg_rating)) +
  geom_bar(stat = "identity", fill = "lightcoral") +
  coord_flip() +
  labs(title = "Average Rating by Sub Category", x = "Sub Category", y = "Average Rating") +
  theme_minimal()


