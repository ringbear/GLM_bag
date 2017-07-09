library(tidyverse)
library(data.table)

data <- read.csv('Data/fire_contents_model.csv', na.strings=c(''))

View(data)
str(data)

## Summary
summarise(data, sum(sumInsured))
summarise_if(data, is.numeric, mean)
summarise_all(data, mean)

count(data, company)

## Group Class
data %>%
  group_by(company) %>%
  summarise(avg_sumInsured = mean(sumInsured))

grouped <- group_by(data, company)
ungroup <- ungroup(grouped)

remove(grouped, ungroup, test)

## Manipulate Cases (rows)
# sampling
vero <- filter(data, company == 3)
vero_2 <- filter(data, channel %in% c("BROKER","ADVISOR"))
distinct(vero, channel)

subset <- sample_frac(data, 0.5)
boostrap <- sample_frac(data, 0.5, replace = F)

samp_100 <- sample_n(data, 100)

# select rows by position
sliced <- slice(data, 100:200)

top_100 <- top_n(data, 100)

# sort
sort_asc <- arrange(data, sumInsured)
sort_desc <- arrange(data, desc(sumInsured))

# add row
added <- add_row(data, company = 99)

## Manipulate Variables (columns)
# extract
vert_set <- select(data, sumInsured, company, busnessrc)
data %>%
  select(contains(SECY)) %>%
  select(ends_with(fr_c)) %>%
  select(matches(yearOfConstr)) %>%
  select(num_range(length_rddrv_301:length_rddrv_308)) %>%
  select(one_of(company)) %>%
  select(starts_with(occ))

test <- select(starts_with(occ))

# make new variables
test <- mutate(data, ind = (company == 3)) # create var

test <- transmute(data, ind = (company == 3)) # Create var and remove the orginal mentioned vars

# apply a function to all vars
test <- mutate_all(data, function(x){as.factor(x)})
test <- mutate_all(data, funs(as.factor(.)))

# apply function to single variable
test <- mutate_at(data, vars(yearsInsured), funs(log(.)))
test <- mutate_if(data, is.numeric, funs(mean(.)))

test <- add_column(data, new = 1:nrow(data))

test <- rename(data, testing_var = company) # new_var = old_var

test <- dplyr::coalesce(as.vector(data["SECY_ELECTRONIC"]), as.vector(data["SECY_LIGHTING"]))


## Joins
# combine vars
left_join()
right_join()
inner_join()
full_join()
bind_cols()

# combine cols
bind_row()
intersect()
setdiff(x,y) # rows that appear in x but not y
union() # no duplicates
union_all # retains duplicates

### other cheatsheet
## syntax
dplyr::glimpse(data)

## reshaping data
test <- tidyr::gather(data[1:3], "company", "pds", 1:3)
test <- spread(data[1:2], "company", "busnessrc")
View(test)

test <- tidyr::unite(data, combination, company, busnessrc, product, sep = " ") # combine columns
View(test)
test_2 <- tidyr::separate(test, combination , c("company", "busnessrc", "product"), sep = " ")
View(test_2)








































