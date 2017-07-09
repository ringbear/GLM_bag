library(insuranceData)
?insuranceData


table_na <- function(...) {
  table(..., useNA="ifany")
}

data("AutoClaims")

AutoClaims[sample(1:length(AutoClaims), 1)] <- NA
length(AutoClaims)

View(AutoClaims)
table_na(AutoClaims)

View(table_na(AutoClaims))
