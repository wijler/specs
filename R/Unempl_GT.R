#' Unemployment and Google Trends Data
#'
#' Time series data on Dutch unemployment from Statistics Netherlands, and Google Trends popularity index for search terms related to unemployment. The Google Trends data can be used to nowcast unemployment.
#'
#' @format A time series object where the first column contains monthly total unemployment in the Netherlands (x1000, seasonally unadjusted), and the remaining 87 columns are monthly Google Trends series with popularity of Dutch search terms related to unemployment.
#'
#' @source CBS StatLine, https://opendata.cbs.nl/statline, and Google Trends, https://www.google.nl/trends
"Unempl_GT"
