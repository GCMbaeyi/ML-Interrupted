# =============================================================================
# EMPIRICAL RE-ANALYSIS OF OMOTOSHO INTERRUPTED TIME SERIES
# Matched to Monte Carlo
# =============================================================================
#
# Study data: monthly power-generation data, January 2010 to December 2021
# Intervention: January 2016 step intervention
#
# PRIMARY HOLD-OUT:
#   Training = January 2010 to December 2020 (132 months)
#   Test     = January 2021 to December 2021 (12 months; one full year)
#
# This script was designed to address the reviewer's main concerns by:
#   * fitting the same nine models used in Monte Carlo Version 5;
#   * explicitly reporting all XGBoost hyperparameters and PACF-selected lags;
#   * explicitly reporting the stacking rule and fitted stacking weights;
#   * computing RMSE, MAE, MAPE and sMAPE on the 2021 test year;
#   * computing moving-block-bootstrap 95% CIs for the accuracy measures;
#   * providing Diebold-Mariano-HLN sensitivity tests, especially versus M5;
#   * breaking forecast errors down by observed-value magnitude;
#   * repeating one-year-ahead backtests for 2017-2021;
#   * reporting selected ARIMA/I-ARIMA orders and coefficients;
#   * comparing the empirical results with saved Monte Carlo Version-5 outputs
#     when those files are available.
#
# =============================================================================


# -----------------------------------------------------------------------------
# 0. PACKAGES
# -----------------------------------------------------------------------------
if (getRversion() < "4.1.0") {
  stop("R 4.1.0 or newer is required.")
}

required_pkgs <- c(
  "xgboost",
  "dplyr",
  "tidyr",
  "tibble",
  "readr",
  "ggplot2"
)

missing_pkgs <- required_pkgs[
  !vapply(required_pkgs, requireNamespace, quietly = TRUE, FUN.VALUE = logical(1))
]

if (length(missing_pkgs) > 0L) {
  stop(
    "\nMissing packages: ", paste(missing_pkgs, collapse = ", "),
    "\nInstall with:\ninstall.packages(c(",
    paste(sprintf('"%s"', missing_pkgs), collapse = ", "),
    "))\n"
  )
}

suppressPackageStartupMessages({
  library(xgboost)
  library(dplyr)
  library(tidyr)
  library(tibble)
  library(readr)
  library(ggplot2)
})


# -----------------------------------------------------------------------------
# 1. SETTINGS MATCHED TO MONTE CARLO VERSION 5
# -----------------------------------------------------------------------------
BASE_SEED <- 20260923L

MAX_PACF_LAG <- 12L
CLIP_POSITIVE <- TRUE

ARIMA_MAX_P <- 2L
ARIMA_MAX_D <- 1L
ARIMA_MAX_Q <- 2L

STACK_META_FRAC <- 0.20
STACK_MIN_META_OBS <- 5L
STACK_MIN_POST_INT_BASE_OBS <- 2L

XGB_PARAMS <- list(
  objective = "reg:squarederror",
  eta = 0.10,
  max_depth = 3L,
  min_child_weight = 1,
  subsample = 0.80,
  colsample_bytree = 0.80,
  gamma = 0,
  lambda = 1,
  alpha = 0,
  tree_method = "hist",
  max_bin = 64L,
  nthread = 1L,
  verbosity = 0L
)

XGB_NROUNDS <- 100L

MODEL_LABELS <- c(
  M1 = "ARIMA",
  M2 = "Intervention-ARIMA",
  M3 = "XGBoost-original",
  M4 = "XGBoost-transformed",
  M5 = "XGBoost-stationary",
  M6 = "XGBoost-plus-intervention",
  M7 = "Stack-ARIMA-XGBoost",
  M8 = "Stack-IARIMA-XGBoost",
  M9 = "Stack-ARIMA-IARIMA-XGBoost"
)

METRIC_NAMES <- c("RMSE", "MAE", "MAPE", "sMAPE")

RESULT_DIR <- "empirical_reanalysis_2021_test"
dir.create(RESULT_DIR, showWarnings = FALSE, recursive = TRUE)

# If the Monte Carlo summary folder is elsewhere, edit this path.
SIM_SUMMARY_DIR <- file.path("mc_fast_v5_OA54_results", "summaries")

write_csv_quiet <- function(x, file) {
  readr::write_csv(x, file, progress = FALSE)
}


# -----------------------------------------------------------------------------
# 2. STUDY DATA
# -----------------------------------------------------------------------------
years <- 2010:2021
months <- month.name

power_matrix <- matrix(
  c(
    10318,6108,20751,10075,11412,20238,98894,94117,76060,90542,63942.3,96747.9,
    10986,6225,21834,11398,12637,17203,111847,97808,84070,88362,61554.9,93985.7,
    11503,7703,16608,11698,15594,19894,100371,100774,99141,94862,75762.8,80536.4,
    12434,10444,11210,21987,15814,10758,43870,100658,114895,104739,80434.6,56685.5,
    16576,11673,12652,22060,14973,11564,41818,115954,91468,115391,87608.9,65129.9,
    19629,13108,15768,14429,17933,13367,47690,99266,99754,112142,83176.5,60474.6,
    11668,8564,16755,15502,15277,20458,79929,70488,96982,93612,123062,97273.7,
    10213,8157,13328,13559,19790,19887,76044,72435,97088,65626,86745.3,78199.2,
    9780,7086,14987,19600,21115,15669,73651,75731,87708,71631,115683,92587.1,
    5579,12779,21770,12314,22186,22697,85231,114255,71275,64348,93448.3,96092.8,
    5803,14553,16074,11690,20785,15744,89971,111974,88583,72372,103728,110161,
    5028,17600,12897,10016,16603,20535,86763,100215,99322,61191,90213.8,91856.7
  ),
  nrow = 12L,
  ncol = 12L,
  byrow = TRUE,
  dimnames = list(months, as.character(years))
)

study_data <- tibble::tibble(
  date = seq(as.Date("2010-01-01"), as.Date("2021-12-01"), by = "month"),
  year = rep(years, each = 12L),
  month = factor(rep(months, times = length(years)),
                 levels = months, ordered = TRUE),
  y = as.numeric(power_matrix)
) |>
  dplyr::mutate(
    intervention = as.integer(date >= as.Date("2016-01-01"))
  )

stopifnot(
  nrow(study_data) == 144L,
  all(is.finite(study_data$y)),
  all(study_data$y > 0)
)

write_csv_quiet(study_data, file.path(RESULT_DIR, "00_study_data_used.csv"))

TRAIN_END_DATE <- as.Date("2020-12-01")
TEST_START_DATE <- as.Date("2021-01-01")

train_data <- study_data |> dplyr::filter(date <= TRAIN_END_DATE)
test_data  <- study_data |> dplyr::filter(date >= TEST_START_DATE)

stopifnot(nrow(train_data) == 132L, nrow(test_data) == 12L)




# -----------------------------------------------------------------------------
# 3. BOX-COX UTILITIES
# -----------------------------------------------------------------------------
boxcox_transform <- function(y, lambda) {
  y <- as.numeric(y)

  if (any(!is.finite(y)) || any(y <= 0)) {
    stop("Box-Cox transformation requires finite positive observations.")
  }

  if (abs(lambda) < 1e-8) {
    log(y)
  } else {
    (y^lambda - 1) / lambda
  }
}


inv_boxcox <- function(z, lambda) {
  z <- as.numeric(z)

  if (abs(lambda) < 1e-8) {
    out <- exp(z)
  } else {
    base <- 1 + lambda * z
    base <- pmax(base, 1e-10)
    out <- base^(1 / lambda)
  }

  as.numeric(out)
}


estimate_lambda <- function(y, lower = -1, upper = 2) {

  # Self-contained Box-Cox lambda estimator.
  #
  # This deliberately avoids forecast::BoxCox.lambda(), because some R
  # installations can call an optional/recommended dependency that is not
  # installed. The pilot then fails before ARIMA/XGBoost is fitted.
  #
  # Lambda is estimated by maximizing the Gaussian profile log-likelihood
  # of the Box-Cox transformed series after accounting for a linear time
  # trend. Only base-R functions are used here.

  y <- as.numeric(y)

  if (length(y) < 5L) {
    stop("Too few observations to estimate Box-Cox lambda.")
  }

  if (any(!is.finite(y)) || any(y <= 0)) {
    stop("Box-Cox lambda estimation requires finite positive observations.")
  }

  n <- length(y)
  time_index <- seq_len(n)

  # Use a centred/scaled time regressor to improve numerical stability.
  trend <- as.numeric(scale(time_index))

  profile_loglik <- function(lambda) {

    z <- if (abs(lambda) < 1e-8) {
      log(y)
    } else {
      (y^lambda - 1) / lambda
    }

    # Regress transformed observations on intercept + linear time trend.
    X <- cbind(
      intercept = 1,
      trend = trend
    )

    fit <- stats::lm.fit(
      x = X,
      y = z
    )

    rss <- sum(fit$residuals^2)

    if (!is.finite(rss) || rss <= .Machine$double.eps) {
      return(-Inf)
    }

    # Gaussian profile log-likelihood including the Box-Cox Jacobian.
    -(n / 2) * log(rss / n) +
      (lambda - 1) * sum(log(y))
  }

  opt <- stats::optimize(
    f = profile_loglik,
    interval = c(lower, upper),
    maximum = TRUE
  )

  lambda_hat <- as.numeric(opt$maximum)

  if (!is.finite(lambda_hat)) {
    stop("Box-Cox lambda optimization failed.")
  }

  lambda_hat
}


clip_positive <- function(x) {
  x <- as.numeric(x)
  if (CLIP_POSITIVE) {
    pmax(x, 1e-8)
  } else {
    x
  }
}


# -----------------------------------------------------------------------------
# 4. INTERVENTION GENERATOR
# -----------------------------------------------------------------------------
make_intervention <- function(n, type, position) {

  if (!type %in% c("step", "pulse", "ramp")) {
    stop("Unknown intervention type: ", type)
  }

  t0 <- as.integer(round(position * n))
  t0 <- max(2L, min(n, t0))

  x <- numeric(n)

  if (type == "step") {
    x[t0:n] <- 1
  }

  if (type == "pulse") {
    x[t0] <- 1
  }

  if (type == "ramp") {
    ramp_length <- n - t0 + 1L
    x[t0:n] <- seq_len(ramp_length) / ramp_length
  }

  list(
    x = x,
    t0 = t0
  )
}


# -----------------------------------------------------------------------------
# 5. DATA-GENERATING PROCESS
# -----------------------------------------------------------------------------
generate_interrupted_series <- function(
    n,
    dynamics,
    noise_sd,
    intervention_type,
    intervention_position,
    effect_log,
    base_level = 10000,
    drift = 0.002
) {

  if (!dynamics %in% c("linear", "nonlinear")) {
    stop("Unknown dynamics: ", dynamics)
  }

  burn <- 100L
  N <- n + burn

  u <- numeric(N)
  eps <- rnorm(N, mean = 0, sd = noise_sd)

  for (tt in 2:N) {

    if (dynamics == "linear") {

      # Linear AR(1) disturbance.
      u[tt] <- 0.65 * u[tt - 1L] + eps[tt]

    } else {

      # Nonlinear threshold autoregressive disturbance.
      phi <- if (u[tt - 1L] >= 0) 0.75 else 0.25
      u[tt] <- phi * u[tt - 1L] + eps[tt]
    }
  }

  u <- u[(burn + 1L):(burn + n)]

  intervention_object <- make_intervention(
    n = n,
    type = intervention_type,
    position = intervention_position
  )

  time_index <- seq_len(n)

  latent_log <- log(base_level) +
    drift * time_index +
    u +
    effect_log * intervention_object$x

  y <- exp(latent_log)

  if (any(!is.finite(y)) || any(y <= 0)) {
    stop("Simulation produced invalid observations.")
  }

  list(
    y = as.numeric(y),
    intervention = intervention_object$x,
    t0 = intervention_object$t0,
    latent_log = latent_log
  )
}


# -----------------------------------------------------------------------------
# 6. FORECAST ACCURACY METRICS
# -----------------------------------------------------------------------------
accuracy_metrics <- function(actual, pred) {

  actual <- as.numeric(actual)
  pred <- as.numeric(pred)

  ok <- is.finite(actual) & is.finite(pred)

  if (!any(ok)) {
    return(
      c(
        RMSE = NA_real_,
        MAE = NA_real_,
        MAPE = NA_real_,
        sMAPE = NA_real_
      )
    )
  }

  actual <- actual[ok]
  pred <- pred[ok]

  error <- actual - pred

  RMSE <- sqrt(mean(error^2))
  MAE <- mean(abs(error))

  nonzero <- abs(actual) > 1e-12

  MAPE <- if (any(nonzero)) {
    100 * mean(abs(error[nonzero] / actual[nonzero]))
  } else {
    NA_real_
  }

  denominator <- abs(actual) + abs(pred)

  sMAPE <- 100 * mean(
    2 * abs(error) / pmax(denominator, 1e-12)
  )

  c(
    RMSE = RMSE,
    MAE = MAE,
    MAPE = MAPE,
    sMAPE = sMAPE
  )
}


# -----------------------------------------------------------------------------
# 7. PACF-BASED LAG SELECTION
# -----------------------------------------------------------------------------
select_pacf_order <- function(x, max_lag = MAX_PACF_LAG) {

  x <- as.numeric(x)

  usable_max <- min(
    max_lag,
    floor(length(x) / 4),
    length(x) - 2L
  )

  if (usable_max < 1L) {
    return(1L)
  }

  pacf_object <- tryCatch(
    stats::pacf(
      x,
      plot = FALSE,
      lag.max = usable_max
    )$acf,
    error = function(e) NULL
  )

  if (is.null(pacf_object)) {
    return(1L)
  }

  pacf_values <- as.numeric(pacf_object)
  significance_limit <- 1.96 / sqrt(length(x))

  significant_lags <- which(
    abs(pacf_values) > significance_limit
  )

  p <- if (length(significant_lags) == 0L) {
    1L
  } else {
    max(significant_lags)
  }

  max(1L, min(as.integer(p), usable_max))
}


make_lagged_xy <- function(series, p, xreg = NULL) {

  series <- as.numeric(series)
  n <- length(series)

  if (n <= p + 2L) {
    stop("Series is too short for the selected lag order.")
  }

  index <- (p + 1L):n

  X <- sapply(
    seq_len(p),
    function(j) series[index - j]
  )

  X <- as.matrix(X)
  colnames(X) <- paste0("lag", seq_len(p))

  if (!is.null(xreg)) {

    xreg <- as.numeric(xreg)

    if (length(xreg) != n) {
      stop("xreg length does not match the response-series length.")
    }

    X <- cbind(
      X,
      intervention = xreg[index]
    )
  }

  list(
    X = X,
    y = series[index]
  )
}


# -----------------------------------------------------------------------------
# 8. RECURSIVE XGBOOST FORECASTER
# -----------------------------------------------------------------------------
xgb_recursive_forecast <- function(
    train_series,
    h,
    xreg_train = NULL,
    xreg_future = NULL,
    seed = 1L
) {

  train_series <- as.numeric(train_series)

  p <- select_pacf_order(train_series)

  lagged <- make_lagged_xy(
    series = train_series,
    p = p,
    xreg = xreg_train
  )

  set.seed(seed)

  dtrain <- xgboost::xgb.DMatrix(
    data = lagged$X,
    label = lagged$y
  )

  fit <- xgboost::xgb.train(
    params = XGB_PARAMS,
    data = dtrain,
    nrounds = XGB_NROUNDS,
    verbose = 0
  )

  history <- train_series
  forecast_values <- numeric(h)

  for (k in seq_len(h)) {

    lag_values <- vapply(
      seq_len(p),
      function(j) {
        history[length(history) - j + 1L]
      },
      FUN.VALUE = numeric(1)
    )

    new_X <- matrix(
      lag_values,
      nrow = 1L
    )

    colnames(new_X) <- paste0("lag", seq_len(p))

    if (!is.null(xreg_train)) {

      if (is.null(xreg_future) || length(xreg_future) < h) {
        stop("Future intervention regressor is missing or too short.")
      }

      new_X <- cbind(
        new_X,
        intervention = as.numeric(xreg_future[k])
      )
    }

    prediction <- as.numeric(
      stats::predict(
        fit,
        xgboost::xgb.DMatrix(new_X)
      )
    )

    forecast_values[k] <- prediction

    history <- c(
      history,
      prediction
    )
  }

  list(
    pred = forecast_values,
    p = p
  )
}


# -----------------------------------------------------------------------------
# 9. BASE-R ARIMA HELPERS AND MODEL FORECAST FUNCTIONS
# -----------------------------------------------------------------------------
# Why base R here?
# ----------------
# Earlier pilot runs showed that XGBoost Models 3-6 worked, while Models 1-2
# failed inside forecast::auto.arima() because a package dependency was absent
# on the local installation. To make the Monte Carlo experiment portable, the
# ARIMA pathway below uses only stats::arima() and stats::predict().
#
# Candidate orders are selected by AICc from a prespecified compact set of
# common low-order non-seasonal ARIMA models. This retains automatic,
# data-driven order selection while avoiding the computational cost of fitting
# every p=0..3, d=0..1, q=0..3 combination in every Monte Carlo replication.
#
# The same candidate set is used for both ARIMA and Intervention-ARIMA.


aicc_arima <- function(fit, n_obs) {

  k <- length(stats::coef(fit)) + 1L  # +1 for innovation variance

  aic <- fit$aic

  if (!is.finite(aic)) {
    return(Inf)
  }

  denom <- n_obs - k - 1L

  if (denom <= 0) {
    return(Inf)
  }

  aic + (2 * k * (k + 1L)) / denom
}


fit_best_arima_base <- function(
    z,
    xreg = NULL,
    max_p = ARIMA_MAX_P,
    max_d = ARIMA_MAX_D,
    max_q = ARIMA_MAX_Q
) {

  # max_p/max_d/max_q are retained in the function signature for backward
  # compatibility with the validated pilot pathway. Version 5 uses the fixed
  # compact candidate set defined below for speed.

  z <- as.numeric(z)

  if (any(!is.finite(z))) {
    stop("ARIMA input contains non-finite observations.")
  }

  if (!is.null(xreg)) {

    xreg <- as.matrix(xreg)

    if (nrow(xreg) != length(z)) {
      stop("xreg has the wrong number of rows for ARIMA fitting.")
    }

    if (any(!is.finite(xreg))) {
      stop("xreg contains non-finite values.")
    }
  }

  # Compact candidate set: 12 common low-order models.
  # p/q values up to 2 capture the DGP's short-memory structure while both
  # stationary (d=0) and first-differenced (d=1) specifications are considered.
  candidates <- data.frame(
    p = c(
      1L, 0L, 1L, 2L, 0L, 2L,
      0L, 1L, 0L, 1L, 2L, 0L
    ),
    d = c(
      0L, 0L, 0L, 0L, 0L, 0L,
      1L, 1L, 1L, 1L, 1L, 1L
    ),
    q = c(
      0L, 1L, 1L, 0L, 2L, 1L,
      0L, 0L, 1L, 1L, 0L, 2L
    )
  )

  best_fit <- NULL
  best_order <- NULL
  best_aicc <- Inf
  failure_messages <- character(0)

  for (ii in seq_len(nrow(candidates))) {

    ord <- as.integer(
      candidates[ii, c("p", "d", "q")]
    )

    fit <- tryCatch(
      suppressWarnings(
        stats::arima(
          x = z,
          order = ord,
          xreg = xreg,
          include.mean = (ord[2] == 0L),
          method = "ML"
        )
      ),
      error = function(e) e
    )

    if (inherits(fit, "error")) {

      failure_messages <- c(
        failure_messages,
        paste0(
          "ARIMA(",
          paste(ord, collapse = ","),
          "): ",
          conditionMessage(fit)
        )
      )

      next
    }

    candidate_aicc <- aicc_arima(
      fit = fit,
      n_obs = length(z)
    )

    if (
      is.finite(candidate_aicc) &&
      candidate_aicc < best_aicc
    ) {

      # IMPORTANT SCOPING FIX FOR stats::predict.Arima:
      # stats::arima() records xreg symbolically in fit$call. If that
      # symbolic name is later evaluated outside this helper function,
      # predict.Arima can throw "object 'xreg' not found".
      #
      # Store the actual matrix in the call when xreg is used, and remove
      # the call entry entirely when no xreg is used.
      if (is.null(xreg)) {
        fit$call$xreg <- NULL
      } else {
        fit$call$xreg <- xreg
      }

      best_fit <- fit
      best_order <- ord
      best_aicc <- candidate_aicc
    }
  }

  if (is.null(best_fit)) {

    detail <- if (length(failure_messages) > 0L) {
      paste(
        head(unique(failure_messages), 5L),
        collapse = " | "
      )
    } else {
      "No candidate model returned a finite AICc."
    }

    stop(
      "No candidate ARIMA model could be fitted. ",
      detail
    )
  }

  list(
    fit = best_fit,
    order = best_order,
    aicc = best_aicc,
    training_xreg = xreg
  )
}


predict_arima_base <- function(
    fitted_object,
    h,
    newxreg = NULL
) {

  fit <- fitted_object$fit

  # Defensive safeguard for predict.Arima's historical non-standard
  # evaluation of xreg. Normally the fitted call has already been
  # sanitized in fit_best_arima_base().
  xreg <- fitted_object$training_xreg

  if (!is.null(newxreg)) {

    newxreg <- as.matrix(newxreg)

    if (nrow(newxreg) != h) {
      stop("newxreg has the wrong number of rows for forecasting.")
    }

    pred <- stats::predict(
      fit,
      n.ahead = h,
      newxreg = newxreg
    )$pred

  } else {

    pred <- stats::predict(
      fit,
      n.ahead = h
    )$pred
  }

  pred <- as.numeric(pred)

  if (
    length(pred) != h ||
    any(!is.finite(pred))
  ) {
    stop("Base-R ARIMA forecast was invalid.")
  }

  pred
}


# MODEL 1: ARIMA
forecast_M1_ARIMA <- function(y_train, h) {

  lambda <- estimate_lambda(y_train)
  z <- boxcox_transform(y_train, lambda)

  selected <- fit_best_arima_base(
    z = z
  )

  zhat <- predict_arima_base(
    fitted_object = selected,
    h = h
  )

  list(
    pred = clip_positive(
      inv_boxcox(zhat, lambda)
    ),
    note = paste0(
      "lambda=", round(lambda, 4),
      "; ARIMA(",
      paste(selected$order, collapse = ","),
      "); AICc=",
      round(selected$aicc, 3)
    )
  )
}


# MODEL 2: Intervention-ARIMA
forecast_M2_IARIMA <- function(
    y_train,
    int_train,
    int_future,
    h
) {

  lambda <- estimate_lambda(y_train)
  z <- boxcox_transform(y_train, lambda)

  int_train <- as.numeric(int_train)
  int_future <- as.numeric(int_future)

  # If the intervention regressor has no variation in the estimation window,
  # its coefficient is unidentified. For that internal stacking window only,
  # fit the ARIMA component and record the fallback explicitly.
  if (length(unique(int_train)) < 2L) {

    selected <- fit_best_arima_base(
      z = z
    )

    zhat <- predict_arima_base(
      fitted_object = selected,
      h = h
    )

    return(
      list(
        pred = clip_positive(
          inv_boxcox(zhat, lambda)
        ),
        note = paste0(
          "fallback: intervention regressor had no variation; ARIMA(",
          paste(selected$order, collapse = ","),
          ")"
        )
      )
    )
  }

  xr_train <- matrix(
    int_train,
    ncol = 1L,
    dimnames = list(NULL, "intervention")
  )

  xr_future <- matrix(
    int_future,
    ncol = 1L,
    dimnames = list(NULL, "intervention")
  )

  selected <- fit_best_arima_base(
    z = z,
    xreg = xr_train
  )

  zhat <- predict_arima_base(
    fitted_object = selected,
    h = h,
    newxreg = xr_future
  )

  list(
    pred = clip_positive(
      inv_boxcox(zhat, lambda)
    ),
    note = paste0(
      "lambda=", round(lambda, 4),
      "; ARIMA(",
      paste(selected$order, collapse = ","),
      ") + intervention; AICc=",
      round(selected$aicc, 3)
    )
  )
}


# MODEL 3: XGBoost on original series
forecast_M3_XGB_original <- function(
    y_train,
    h,
    seed
) {

  ans <- xgb_recursive_forecast(
    train_series = y_train,
    h = h,
    seed = seed
  )

  list(
    pred = clip_positive(ans$pred),
    note = paste0("PACF lag order=", ans$p)
  )
}


# MODEL 4: XGBoost on transformed series
forecast_M4_XGB_transformed <- function(
    y_train,
    h,
    seed
) {

  lambda <- estimate_lambda(y_train)
  z <- boxcox_transform(y_train, lambda)

  ans <- xgb_recursive_forecast(
    train_series = z,
    h = h,
    seed = seed
  )

  list(
    pred = clip_positive(
      inv_boxcox(ans$pred, lambda)
    ),
    note = paste0(
      "lambda=", round(lambda, 4),
      "; PACF lag order=", ans$p
    )
  )
}


# MODEL 5: XGBoost on stationary transformed + differenced series
forecast_M5_XGB_stationary <- function(
    y_train,
    h,
    seed
) {

  lambda <- estimate_lambda(y_train)
  z <- boxcox_transform(y_train, lambda)
  dz <- diff(z)

  ans <- xgb_recursive_forecast(
    train_series = dz,
    h = h,
    seed = seed
  )

  zhat <- tail(z, 1L) +
    cumsum(ans$pred)

  list(
    pred = clip_positive(
      inv_boxcox(zhat, lambda)
    ),
    note = paste0(
      "lambda=", round(lambda, 4),
      "; differenced; PACF lag order=", ans$p
    )
  )
}


# MODEL 6: XGBoost + intervention regressor
forecast_M6_XGB_intervention <- function(
    y_train,
    int_train,
    int_future,
    h,
    seed
) {

  lambda <- estimate_lambda(y_train)
  z <- boxcox_transform(y_train, lambda)

  ans <- xgb_recursive_forecast(
    train_series = z,
    h = h,
    xreg_train = int_train,
    xreg_future = int_future,
    seed = seed
  )

  list(
    pred = clip_positive(
      inv_boxcox(ans$pred, lambda)
    ),
    note = paste0(
      "lambda=", round(lambda, 4),
      "; intervention predictor included; PACF lag order=", ans$p
    )
  )
}


# -----------------------------------------------------------------------------
# 10. SAFE MODEL EXECUTION
# -----------------------------------------------------------------------------
run_safely <- function(
    model_id,
    model_label,
    h,
    expression
) {

  tryCatch(
    {

      ans <- force(expression)
      pred <- as.numeric(ans$pred)

      if (length(pred) != h) {
        stop(
          "Forecast length was ",
          length(pred),
          " but expected ",
          h,
          "."
        )
      }

      if (any(!is.finite(pred))) {
        stop("Forecast contains non-finite values.")
      }

      list(
        pred = pred,
        status = "OK",
        message = "",
        note = ifelse(
          is.null(ans$note),
          "",
          as.character(ans$note)
        )
      )
    },
    error = function(e) {

      list(
        pred = rep(NA_real_, h),
        status = "FAILED",
        message = conditionMessage(e),
        note = ""
      )
    }
  )
}




# -----------------------------------------------------------------------------
# 11. CONVEX STACKING META-LEARNER
# -----------------------------------------------------------------------------
# The pilot's unconstrained OLS stacking occasionally produced extremely large
# forecasts. The full experiment therefore uses a constrained convex forecast
# combination. All weights are >= 0 and sum to 1.
#
# This is estimated by minimizing mean squared error on a chronological
# meta-validation block using a softmax parameterization.


remove_redundant_columns <- function(
    X,
    tolerance = 1e-10
) {

  X <- as.data.frame(X)

  if (ncol(X) == 0L) {
    return(character(0))
  }

  kept <- character(0)

  for (nm in names(X)) {

    candidate_names <- c(
      kept,
      nm
    )

    candidate <- as.matrix(
      X[
        ,
        candidate_names,
        drop = FALSE
      ]
    )

    # Convex stacking has no free intercept. Therefore redundancy is
    # assessed among the forecast columns themselves, not after adding an
    # intercept column.
    rank_without <- if (
      length(kept) == 0L
    ) {

      0L

    } else {

      qr(
        as.matrix(
          X[
            ,
            kept,
            drop = FALSE
          ]
        ),
        tol = tolerance
      )$rank
    }

    rank_with <- qr(
      candidate,
      tol = tolerance
    )$rank

    if (rank_with > rank_without) {
      kept <- c(
        kept,
        nm
      )
    }
  }

  kept
}


softmax_weights <- function(theta) {

  theta <- as.numeric(theta)

  shifted <- theta - max(theta)
  exp_theta <- exp(shifted)

  exp_theta / sum(exp_theta)
}


fit_stack_convex <- function(
    meta_y,
    predictors,
    new_predictors
) {

  predictors <- as.data.frame(
    predictors
  )

  new_predictors <- as.data.frame(
    new_predictors
  )

  all_predictor_names <- names(
    predictors
  )

  training_data <- data.frame(
    y = as.numeric(meta_y),
    predictors,
    check.names = FALSE
  )

  ok <- complete.cases(
    training_data
  )

  training_data <- training_data[
    ok,
    ,
    drop = FALSE
  ]

  predictors_complete <- training_data[
    ,
    setdiff(
      names(training_data),
      "y"
    ),
    drop = FALSE
  ]

  if (
    nrow(training_data) <
      STACK_MIN_META_OBS
  ) {

    stop(
      "Too few complete observations for the stacking meta-learner: ",
      nrow(training_data)
    )
  }

  used_predictors <- remove_redundant_columns(
    predictors_complete
  )

  if (
    length(used_predictors) == 0L
  ) {

    stop(
      "No usable predictor remained for stacking."
    )
  }

  X <- as.matrix(
    predictors_complete[
      ,
      used_predictors,
      drop = FALSE
    ]
  )

  Xnew <- as.matrix(
    new_predictors[
      ,
      used_predictors,
      drop = FALSE
    ]
  )

  y <- training_data$y

  if (
    length(used_predictors) == 1L
  ) {

    estimated_weights <- 1

  } else {

    objective <- function(theta) {

      w <- softmax_weights(
        theta
      )

      fitted <- as.numeric(
        X %*% w
      )

      mean(
        (y - fitted)^2
      )
    }

    optimum <- stats::optim(
      par = rep(
        0,
        length(used_predictors)
      ),
      fn = objective,
      method = "BFGS",
      control = list(
        maxit = 1000,
        reltol = 1e-10
      )
    )

    if (
      optimum$convergence != 0L
    ) {

      stop(
        "Convex stacking optimizer did not converge. Code: ",
        optimum$convergence
      )
    }

    estimated_weights <- softmax_weights(
      optimum$par
    )
  }

  names(
    estimated_weights
  ) <- used_predictors

  full_weights <- setNames(
    rep(
      0,
      length(all_predictor_names)
    ),
    all_predictor_names
  )

  full_weights[
    used_predictors
  ] <- estimated_weights

  prediction <- as.numeric(
    Xnew %*%
      estimated_weights
  )

  if (
    any(!is.finite(prediction))
  ) {

    stop(
      "Convex stacking produced non-finite predictions."
    )
  }

  list(
    pred = clip_positive(
      prediction
    ),
    weights = full_weights,
    used = paste(
      used_predictors,
      collapse = "+"
    )
  )
}


# -----------------------------------------------------------------------------
# 12. BASE FORECASTS FOR STACKING
# -----------------------------------------------------------------------------
base_forecasts_for_stack <- function(
    y_train,
    int_train,
    int_future,
    h,
    seed
) {

  M1 <- forecast_M1_ARIMA(
    y_train = y_train,
    h = h
  )

  M2 <- forecast_M2_IARIMA(
    y_train = y_train,
    int_train = int_train,
    int_future = int_future,
    h = h
  )

  M3 <- forecast_M3_XGB_original(
    y_train = y_train,
    h = h,
    seed = seed
  )

  list(
    M1 = M1,
    M2 = M2,
    M3 = M3
  )
}


# -----------------------------------------------------------------------------
# 13. ADAPTIVE STACKING SPLIT
# -----------------------------------------------------------------------------
choose_stack_split <- function(
    int_train
) {

  int_train <- as.numeric(
    int_train
  )

  n_train <- length(
    int_train
  )

  desired_meta_n <- max(
    STACK_MIN_META_OBS,
    as.integer(
      floor(
        STACK_META_FRAC *
          n_train
      )
    )
  )

  base_end <- n_train -
    desired_meta_n

  first_intervention <- which(
    abs(int_train) >
      1e-12
  )

  first_intervention <- if (
    length(first_intervention) > 0L
  ) {

    min(
      first_intervention
    )

  } else {

    NA_integer_
  }

  # If an intervention occurs in training, force the base-training segment
  # to contain at least a small amount of intervention information.
  if (
    !is.na(first_intervention)
  ) {

    minimum_base_end <- min(
      n_train -
        STACK_MIN_META_OBS,
      first_intervention +
        STACK_MIN_POST_INT_BASE_OBS -
        1L
    )

    base_end <- max(
      base_end,
      minimum_base_end
    )
  }

  # Absolute safety condition.
  base_end <- min(
    base_end,
    n_train -
      STACK_MIN_META_OBS
  )

  if (
    base_end < 20L
  ) {

    stop(
      "Base-training segment is too short for stacking."
    )
  }

  meta_start <- base_end + 1L
  meta_n <- n_train -
    base_end

  if (
    meta_n <
      STACK_MIN_META_OBS
  ) {

    stop(
      "Meta-validation segment is too short for stacking."
    )
  }

  list(
    base_end = base_end,
    meta_start = meta_start,
    meta_n = meta_n,
    first_intervention = first_intervention
  )
}


# -----------------------------------------------------------------------------
# 14. CONVEX STACKING FORECASTS
# -----------------------------------------------------------------------------
stacking_forecasts <- function(
    y_train,
    int_train,
    final_base,
    seed
) {

  split <- choose_stack_split(
    int_train = int_train
  )

  base_end <- split$base_end
  meta_start <- split$meta_start
  meta_n <- split$meta_n

  y_base <- y_train[
    seq_len(
      base_end
    )
  ]

  int_base <- int_train[
    seq_len(
      base_end
    )
  ]

  y_meta <- y_train[
    meta_start:
      length(y_train)
  ]

  int_meta <- int_train[
    meta_start:
      length(int_train)
  ]

  meta_base <- base_forecasts_for_stack(
    y_train = y_base,
    int_train = int_base,
    int_future = int_meta,
    h = meta_n,
    seed = seed +
      1000L
  )

  meta_M1 <- meta_base$M1$pred
  meta_M2 <- meta_base$M2$pred
  meta_M3 <- meta_base$M3$pred

  final_M1 <- final_base$M1$pred
  final_M2 <- final_base$M2$pred
  final_M3 <- final_base$M3$pred

  # Model 7: ARIMA + XGBoost
  fit7 <- fit_stack_convex(
    meta_y = y_meta,
    predictors = data.frame(
      M1 = meta_M1,
      M3 = meta_M3
    ),
    new_predictors = data.frame(
      M1 = final_M1,
      M3 = final_M3
    )
  )

  # Model 8: Intervention-ARIMA + XGBoost
  fit8 <- fit_stack_convex(
    meta_y = y_meta,
    predictors = data.frame(
      M2 = meta_M2,
      M3 = meta_M3
    ),
    new_predictors = data.frame(
      M2 = final_M2,
      M3 = final_M3
    )
  )

  # Model 9: ARIMA + Intervention-ARIMA + XGBoost
  fit9 <- fit_stack_convex(
    meta_y = y_meta,
    predictors = data.frame(
      M1 = meta_M1,
      M2 = meta_M2,
      M3 = meta_M3
    ),
    new_predictors = data.frame(
      M1 = final_M1,
      M2 = final_M2,
      M3 = final_M3
    )
  )

  weight_rows <- dplyr::bind_rows(
    tibble::tibble(
      stack_model = "M7",
      base_model = names(
        fit7$weights
      ),
      weight = as.numeric(
        fit7$weights
      )
    ),
    tibble::tibble(
      stack_model = "M8",
      base_model = names(
        fit8$weights
      ),
      weight = as.numeric(
        fit8$weights
      )
    ),
    tibble::tibble(
      stack_model = "M9",
      base_model = names(
        fit9$weights
      ),
      weight = as.numeric(
        fit9$weights
      )
    )
  )

  intervention_seen_in_base <- length(
    unique(
      int_base
    )
  ) > 1L

  fallback_detected <- grepl(
    "fallback",
    meta_base$M2$note,
    ignore.case = TRUE
  )

  diagnostics <- tibble::tibble(
    base_end = base_end,
    meta_start = meta_start,
    meta_n = meta_n,
    first_intervention = split$first_intervention,
    intervention_seen_in_base = intervention_seen_in_base,
    fallback_detected = fallback_detected,
    M7_predictors = fit7$used,
    M8_predictors = fit8$used,
    M9_predictors = fit9$used
  )

  stacking_note <- paste0(
    "convex stacking; meta n=",
    meta_n,
    "; base end=",
    base_end,
    "; M7=",
    fit7$used,
    "; M8=",
    fit8$used,
    "; M9=",
    fit9$used,
    "; meta-M2=",
    meta_base$M2$note
  )

  list(
    M7 = fit7$pred,
    M8 = fit8$pred,
    M9 = fit9$pred,
    weights = weight_rows,
    diagnostics = diagnostics,
    note = stacking_note
  )
}





# -----------------------------------------------------------------------------
# 15. RUN ALL NINE MODELS ON ONE TRAIN/TEST WINDOW
# -----------------------------------------------------------------------------
make_failed_model_result <- function(h, message) {
  list(
    pred = rep(NA_real_, h),
    status = "FAILED",
    message = as.character(message),
    note = ""
  )
}

run_nine_models <- function(y_train, int_train, int_future, h, seed) {

  y_train <- as.numeric(y_train)
  int_train <- as.numeric(int_train)
  int_future <- as.numeric(int_future)

  M1 <- run_safely(
    "M1", MODEL_LABELS["M1"], h,
    forecast_M1_ARIMA(y_train = y_train, h = h)
  )

  M2 <- run_safely(
    "M2", MODEL_LABELS["M2"], h,
    forecast_M2_IARIMA(
      y_train = y_train,
      int_train = int_train,
      int_future = int_future,
      h = h
    )
  )

  M3 <- run_safely(
    "M3", MODEL_LABELS["M3"], h,
    forecast_M3_XGB_original(y_train, h, seed + 30L)
  )

  M4 <- run_safely(
    "M4", MODEL_LABELS["M4"], h,
    forecast_M4_XGB_transformed(y_train, h, seed + 40L)
  )

  M5 <- run_safely(
    "M5", MODEL_LABELS["M5"], h,
    forecast_M5_XGB_stationary(y_train, h, seed + 50L)
  )

  M6 <- run_safely(
    "M6", MODEL_LABELS["M6"], h,
    forecast_M6_XGB_intervention(
      y_train, int_train, int_future, h, seed + 60L
    )
  )

  final_base <- list(
    M1 = list(pred = M1$pred, note = M1$note),
    M2 = list(pred = M2$pred, note = M2$note),
    M3 = list(pred = M3$pred, note = M3$note)
  )

  stack_weights <- tibble::tibble()
  stack_diagnostics <- tibble::tibble()

  stack <- tryCatch(
    {
      if (any(c(M1$status, M2$status, M3$status) != "OK")) {
        stop("A required base model failed.")
      }

      ans <- stacking_forecasts(
        y_train = y_train,
        int_train = int_train,
        final_base = final_base,
        seed = seed + 70L
      )

      stack_weights <- ans$weights
      stack_diagnostics <- ans$diagnostics

      list(
        M7 = list(pred = ans$M7, status = "OK", message = "", note = ans$note),
        M8 = list(pred = ans$M8, status = "OK", message = "", note = ans$note),
        M9 = list(pred = ans$M9, status = "OK", message = "", note = ans$note)
      )
    },
    error = function(e) {
      failed <- make_failed_model_result(h, conditionMessage(e))
      list(M7 = failed, M8 = failed, M9 = failed)
    }
  )

  models <- list(
    M1=M1, M2=M2, M3=M3, M4=M4, M5=M5, M6=M6,
    M7=stack$M7, M8=stack$M8, M9=stack$M9
  )

  status <- dplyr::bind_rows(lapply(names(models), function(id) {
    z <- models[[id]]
    tibble::tibble(
      model = id,
      model_label = unname(MODEL_LABELS[id]),
      status = z$status,
      message = z$message,
      note = z$note
    )
  }))

  forecasts <- dplyr::bind_rows(lapply(names(models), function(id) {
    tibble::tibble(
      model = id,
      model_label = unname(MODEL_LABELS[id]),
      horizon = seq_len(h),
      forecast = as.numeric(models[[id]]$pred)
    )
  }))

  list(
    models=models,
    status=status,
    forecasts=forecasts,
    stack_weights=stack_weights,
    stack_diagnostics=stack_diagnostics
  )
}


# -----------------------------------------------------------------------------
# 16. ACCURACY, RANKS AND OBSERVATION-LEVEL ERRORS
# -----------------------------------------------------------------------------
calculate_window_metrics <- function(actual, forecast_long) {

  forecast_long |>
    dplyr::group_by(model, model_label) |>
    dplyr::summarise(
      RMSE = unname(accuracy_metrics(actual, forecast)["RMSE"]),
      MAE  = unname(accuracy_metrics(actual, forecast)["MAE"]),
      MAPE = unname(accuracy_metrics(actual, forecast)["MAPE"]),
      sMAPE= unname(accuracy_metrics(actual, forecast)["sMAPE"]),
      .groups = "drop"
    ) |>
    dplyr::mutate(
      Rank_RMSE = rank(RMSE, ties.method = "average"),
      Rank_MAE  = rank(MAE, ties.method = "average"),
      Rank_MAPE = rank(MAPE, ties.method = "average"),
      Rank_sMAPE= rank(sMAPE, ties.method = "average")
    )
}

make_observation_error_table <- function(test_dates, actual, forecast_long) {

  forecast_long |>
    dplyr::mutate(
      date = test_dates[horizon],
      actual = actual[horizon],
      error = actual - forecast,
      absolute_error = abs(error),
      squared_error = error^2,
      APE = 100 * absolute_error / pmax(abs(actual), 1e-12),
      sAPE = 100 * 2 * absolute_error /
        pmax(abs(actual) + abs(forecast), 1e-12)
    )
}


# -----------------------------------------------------------------------------
# 17. MOVING-BLOCK BOOTSTRAP 95% CONFIDENCE INTERVALS
# -----------------------------------------------------------------------------
circular_block_indices <- function(n, block_length) {
  starts <- sample.int(n, ceiling(n/block_length), replace = TRUE)
  idx <- unlist(lapply(starts, function(s) {
    ((s - 1L + seq_len(block_length) - 1L) %% n) + 1L
  }), use.names = FALSE)
  idx[seq_len(n)]
}

bootstrap_accuracy_ci <- function(
    actual, pred, B = 5000L, block_length = 3L, seed = BASE_SEED
) {

  set.seed(seed)
  estimate <- accuracy_metrics(actual, pred)

  boot <- matrix(
    NA_real_, nrow = B, ncol = length(METRIC_NAMES),
    dimnames = list(NULL, METRIC_NAMES)
  )

  for (b in seq_len(B)) {
    idx <- circular_block_indices(length(actual), block_length)
    boot[b,] <- accuracy_metrics(actual[idx], pred[idx])
  }

  dplyr::bind_rows(lapply(METRIC_NAMES, function(metric) {
    v <- boot[,metric]
    tibble::tibble(
      metric = metric,
      estimate = unname(estimate[metric]),
      bootstrap_mean = mean(v, na.rm = TRUE),
      CI_2.5 = as.numeric(quantile(v, .025, na.rm=TRUE, names=FALSE, type=8)),
      CI_97.5 = as.numeric(quantile(v, .975, na.rm=TRUE, names=FALSE, type=8))
    )
  }))
}


# -----------------------------------------------------------------------------
# 18. DIEBOLD-MARIANO-HLN SENSITIVITY TEST
# -----------------------------------------------------------------------------
dm_hln_test <- function(error_a, error_b, power = 2L, h = 1L) {

  ok <- is.finite(error_a) & is.finite(error_b)
  error_a <- as.numeric(error_a[ok])
  error_b <- as.numeric(error_b[ok])
  n <- length(error_a)

  if (n < 5L) {
    return(tibble::tibble(
      n=n, mean_loss_difference=NA_real_,
      statistic=NA_real_, p_value=NA_real_
    ))
  }

  d <- abs(error_a)^power - abs(error_b)^power
  dbar <- mean(d)
  cen <- d - dbar
  lrv <- sum(cen^2)/n

  if (h > 1L) {
    for (lag in seq_len(min(h-1L, n-1L))) {
      gam <- sum(cen[(lag+1L):n] * cen[1L:(n-lag)]) / n
      lrv <- lrv + 2*gam
    }
  }

  if (!is.finite(lrv) || lrv <= .Machine$double.eps) {
    return(tibble::tibble(
      n=n, mean_loss_difference=dbar,
      statistic=NA_real_, p_value=NA_real_
    ))
  }

  dm <- dbar / sqrt(lrv/n)
  hln <- sqrt((n + 1 - 2*h + h*(h-1)/n)/n)
  stat <- dm*hln

  tibble::tibble(
    n=n,
    mean_loss_difference=dbar,
    statistic=stat,
    p_value=2*stats::pt(-abs(stat), df=n-1L)
  )
}

pairwise_dm_tests <- function(observation_errors) {

  wide <- observation_errors |>
    dplyr::select(date, model, error) |>
    tidyr::pivot_wider(names_from=model, values_from=error)

  pairs <- utils::combn(names(MODEL_LABELS), 2L, simplify=FALSE)

  out <- list()
  k <- 0L

  for (pair in pairs) {
    for (loss in c("absolute_error","squared_error")) {
      k <- k + 1L
      power <- ifelse(loss=="absolute_error",1L,2L)
      ans <- dm_hln_test(
        wide[[pair[1]]], wide[[pair[2]]], power=power, h=1L
      )
      out[[k]] <- ans |>
        dplyr::mutate(
          model_A=pair[1], label_A=unname(MODEL_LABELS[pair[1]]),
          model_B=pair[2], label_B=unname(MODEL_LABELS[pair[2]]),
          loss=loss, .before=1
        )
    }
  }

  dplyr::bind_rows(out) |>
    dplyr::group_by(loss) |>
    dplyr::mutate(
      p_value_BH=stats::p.adjust(p_value, method="BH")
    ) |>
    dplyr::ungroup()
}


# -----------------------------------------------------------------------------
# 19. ERROR BREAKDOWN BY OBSERVATION MAGNITUDE
# -----------------------------------------------------------------------------
make_magnitude_breakdown <- function(observation_errors) {

  magnitude_lookup <- observation_errors |>
    dplyr::distinct(date, actual) |>
    dplyr::arrange(actual) |>
    dplyr::mutate(
      nt = dplyr::ntile(actual,3L),
      magnitude_group = dplyr::case_when(
        nt==1L ~ "Low actual",
        nt==2L ~ "Medium actual",
        TRUE ~ "High actual"
      )
    )

  observation_errors |>
    dplyr::left_join(
      magnitude_lookup |> dplyr::select(date,magnitude_group),
      by="date"
    ) |>
    dplyr::group_by(model,model_label,magnitude_group) |>
    dplyr::summarise(
      n=dplyr::n(),
      actual_min=min(actual),
      actual_mean=mean(actual),
      actual_max=max(actual),
      RMSE=sqrt(mean(squared_error)),
      MAE=mean(absolute_error),
      MAPE=mean(APE),
      sMAPE=mean(sAPE),
      .groups="drop"
    )
}


# -----------------------------------------------------------------------------
# 20. ARIMA / I-ARIMA DETAILS
# -----------------------------------------------------------------------------
extract_arima_details <- function(y_train, int_train=NULL, model_id) {

  lambda <- estimate_lambda(y_train)
  z <- boxcox_transform(y_train,lambda)

  if (is.null(int_train)) {
    selected <- fit_best_arima_base(z=z)
  } else {
    xreg <- matrix(
      as.numeric(int_train), ncol=1L,
      dimnames=list(NULL,"intervention")
    )
    selected <- fit_best_arima_base(z=z,xreg=xreg)
  }

  fit <- selected$fit
  co <- stats::coef(fit)
  vc <- fit$var.coef
  se <- if (is.null(vc)) rep(NA_real_,length(co)) else sqrt(diag(vc))
  zval <- co/se
  pval <- 2*stats::pnorm(-abs(zval))

  list(
    order=tibble::tibble(
      model=model_id,
      model_label=unname(MODEL_LABELS[model_id]),
      BoxCox_lambda=lambda,
      ARIMA_p=selected$order[1],
      ARIMA_d=selected$order[2],
      ARIMA_q=selected$order[3],
      AICc=selected$aicc
    ),
    coefficients=tibble::tibble(
      model=model_id,
      model_label=unname(MODEL_LABELS[model_id]),
      BoxCox_lambda=lambda,
      ARIMA_p=selected$order[1],
      ARIMA_d=selected$order[2],
      ARIMA_q=selected$order[3],
      AICc=selected$aicc,
      term=names(co),
      estimate=as.numeric(co),
      standard_error=as.numeric(se),
      z_value=as.numeric(zval),
      p_value=as.numeric(pval),
      CI_2.5=as.numeric(co-1.96*se),
      CI_97.5=as.numeric(co+1.96*se),
      scale="Box-Cox transformed scale"
    )
  )
}


# -----------------------------------------------------------------------------
# 21. HYPERPARAMETER TABLE
# -----------------------------------------------------------------------------
make_xgb_hyperparameter_table <- function() {
  tibble::tibble(
    parameter=c(
      "objective","eta","max_depth","min_child_weight","subsample",
      "colsample_bytree","gamma","lambda (L2)","alpha (L1)",
      "tree_method","max_bin","nrounds","maximum PACF lag",
      "lag-selection rule","hyperparameter tuning"
    ),
    value=as.character(c(
      XGB_PARAMS$objective,XGB_PARAMS$eta,XGB_PARAMS$max_depth,
      XGB_PARAMS$min_child_weight,XGB_PARAMS$subsample,
      XGB_PARAMS$colsample_bytree,XGB_PARAMS$gamma,
      XGB_PARAMS$lambda,XGB_PARAMS$alpha,XGB_PARAMS$tree_method,
      XGB_PARAMS$max_bin,XGB_NROUNDS,MAX_PACF_LAG,
      "Largest significant PACF lag within maximum 12; fallback lag 1",
      "Held fixed across XGBoost-containing models"
    ))
  )
}


# -----------------------------------------------------------------------------
# 22. PRIMARY 2021 HOLD-OUT ANALYSIS
# -----------------------------------------------------------------------------
run_primary_2021_analysis <- function() {

  model_output <- run_nine_models(
    y_train=train_data$y,
    int_train=train_data$intervention,
    int_future=test_data$intervention,
    h=12L,
    seed=BASE_SEED+2021L
  )

  forecasts <- model_output$forecasts |>
    dplyr::mutate(
      date=test_data$date[horizon],
      actual=test_data$y[horizon],
      .after=model_label
    )

  metrics <- calculate_window_metrics(
    test_data$y, model_output$forecasts
  )

  observation_errors <- make_observation_error_table(
    test_data$date, test_data$y, model_output$forecasts
  )

  magnitude_breakdown <- make_magnitude_breakdown(observation_errors)

  bootstrap_ci <- dplyr::bind_rows(
    lapply(names(MODEL_LABELS), function(id) {
      pred <- forecasts |>
        dplyr::filter(model==id) |>
        dplyr::arrange(horizon) |>
        dplyr::pull(forecast)

      bootstrap_accuracy_ci(
        test_data$y, pred, B=5000L, block_length=3L,
        seed=BASE_SEED+match(id,names(MODEL_LABELS))
      ) |>
        dplyr::mutate(
          model=id, model_label=unname(MODEL_LABELS[id]), .before=1
        )
    })
  )

  dm_all <- pairwise_dm_tests(observation_errors)

  dm_vs_M5 <- dm_all |>
    dplyr::filter(model_A=="M5" | model_B=="M5") |>
    dplyr::mutate(
      competitor=ifelse(model_A=="M5",model_B,model_A),
      competitor_label=unname(MODEL_LABELS[competitor]),
      competitor_minus_M5_loss=ifelse(
        model_B=="M5",mean_loss_difference,-mean_loss_difference
      ),
      interpretation=ifelse(
        competitor_minus_M5_loss>0,
        "Positive: M5 had lower mean loss",
        "Negative: competitor had lower mean loss"
      )
    )

  a1 <- extract_arima_details(train_data$y,NULL,"M1")
  a2 <- extract_arima_details(
    train_data$y,train_data$intervention,"M2"
  )

  list(
    model_output=model_output,
    forecasts=forecasts,
    metrics=metrics,
    observation_errors=observation_errors,
    magnitude_breakdown=magnitude_breakdown,
    bootstrap_ci=bootstrap_ci,
    dm_all=dm_all,
    dm_vs_M5=dm_vs_M5,
    arima_orders=dplyr::bind_rows(a1$order,a2$order),
    arima_coefficients=dplyr::bind_rows(a1$coefficients,a2$coefficients),
    xgb_hyperparameters=make_xgb_hyperparameter_table()
  )
}


# -----------------------------------------------------------------------------
# 23. REPEATED ONE-YEAR-AHEAD BACKTESTS: 2017-2021
# -----------------------------------------------------------------------------
run_one_backtest_year <- function(test_year) {

  train_window <- study_data |>
    dplyr::filter(
      date <= as.Date(sprintf("%d-12-01",test_year-1L))
    )

  test_window <- study_data |>
    dplyr::filter(
      date >= as.Date(sprintf("%d-01-01",test_year)),
      date <= as.Date(sprintf("%d-12-01",test_year))
    )

  stopifnot(nrow(test_window)==12L)

  z <- run_nine_models(
    train_window$y,
    train_window$intervention,
    test_window$intervention,
    12L,
    BASE_SEED+test_year
  )

  metrics <- calculate_window_metrics(
    test_window$y,z$forecasts
  ) |>
    dplyr::mutate(
      test_year=test_year,
      window_type=ifelse(
        test_year==2021L,
        "Primary held-out test",
        "Historical robustness backtest"
      ),
      train_n=nrow(train_window),
      .before=1
    )

  forecasts <- z$forecasts |>
    dplyr::mutate(
      test_year=test_year,
      date=test_window$date[horizon],
      actual=test_window$y[horizon],
      .before=1
    )

  list(
    metrics=metrics,
    forecasts=forecasts,
    status=z$status |> dplyr::mutate(test_year=test_year,.before=1),
    weights=z$stack_weights |> dplyr::mutate(test_year=test_year,.before=1)
  )
}

run_repeated_backtest <- function() {

  years <- 2017:2021
  all <- lapply(years,function(yr) {
    cat("  Backtest year ",yr," ... ",sep="")
    ans <- run_one_backtest_year(yr)
    cat("OK\n")
    ans
  })

  metrics <- dplyr::bind_rows(lapply(all, function(x) x$metrics))
  forecasts <- dplyr::bind_rows(lapply(all, function(x) x$forecasts))
  status <- dplyr::bind_rows(lapply(all, function(x) x$status))
  weights <- dplyr::bind_rows(lapply(all, function(x) x$weights))

  long <- metrics |>
    tidyr::pivot_longer(
      cols=dplyr::all_of(METRIC_NAMES),
      names_to="metric",
      values_to="value"
    )

  summary <- long |>
    dplyr::group_by(model,model_label,metric) |>
    dplyr::summarise(
      windows=dplyr::n(),
      mean=mean(value,na.rm=TRUE),
      median=median(value,na.rm=TRUE),
      sd=sd(value,na.rm=TRUE),
      min=min(value,na.rm=TRUE),
      max=max(value,na.rm=TRUE),
      .groups="drop"
    )

  winners <- long |>
    dplyr::group_by(test_year,metric) |>
    dplyr::mutate(
      best=min(value,na.rm=TRUE),
      is_winner=abs(value-best)<=1e-12,
      win_credit=ifelse(is_winner,1/sum(is_winner),0)
    ) |>
    dplyr::filter(is_winner) |>
    dplyr::ungroup() |>
    dplyr::group_by(model,model_label,metric) |>
    dplyr::summarise(
      annual_win_credit=sum(win_credit),
      years_with_strict_or_shared_win=dplyr::n(),
      .groups="drop"
    )

  list(
    metrics=metrics,forecasts=forecasts,status=status,weights=weights,
    summary=summary,winners=winners
  )
}


# -----------------------------------------------------------------------------
# 24. OPTIONAL EMPIRICAL-VS-MONTE-CARLO COMPARISON
# -----------------------------------------------------------------------------
compare_with_simulation <- function(empirical_metrics,empirical_weights) {

  metric_file <- file.path(
    SIM_SUMMARY_DIR,"02_overall_metric_summary.csv"
  )
  rank_file <- file.path(
    SIM_SUMMARY_DIR,"05_overall_rank_summary.csv"
  )
  win_file <- file.path(
    SIM_SUMMARY_DIR,"04_overall_win_rates.csv"
  )
  stack_file <- file.path(
    SIM_SUMMARY_DIR,"10_stack_weight_summary.csv"
  )

  metric_comparison <- tibble::tibble()
  stack_comparison <- tibble::tibble()

  if (file.exists(metric_file)) {

    sim <- readr::read_csv(
      metric_file,show_col_types=FALSE,progress=FALSE
    )

    empirical_long <- empirical_metrics |>
      dplyr::select(model,model_label,dplyr::all_of(METRIC_NAMES)) |>
      tidyr::pivot_longer(
        cols=dplyr::all_of(METRIC_NAMES),
        names_to="metric",
        values_to="empirical_2021_value"
      )

    metric_comparison <- empirical_long |>
      dplyr::left_join(
        sim |>
          dplyr::select(
            model,metric,
            simulation_mean=mean_value,
            simulation_median=median_value,
            simulation_empirical_2.5=empirical_2.5,
            simulation_empirical_97.5=empirical_97.5
          ),
        by=c("model","metric")
      )

    if (file.exists(rank_file)) {
      ranks <- readr::read_csv(
        rank_file,show_col_types=FALSE,progress=FALSE
      )
      metric_comparison <- metric_comparison |>
        dplyr::left_join(
          ranks |>
            dplyr::select(
              model,metric,
              simulation_mean_rank=mean_rank,
              simulation_median_rank=median_rank
            ),
          by=c("model","metric")
        )
    }

    if (file.exists(win_file)) {
      wins <- readr::read_csv(
        win_file,show_col_types=FALSE,progress=FALSE
      )
      metric_comparison <- metric_comparison |>
        dplyr::left_join(
          wins |>
            dplyr::select(
              model,metric,
              simulation_mean_win_credit=mean_win_credit,
              simulation_win_percent=strict_or_shared_win_percent
            ),
          by=c("model","metric")
        )
    }
  }

  if (file.exists(stack_file) && nrow(empirical_weights)>0L) {
    sw <- readr::read_csv(
      stack_file,show_col_types=FALSE,progress=FALSE
    )
    stack_comparison <- empirical_weights |>
      dplyr::rename(empirical_2021_weight=weight) |>
      dplyr::left_join(
        sw |>
          dplyr::select(
            stack_model,base_model,
            simulation_mean_weight=mean_weight,
            simulation_median_weight=median_weight,
            simulation_weight_2.5=empirical_2.5,
            simulation_weight_97.5=empirical_97.5
          ),
        by=c("stack_model","base_model")
      )
  }

  list(
    metric_comparison=metric_comparison,
    stack_comparison=stack_comparison
  )
}


# -----------------------------------------------------------------------------
# 25. FIGURES
# -----------------------------------------------------------------------------
make_figures <- function(primary,backtest) {

  p1 <- ggplot2::ggplot(
    study_data,ggplot2::aes(date,y)
  ) +
    ggplot2::geom_line(linewidth=.7) +
    ggplot2::geom_vline(
      xintercept=as.numeric(as.Date("2016-01-01")),
      linetype="dashed"
    ) +
    ggplot2::geom_vline(
      xintercept=as.numeric(as.Date("2021-01-01")),
      linetype="dotted"
    ) +
    ggplot2::labs(
      title="Monthly power-generation series",
      subtitle="Dashed: January 2016 intervention; dotted: 2021 held-out test",
      x="Date",y="Power generation"
    ) +
    ggplot2::theme_minimal()

  ggplot2::ggsave(
    file.path(RESULT_DIR,"Figure_1_full_series_intervention_test_split.png"),
    p1,width=10,height=5.5,dpi=300
  )

  p2data <- dplyr::bind_rows(
    test_data |>
      dplyr::transmute(date,series="Actual",value=y),
    primary$forecasts |>
      dplyr::transmute(
        date,
        series=paste(model,model_label),
        value=forecast
      )
  )

  p2 <- ggplot2::ggplot(
    p2data,
    ggplot2::aes(date,value,group=series,colour=series)
  ) +
    ggplot2::geom_line(linewidth=.7) +
    ggplot2::labs(
      title="Actual and forecast values for the 2021 held-out test",
      x="Month",y="Power generation",colour="Series"
    ) +
    ggplot2::theme_minimal() +
    ggplot2::theme(legend.position="bottom")

  ggplot2::ggsave(
    file.path(RESULT_DIR,"Figure_2_2021_actual_vs_forecasts.png"),
    p2,width=12,height=7,dpi=300
  )

  p3 <- ggplot2::ggplot(
    primary$magnitude_breakdown,
    ggplot2::aes(magnitude_group,MAPE,group=model)
  ) +
    ggplot2::geom_line(linewidth=.6) +
    ggplot2::geom_point(size=1.5) +
    ggplot2::facet_wrap(~model,scales="free_y") +
    ggplot2::labs(
      title="2021 MAPE by observed-value magnitude",
      x="Observed-value magnitude group",y="MAPE (%)"
    ) +
    ggplot2::theme_minimal()

  ggplot2::ggsave(
    file.path(RESULT_DIR,"Figure_3_MAPE_by_observation_magnitude.png"),
    p3,width=11,height=8,dpi=300
  )

  p4 <- ggplot2::ggplot(
    backtest$metrics,
    ggplot2::aes(test_year,sMAPE,group=model,colour=model)
  ) +
    ggplot2::geom_line(linewidth=.7) +
    ggplot2::geom_point(size=1.6) +
    ggplot2::scale_x_continuous(breaks=2017:2021) +
    ggplot2::labs(
      title="Repeated one-year-ahead backtests",
      subtitle="2021 is the primary held-out test",
      x="Test year",y="sMAPE (%)",colour="Model"
    ) +
    ggplot2::theme_minimal() +
    ggplot2::theme(legend.position="bottom")

  ggplot2::ggsave(
    file.path(RESULT_DIR,"Figure_4_repeated_backtest_sMAPE.png"),
    p4,width=11,height=7,dpi=300
  )
}


# -----------------------------------------------------------------------------
# 26. SAVE OUTPUTS
# -----------------------------------------------------------------------------
save_outputs <- function(primary,backtest,comparison) {

  files <- list(
    "01_primary_2021_accuracy_metrics.csv"=primary$metrics,
    "02_primary_2021_forecasts.csv"=primary$forecasts,
    "03_primary_model_status_and_notes.csv"=primary$model_output$status,
    "04_xgboost_hyperparameters.csv"=primary$xgb_hyperparameters,
    "05_primary_stacking_weights.csv"=primary$model_output$stack_weights,
    "06_primary_stacking_diagnostics.csv"=primary$model_output$stack_diagnostics,
    "07_primary_metric_bootstrap_95CI.csv"=primary$bootstrap_ci,
    "08_primary_DM_HLN_tests_vs_M5.csv"=primary$dm_vs_M5,
    "09_primary_all_pairwise_DM_HLN_tests.csv"=primary$dm_all,
    "10_error_by_observation_magnitude.csv"=primary$magnitude_breakdown,
    "11_observation_level_2021_errors.csv"=primary$observation_errors,
    "12_ARIMA_selected_orders.csv"=primary$arima_orders,
    "13_ARIMA_IARIMA_coefficients.csv"=primary$arima_coefficients,
    "14_repeated_annual_backtest_metrics.csv"=backtest$metrics,
    "15_repeated_annual_backtest_summary.csv"=backtest$summary,
    "16_repeated_annual_backtest_winners.csv"=backtest$winners,
    "17_repeated_backtest_model_status.csv"=backtest$status,
    "18_repeated_backtest_stacking_weights.csv"=backtest$weights
  )

  for (nm in names(files)) {
    write_csv_quiet(files[[nm]],file.path(RESULT_DIR,nm))
  }

  if (nrow(comparison$metric_comparison)>0L) {
    write_csv_quiet(
      comparison$metric_comparison,
      file.path(RESULT_DIR,"19_empirical_vs_simulation_metrics.csv")
    )
  }

  if (nrow(comparison$stack_comparison)>0L) {
    write_csv_quiet(
      comparison$stack_comparison,
      file.path(RESULT_DIR,"20_empirical_vs_simulation_stacking_weights.csv")
    )
  }

  capture.output(
    sessionInfo(),
    file=file.path(RESULT_DIR,"21_sessionInfo.txt")
  )

  saveRDS(
    list(primary=primary,backtest=backtest,comparison=comparison),
    file.path(RESULT_DIR,"empirical_reanalysis_complete.rds")
  )
}


# -----------------------------------------------------------------------------
# 27. PRINT MAIN RESULTS
# -----------------------------------------------------------------------------
print_reanalysis_results <- function(results) {

  cat("\n============================================================\n")
  cat(" EMPIRICAL RE-ANALYSIS RESULTS\n")
  cat("============================================================\n")
  cat("Training: 2010-2020 (132 months)\n")
  cat("Test:     2021 (12 months)\n")
  cat("Break:    January 2016 step intervention\n")
  cat("============================================================\n\n")

  cat("PRIMARY MODEL STATUS\n")
  print(results$primary$model_output$status,n=Inf,width=Inf)

  cat("\nPRIMARY 2021 FORECAST ACCURACY\n")
  print(
    results$primary$metrics |> dplyr::arrange(sMAPE),
    n=Inf,width=Inf
  )

  cat("\n95% MOVING-BLOCK BOOTSTRAP INTERVALS\n")
  print(
    results$primary$bootstrap_ci |>
      dplyr::arrange(metric,estimate),
    n=Inf,width=Inf
  )

  cat("\nSTACKING WEIGHTS\n")
  print(results$primary$model_output$stack_weights,n=Inf,width=Inf)

  cat("\nARIMA / I-ARIMA SELECTED ORDERS\n")
  print(results$primary$arima_orders,n=Inf,width=Inf)

  cat("\nDM-HLN TESTS AGAINST M5\n")
  print(
    results$primary$dm_vs_M5 |> dplyr::arrange(loss,p_value),
    n=Inf,width=Inf
  )

  cat("\nERROR BY 2021 OBSERVATION MAGNITUDE\n")
  print(
    results$primary$magnitude_breakdown |>
      dplyr::arrange(magnitude_group,MAPE),
    n=Inf,width=Inf
  )

  cat("\nREPEATED 2017-2021 BACKTEST SUMMARY\n")
  print(
    results$backtest$summary |>
      dplyr::filter(metric %in% c("MAPE","sMAPE")) |>
      dplyr::arrange(metric,mean),
    n=Inf,width=Inf
  )

  cat("\nREPEATED BACKTEST YEAR-WIN COUNTS\n")
  print(
    results$backtest$winners |>
      dplyr::arrange(metric,dplyr::desc(annual_win_credit)),
    n=Inf,width=Inf
  )

  if (nrow(results$comparison$metric_comparison)>0L) {
    cat("\nEMPIRICAL 2021 VS MONTE CARLO VERSION 5\n")
    print(
      results$comparison$metric_comparison |>
        dplyr::filter(metric %in% c("MAPE","sMAPE")) |>
        dplyr::arrange(metric,empirical_2021_value),
      n=Inf,width=Inf
    )
  } else {
    cat("\nMonte Carlo summary folder not found at:\n  ")
    cat(normalizePath(SIM_SUMMARY_DIR,mustWork=FALSE),"\n")
  }

  cat("\nOutputs saved in:\n  ",normalizePath(RESULT_DIR),"\n",sep="")
  cat("============================================================\n")

  invisible(results)
}


# -----------------------------------------------------------------------------
# 28. MASTER FUNCTION
# -----------------------------------------------------------------------------
run_empirical_reanalysis <- function() {

  cat("\n============================================================\n")
  cat(" EMPIRICAL STUDY RE-ANALYSIS\n")
  cat("============================================================\n")
  cat("Primary training: 2010-2020\n")
  cat("Primary test:     2021\n")
  cat("Intervention:     January 2016 step\n")
  cat("Models:           M1-M9 matched to Monte Carlo V5\n")
  cat("============================================================\n\n")

  cat("Running primary 2021 held-out analysis ...\n")
  primary <- run_primary_2021_analysis()

  cat("Running repeated annual backtests 2017-2021 ...\n")
  backtest <- run_repeated_backtest()

  comparison <- compare_with_simulation(
    primary$metrics,
    primary$model_output$stack_weights
  )

  save_outputs(primary,backtest,comparison)
  make_figures(primary,backtest)

  results <- list(
    primary=primary,
    backtest=backtest,
    comparison=comparison
  )

  cat("\nAnalysis completed.\n")
  print_reanalysis_results(results)

  invisible(results)
}


# -----------------------------------------------------------------------------
# 29. START MESSAGE
# -----------------------------------------------------------------------------
cat("\n============================================================\n")
cat(" EMPIRICAL RE-ANALYSIS SCRIPT LOADED SUCCESSFULLY\n")
cat("============================================================\n")
cat("Training:     January 2010 - December 2020\n")
cat("Test:         January 2021 - December 2021\n")
cat("Intervention: January 2016 step\n")
cat("Models:       M1-M9 matched to Monte Carlo Version 5\n")
cat("============================================================\n")
cat("\nRun:\n")
cat("  empirical_results <- run_empirical_reanalysis()\n")
cat("\nPrint again later with:\n")
cat("  print_reanalysis_results(empirical_results)\n")
cat("============================================================\n")
