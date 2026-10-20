# ============================================================================
# MONTE CARLO EXPERIMENT FOR INTERRUPTED TIME-SERIES FORECASTING
# Nine-model comparison used in the manuscript:
# M1  ARIMA
# M2  Intervention-ARIMA (I-ARIMA)
# M3  XGBoost on original series
# M4  XGBoost on Box-Cox transformed series
# M5  XGBoost on stationary (Box-Cox transformed + first-differenced) series
# M6  XGBoost + intervention regressor
# M7  Stacking: ARIMA + XGBoost
# M8  Stacking: I-ARIMA + XGBoost
# M9  Stacking: ARIMA + I-ARIMA + XGBoost
#
# Simulation factors:
#   intervention type: step, pulse, ramp
#   intervention magnitude: small, moderate, large
#   series length: 120, 240, 480
#   noise level: low, moderate, high
#   intervention position: 25%, 50%, 75%
#   underlying dynamics: linear, nonlinear
#
# Full factorial = 3*3*3*3*3*2 = 486 scenarios.
# At 250 replications = 121,500 scenario-replications.
# At 500 replications = 243,000 scenario-replications.
# ============================================================================

# ---------------------------
# 0. Packages
# ---------------------------
required_pkgs <- c(
  "forecast", "xgboost", "dplyr", "tidyr",
  "future", "future.apply", "readr"
)

missing_pkgs <- required_pkgs[!vapply(required_pkgs, requireNamespace,
                                     quietly = TRUE, FUN.VALUE = logical(1))]
if (length(missing_pkgs) > 0L) {
  stop(
    "Install these packages first: ",
    paste(missing_pkgs, collapse = ", "),
    "\nExample: install.packages(c(",
    paste(sprintf('"%s"', missing_pkgs), collapse = ", "), "))"
  )
}

suppressPackageStartupMessages({
  library(forecast)
  library(xgboost)
  library(dplyr)
  library(tidyr)
  library(future)
  library(future.apply)
  library(readr)
})

# ---------------------------
# 1. Global settings
# ---------------------------
BASE_SEED <- 20260922L
RNGkind("L'Ecuyer-CMRG")
set.seed(BASE_SEED)

# IMPORTANT:
# Start with N_REP <- 5 or 10 to debug the full pipeline.
# Use N_REP <- 250 for the main experiment; increase to 500 if computing permits.
N_REP <- 250L

TEST_FRAC <- 0.20       # final 20% of each series is the test window
META_FRAC <- 0.20       # final 20% of training data used for stacking meta-learner
MAX_PACF_LAG <- 12L     # monthly-series lag cap
CLIP_POSITIVE <- TRUE   # appropriate for power-generation-like positive outcomes

# Outer parallelization: XGBoost is restricted to 1 thread to avoid oversubscription.
N_WORKERS <- max(1L, future::availableCores() - 1L)
future::plan(future::multisession, workers = N_WORKERS)

RESULT_DIR <- "mc_results"
dir.create(RESULT_DIR, showWarnings = FALSE, recursive = TRUE)

# Fixed XGBoost hyperparameters across M3-M6 and across scenarios.
# This isolates the effect of preprocessing/model structure rather than allowing
# model-specific tuning to confound the comparison.
XGB_PARAMS <- list(
  objective = "reg:squarederror",
  eta = 0.05,
  max_depth = 3L,
  min_child_weight = 1,
  subsample = 0.80,
  colsample_bytree = 0.80,
  gamma = 0,
  lambda = 1,
  alpha = 0,
  nthread = 1L,
  verbosity = 0L
)
XGB_NROUNDS <- 250L

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

# ---------------------------
# 2. Scenario grid
# ---------------------------
# Intervention magnitudes are defined on the latent log scale so they have a
# direct multiplicative interpretation on the original positive series:
# small    = 10% level effect
# moderate = 30% level effect
# large    = 60% level effect
magnitude_tbl <- tibble::tibble(
  magnitude = c("small", "moderate", "large"),
  effect_log = log(c(1.10, 1.30, 1.60))
)

# Innovation standard deviation on the latent log scale.
noise_tbl <- tibble::tibble(
  noise_level = c("low", "moderate", "high"),
  noise_sd = c(0.03, 0.08, 0.15)
)

scenarios <- tidyr::expand_grid(
  intervention_type = c("step", "pulse", "ramp"),
  magnitude = c("small", "moderate", "large"),
  n = c(120L, 240L, 480L),
  noise_level = c("low", "moderate", "high"),
  intervention_position = c(0.25, 0.50, 0.75),
  dynamics = c("linear", "nonlinear")
) %>%
  left_join(magnitude_tbl, by = "magnitude") %>%
  left_join(noise_tbl, by = "noise_level") %>%
  mutate(
    scenario_id = sprintf("S%03d", row_number()),
    intervention_pct = paste0(as.integer(100 * intervention_position), "%")
  ) %>%
  select(
    scenario_id, intervention_type, magnitude, effect_log,
    n, noise_level, noise_sd,
    intervention_position, intervention_pct, dynamics
  )

stopifnot(nrow(scenarios) == 486L)

readr::write_csv(scenarios, file.path(RESULT_DIR, "scenario_grid.csv"))

# ---------------------------
# 3. Utility functions
# ---------------------------
boxcox_transform <- function(y, lambda) {
  stopifnot(all(is.finite(y)), all(y > 0))
  if (abs(lambda) < 1e-8) {
    log(y)
  } else {
    (y^lambda - 1) / lambda
  }
}

inv_boxcox <- function(z, lambda) {
  if (abs(lambda) < 1e-8) {
    out <- exp(z)
  } else {
    # Keep the inverse inside the Box-Cox domain.
    base <- 1 + lambda * z
    base <- pmax(base, 1e-10)
    out <- base^(1 / lambda)
  }
  as.numeric(out)
}

estimate_lambda <- function(y) {
  # log-likelihood method is reproducible and suitable for non-seasonal use here.
  as.numeric(forecast::BoxCox.lambda(
    y,
    method = "loglik",
    lower = -1,
    upper = 2
  ))
}

make_intervention <- function(n, type, position) {
  stopifnot(type %in% c("step", "pulse", "ramp"))
  t0 <- as.integer(round(position * n))
  t0 <- max(2L, min(n, t0))

  x <- numeric(n)

  if (type == "step") {
    x[t0:n] <- 1
  } else if (type == "pulse") {
    x[t0] <- 1
  } else if (type == "ramp") {
    # Normalized ramp: starts at the interruption and reaches 1 by the final time.
    len <- n - t0 + 1L
    x[t0:n] <- seq_len(len) / len
  }

  list(x = x, t0 = t0)
}

# ---------------------------
# 4. Data-generating processes
# ---------------------------
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
  stopifnot(dynamics %in% c("linear", "nonlinear"))

  # Stationary disturbance around a deterministic latent trend.
  # After log transformation, the trend induces non-stationarity in mean;
  # first differencing produces an approximately stationary representation.
  # Exponentiation produces a positive series with level-dependent variance,
  # making Box-Cox/log transformation relevant.
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
  time_index <- seq_len(n)

  int_obj <- make_intervention(
    n = n,
    type = intervention_type,
    position = intervention_position
  )

  latent_log <- log(base_level) + drift * time_index + u +
    effect_log * int_obj$x

  y <- exp(latent_log)

  list(
    y = as.numeric(y),
    intervention = int_obj$x,
    t0 = int_obj$t0,
    latent_log = latent_log
  )
}

# ---------------------------
# 5. Accuracy metrics
# ---------------------------
accuracy_metrics <- function(actual, pred) {
  ok <- is.finite(actual) & is.finite(pred)
  actual <- actual[ok]
  pred <- pred[ok]

  if (length(actual) == 0L) {
    return(c(RMSE = NA_real_, MAE = NA_real_, MAPE = NA_real_, sMAPE = NA_real_))
  }

  e <- actual - pred
  rmse <- sqrt(mean(e^2))
  mae <- mean(abs(e))

  # MAPE is reported explicitly as percentage (0-100 scale), not proportion.
  nz <- abs(actual) > 1e-12
  mape <- if (any(nz)) {
    100 * mean(abs(e[nz] / actual[nz]))
  } else {
    NA_real_
  }

  denom <- abs(actual) + abs(pred)
  smape <- 100 * mean(2 * abs(e) / pmax(denom, 1e-12))

  c(RMSE = rmse, MAE = mae, MAPE = mape, sMAPE = smape)
}

clip_positive <- function(x) {
  if (CLIP_POSITIVE) pmax(as.numeric(x), 1e-8) else as.numeric(x)
}

# ---------------------------
# 6. PACF-based lag selection
# ---------------------------
select_pacf_order <- function(x, max_lag = MAX_PACF_LAG) {
  x <- as.numeric(x)
  max_lag <- min(max_lag, floor(length(x) / 4), length(x) - 2L)
  if (max_lag < 1L) return(1L)

  pcf <- tryCatch(
    stats::pacf(x, plot = FALSE, lag.max = max_lag)$acf,
    error = function(e) NULL
  )

  if (is.null(pcf)) return(1L)

  pcf <- as.numeric(pcf)
  threshold <- 1.96 / sqrt(length(x))
  sig <- which(abs(pcf) > threshold)

  # If PACF identifies significant lags, use all lags 1,...,p where p is the
  # largest significant lag. If no lag is significant, use lag 1.
  p <- if (length(sig) == 0L) 1L else max(sig)
  max(1L, min(as.integer(p), max_lag))
}

make_lagged_xy <- function(series, p, xreg = NULL) {
  n <- length(series)
  if (n <= p + 2L) stop("Series too short for selected lag order.")

  idx <- (p + 1L):n
  X <- sapply(seq_len(p), function(j) series[idx - j])
  X <- as.matrix(X)
  colnames(X) <- paste0("lag", seq_len(p))

  if (!is.null(xreg)) {
    X <- cbind(X, intervention = as.numeric(xreg[idx]))
  }

  list(X = X, y = as.numeric(series[idx]))
}

# ---------------------------
# 7. Recursive XGBoost forecaster
# ---------------------------
xgb_recursive_forecast <- function(
    train_series,
    h,
    xreg_train = NULL,
    xreg_future = NULL,
    seed = 1L
) {
  train_series <- as.numeric(train_series)
  p <- select_pacf_order(train_series)
  dat <- make_lagged_xy(train_series, p = p, xreg = xreg_train)

  set.seed(seed)
  dtrain <- xgboost::xgb.DMatrix(data = dat$X, label = dat$y)
  fit <- xgboost::xgb.train(
    params = XGB_PARAMS,
    data = dtrain,
    nrounds = XGB_NROUNDS,
    verbose = 0
  )

  history <- train_series
  out <- numeric(h)

  for (k in seq_len(h)) {
    lag_values <- vapply(
      seq_len(p),
      function(j) history[length(history) - j + 1L],
      FUN.VALUE = numeric(1)
    )

    newx <- matrix(lag_values, nrow = 1L)

    if (!is.null(xreg_train)) {
      if (is.null(xreg_future) || length(xreg_future) < h) {
        stop("Future intervention regressor is missing or too short.")
      }
      newx <- cbind(newx, as.numeric(xreg_future[k]))
    }

    pred <- as.numeric(stats::predict(fit, newx))
    out[k] <- pred
    history <- c(history, pred)
  }

  list(pred = out, p = p)
}

# ---------------------------
# 8. Model-specific forecast functions
# ---------------------------
forecast_M1_ARIMA <- function(y_train, h) {
  lambda <- estimate_lambda(y_train)
  z <- boxcox_transform(y_train, lambda)

  fit <- forecast::auto.arima(
    z,
    seasonal = FALSE,
    ic = "aicc",
    stepwise = TRUE,
    approximation = FALSE,
    max.p = 5,
    max.q = 5,
    max.d = 2
  )

  zhat <- as.numeric(forecast::forecast(fit, h = h)$mean)
  clip_positive(inv_boxcox(zhat, lambda))
}

forecast_M2_IARIMA <- function(y_train, int_train, int_future, h) {
  lambda <- estimate_lambda(y_train)
  z <- boxcox_transform(y_train, lambda)

  xr_train <- matrix(as.numeric(int_train), ncol = 1L,
                     dimnames = list(NULL, "intervention"))
  xr_future <- matrix(as.numeric(int_future), ncol = 1L,
                      dimnames = list(NULL, "intervention"))

  fit <- forecast::auto.arima(
    z,
    xreg = xr_train,
    seasonal = FALSE,
    ic = "aicc",
    stepwise = TRUE,
    approximation = FALSE,
    max.p = 5,
    max.q = 5,
    max.d = 2
  )

  zhat <- as.numeric(forecast::forecast(
    fit,
    h = h,
    xreg = xr_future
  )$mean)

  clip_positive(inv_boxcox(zhat, lambda))
}

forecast_M3_XGB_original <- function(y_train, h, seed) {
  ans <- xgb_recursive_forecast(
    train_series = y_train,
    h = h,
    seed = seed
  )
  clip_positive(ans$pred)
}

forecast_M4_XGB_transformed <- function(y_train, h, seed) {
  lambda <- estimate_lambda(y_train)
  z <- boxcox_transform(y_train, lambda)

  ans <- xgb_recursive_forecast(
    train_series = z,
    h = h,
    seed = seed
  )

  clip_positive(inv_boxcox(ans$pred, lambda))
}

forecast_M5_XGB_stationary <- function(y_train, h, seed) {
  lambda <- estimate_lambda(y_train)
  z <- boxcox_transform(y_train, lambda)
  dz <- diff(z)

  ans <- xgb_recursive_forecast(
    train_series = dz,
    h = h,
    seed = seed
  )

  # Reconstruct transformed levels, then invert Box-Cox.
  zhat <- tail(z, 1L) + cumsum(ans$pred)
  clip_positive(inv_boxcox(zhat, lambda))
}

forecast_M6_XGB_intervention <- function(
    y_train,
    int_train,
    int_future,
    h,
    seed
) {
  # Operational definition used here:
  # Box-Cox-transformed target + lagged transformed target values + known
  # intervention regressor. This should be retained in the manuscript if it
  # matches the authors' intended Model 6 definition.
  lambda <- estimate_lambda(y_train)
  z <- boxcox_transform(y_train, lambda)

  ans <- xgb_recursive_forecast(
    train_series = z,
    h = h,
    xreg_train = int_train,
    xreg_future = int_future,
    seed = seed
  )

  clip_positive(inv_boxcox(ans$pred, lambda))
}

safe_forecast <- function(expr, h) {
  tryCatch(
    as.numeric(expr),
    error = function(e) rep(NA_real_, h)
  )
}

# ---------------------------
# 9. Base forecasts used by stacking
# ---------------------------
base_forecasts <- function(y_train, int_train, int_future, h, seed) {
  list(
    M1 = safe_forecast(forecast_M1_ARIMA(y_train, h), h),
    M2 = safe_forecast(
      forecast_M2_IARIMA(y_train, int_train, int_future, h), h
    ),
    M3 = safe_forecast(
      forecast_M3_XGB_original(y_train, h, seed = seed + 3L), h
    )
  )
}

# ---------------------------
# 10. Leakage-safe stacking
# ---------------------------
fit_stack_ols <- function(meta_y, predictors, new_predictors) {
  dat <- data.frame(y = as.numeric(meta_y), predictors, check.names = FALSE)
  newdat <- data.frame(new_predictors, check.names = FALSE)

  ok <- complete.cases(dat)
  dat <- dat[ok, , drop = FALSE]

  if (nrow(dat) < 8L) {
    return(rep(NA_real_, nrow(newdat)))
  }

  fit <- tryCatch(
    stats::lm(y ~ ., data = dat),
    error = function(e) NULL
  )

  if (is.null(fit)) {
    return(rep(NA_real_, nrow(newdat)))
  }

  pred <- tryCatch(
    suppressWarnings(as.numeric(stats::predict(fit, newdata = newdat))),
    error = function(e) rep(NA_real_, nrow(newdat))
  )

  clip_positive(pred)
}

stacking_forecasts <- function(
    y_train,
    int_train,
    int_test,
    h_test,
    final_base,
    seed
) {
  ntr <- length(y_train)
  n_meta <- max(12L, as.integer(floor(META_FRAC * ntr)))
  meta_start <- ntr - n_meta + 1L
  base_end <- meta_start - 1L

  y_base <- y_train[seq_len(base_end)]
  int_base <- int_train[seq_len(base_end)]
  int_meta <- int_train[meta_start:ntr]
  y_meta <- y_train[meta_start:ntr]

  # Crucially, the meta-learner is trained on forecasts for observations that
  # were not used to estimate the base models. This avoids in-sample leakage.
  meta_base <- base_forecasts(
    y_train = y_base,
    int_train = int_base,
    int_future = int_meta,
    h = n_meta,
    seed = seed + 1000L
  )

  # M7 = ARIMA + XGBoost
  p7_train <- data.frame(M1 = meta_base$M1, M3 = meta_base$M3)
  p7_test  <- data.frame(M1 = final_base$M1, M3 = final_base$M3)
  M7 <- fit_stack_ols(y_meta, p7_train, p7_test)

  # M8 = I-ARIMA + XGBoost
  p8_train <- data.frame(M2 = meta_base$M2, M3 = meta_base$M3)
  p8_test  <- data.frame(M2 = final_base$M2, M3 = final_base$M3)
  M8 <- fit_stack_ols(y_meta, p8_train, p8_test)

  # M9 = ARIMA + I-ARIMA + XGBoost
  p9_train <- data.frame(M1 = meta_base$M1, M2 = meta_base$M2, M3 = meta_base$M3)
  p9_test  <- data.frame(M1 = final_base$M1, M2 = final_base$M2, M3 = final_base$M3)
  M9 <- fit_stack_ols(y_meta, p9_train, p9_test)

  list(M7 = M7, M8 = M8, M9 = M9)
}

# ---------------------------
# 11. One Monte Carlo replication
# ---------------------------
run_one_replication <- function(scenario_row, rep_id, seed) {
  set.seed(seed)

  n <- as.integer(scenario_row$n)

  sim <- generate_interrupted_series(
    n = n,
    dynamics = scenario_row$dynamics,
    noise_sd = scenario_row$noise_sd,
    intervention_type = scenario_row$intervention_type,
    intervention_position = scenario_row$intervention_position,
    effect_log = scenario_row$effect_log
  )

  n_train <- as.integer(floor((1 - TEST_FRAC) * n))
  h <- n - n_train

  y_train <- sim$y[seq_len(n_train)]
  y_test <- sim$y[(n_train + 1L):n]

  int_train <- sim$intervention[seq_len(n_train)]
  int_test <- sim$intervention[(n_train + 1L):n]

  # Because the latest intervention position is 75% and training ends at 80%,
  # all simulated interruptions are observed before the final forecast window.
  stopifnot(sim$t0 <= n_train)

  # Base models M1-M3, reused by the stacking models.
  base <- base_forecasts(
    y_train = y_train,
    int_train = int_train,
    int_future = int_test,
    h = h,
    seed = seed + 10L
  )

  M1 <- base$M1
  M2 <- base$M2
  M3 <- base$M3

  M4 <- safe_forecast(
    forecast_M4_XGB_transformed(y_train, h, seed = seed + 40L), h
  )
  M5 <- safe_forecast(
    forecast_M5_XGB_stationary(y_train, h, seed = seed + 50L), h
  )
  M6 <- safe_forecast(
    forecast_M6_XGB_intervention(
      y_train, int_train, int_test, h, seed = seed + 60L
    ), h
  )

  stacks <- tryCatch(
    stacking_forecasts(
      y_train = y_train,
      int_train = int_train,
      int_test = int_test,
      h_test = h,
      final_base = base,
      seed = seed + 70L
    ),
    error = function(e) list(
      M7 = rep(NA_real_, h),
      M8 = rep(NA_real_, h),
      M9 = rep(NA_real_, h)
    )
  )

  forecasts <- list(
    M1 = M1,
    M2 = M2,
    M3 = M3,
    M4 = M4,
    M5 = M5,
    M6 = M6,
    M7 = stacks$M7,
    M8 = stacks$M8,
    M9 = stacks$M9
  )

  out <- lapply(names(forecasts), function(m) {
    a <- accuracy_metrics(y_test, forecasts[[m]])
    data.frame(
      scenario_id = scenario_row$scenario_id,
      replicate = rep_id,
      model = m,
      model_label = unname(MODEL_LABELS[m]),
      RMSE = unname(a["RMSE"]),
      MAE = unname(a["MAE"]),
      MAPE = unname(a["MAPE"]),
      sMAPE = unname(a["sMAPE"]),
      stringsAsFactors = FALSE
    )
  })

  dplyr::bind_rows(out)
}

# ---------------------------
# 12. Run full experiment with checkpointing
# ---------------------------
# A separate RDS file is created for every scenario, so the experiment can be
# resumed after interruption without rerunning completed scenarios.
run_monte_carlo <- function(scenarios, n_rep = N_REP) {
  for (s in seq_len(nrow(scenarios))) {
    sc <- scenarios[s, , drop = FALSE]
    checkpoint <- file.path(
      RESULT_DIR,
      paste0(sc$scenario_id, "_raw.rds")
    )

    if (file.exists(checkpoint)) {
      message("Skipping completed ", sc$scenario_id, " (", s, "/", nrow(scenarios), ")")
      next
    }

    message(
      "Running ", sc$scenario_id, " (", s, "/", nrow(scenarios), ") | ",
      sc$intervention_type, ", ", sc$magnitude, ", n=", sc$n,
      ", noise=", sc$noise_level, ", pos=", sc$intervention_pct,
      ", dynamics=", sc$dynamics
    )

    # Deterministic, scenario-specific seeds.
    scenario_num <- as.integer(sub("S", "", sc$scenario_id))
    seeds <- BASE_SEED + scenario_num * 100000L + seq_len(n_rep)

    scenario_results <- future.apply::future_lapply(
      X = seq_len(n_rep),
      FUN = function(r) {
        run_one_replication(
          scenario_row = sc,
          rep_id = r,
          seed = seeds[r]
        )
      },
      future.seed = TRUE,
      future.scheduling = 2
    )

    scenario_results <- dplyr::bind_rows(scenario_results)
    saveRDS(scenario_results, checkpoint)
  }

  invisible(TRUE)
}

# Uncomment to run:
# run_monte_carlo(scenarios, n_rep = N_REP)

# ---------------------------
# 13. Combine checkpoint files
# ---------------------------
combine_results <- function() {
  files <- list.files(
    RESULT_DIR,
    pattern = "^S[0-9]{3}_raw\\.rds$",
    full.names = TRUE
  )

  if (length(files) == 0L) {
    stop("No scenario result files were found. Run run_monte_carlo() first.")
  }

  raw <- bind_rows(lapply(files, readRDS)) %>%
    left_join(scenarios, by = "scenario_id")

  saveRDS(raw, file.path(RESULT_DIR, "all_raw_results.rds"))
  raw
}

# ---------------------------
# 14. Scenario-level summaries
# ---------------------------
summarise_results <- function(raw) {
  metrics_long <- raw %>%
    pivot_longer(
      cols = c(RMSE, MAE, MAPE, sMAPE),
      names_to = "metric",
      values_to = "value"
    )

  summary_long <- metrics_long %>%
    group_by(
      scenario_id,
      intervention_type,
      magnitude,
      n,
      noise_level,
      intervention_position,
      intervention_pct,
      dynamics,
      model,
      model_label,
      metric
    ) %>%
    summarise(
      n_success = sum(is.finite(value)),
      n_failed = sum(!is.finite(value)),
      mean = ifelse(n_success > 0, mean(value, na.rm = TRUE), NA_real_),
      median = ifelse(n_success > 0, median(value, na.rm = TRUE), NA_real_),
      sd = ifelse(n_success > 1, sd(value, na.rm = TRUE), NA_real_),
      q025 = ifelse(
        n_success > 0,
        unname(quantile(value, 0.025, na.rm = TRUE, names = FALSE)),
        NA_real_
      ),
      q975 = ifelse(
        n_success > 0,
        unname(quantile(value, 0.975, na.rm = TRUE, names = FALSE)),
        NA_real_
      ),
      se = ifelse(n_success > 1, sd / sqrt(n_success), NA_real_),
      mean_ci_low = ifelse(
        n_success > 1,
        mean - qt(0.975, df = n_success - 1) * se,
        NA_real_
      ),
      mean_ci_high = ifelse(
        n_success > 1,
        mean + qt(0.975, df = n_success - 1) * se,
        NA_real_
      ),
      .groups = "drop"
    )

  readr::write_csv(
    summary_long,
    file.path(RESULT_DIR, "scenario_metric_summary.csv")
  )

  summary_long
}

# ---------------------------
# 15. Win rates within each replication
# ---------------------------
# A "win" means the model has the smallest error for that scenario/replication.
# Ties share the win equally.
calculate_win_rates <- function(raw) {
  long <- raw %>%
    pivot_longer(
      cols = c(RMSE, MAE, MAPE, sMAPE),
      names_to = "metric",
      values_to = "value"
    ) %>%
    filter(is.finite(value)) %>%
    group_by(scenario_id, replicate, metric) %>%
    mutate(
      best = min(value),
      n_tied = sum(abs(value - best) < 1e-12),
      win_credit = ifelse(abs(value - best) < 1e-12, 1 / n_tied, 0)
    ) %>%
    ungroup()

  wins <- long %>%
    group_by(scenario_id, model, model_label, metric) %>%
    summarise(
      win_rate = mean(win_credit),
      .groups = "drop"
    ) %>%
    left_join(scenarios, by = "scenario_id")

  readr::write_csv(wins, file.path(RESULT_DIR, "scenario_win_rates.csv"))
  wins
}

# ---------------------------
# 16. Paired loss differences relative to Model 5
# ---------------------------
# This is especially useful for assessing whether Model 5's apparent advantage
# is repeatable across Monte Carlo replications rather than a point-estimate accident.
# Difference = competitor error - Model 5 error.
# Positive values therefore favour Model 5; negative values favour the competitor.
paired_differences_vs_M5 <- function(raw) {
  long <- raw %>%
    pivot_longer(
      cols = c(RMSE, MAE, MAPE, sMAPE),
      names_to = "metric",
      values_to = "value"
    )

  ref <- long %>%
    filter(model == "M5") %>%
    select(scenario_id, replicate, metric, M5_value = value)

  diffs <- long %>%
    filter(model != "M5") %>%
    left_join(ref, by = c("scenario_id", "replicate", "metric")) %>%
    mutate(diff_vs_M5 = value - M5_value) %>%
    group_by(scenario_id, model, model_label, metric) %>%
    summarise(
      n_pairs = sum(is.finite(diff_vs_M5)),
      mean_diff = mean(diff_vs_M5, na.rm = TRUE),
      median_diff = median(diff_vs_M5, na.rm = TRUE),
      sd_diff = sd(diff_vs_M5, na.rm = TRUE),
      se_diff = sd_diff / sqrt(n_pairs),
      ci_low = mean_diff - qt(0.975, df = pmax(n_pairs - 1, 1)) * se_diff,
      ci_high = mean_diff + qt(0.975, df = pmax(n_pairs - 1, 1)) * se_diff,
      proportion_M5_better = mean(diff_vs_M5 > 0, na.rm = TRUE),
      .groups = "drop"
    ) %>%
    left_join(scenarios, by = "scenario_id")

  readr::write_csv(
    diffs,
    file.path(RESULT_DIR, "paired_differences_vs_M5.csv")
  )

  diffs
}

# ---------------------------
# 17. Aggregate summaries by design factors
# ---------------------------
summarise_by_factor <- function(raw) {
  long <- raw %>%
    pivot_longer(
      cols = c(RMSE, MAE, MAPE, sMAPE),
      names_to = "metric",
      values_to = "value"
    )

  factor_summary <- long %>%
    group_by(
      intervention_type,
      magnitude,
      n,
      noise_level,
      intervention_pct,
      dynamics,
      model,
      model_label,
      metric
    ) %>%
    summarise(
      mean = mean(value, na.rm = TRUE),
      median = median(value, na.rm = TRUE),
      sd = sd(value, na.rm = TRUE),
      q025 = quantile(value, 0.025, na.rm = TRUE, names = FALSE),
      q975 = quantile(value, 0.975, na.rm = TRUE, names = FALSE),
      .groups = "drop"
    )

  readr::write_csv(
    factor_summary,
    file.path(RESULT_DIR, "factor_level_summary.csv")
  )

  factor_summary
}

# ---------------------------
# 18. Convenience function: finish analysis after simulation
# ---------------------------
finish_analysis <- function() {
  raw <- combine_results()
  summary_long <- summarise_results(raw)
  wins <- calculate_win_rates(raw)
  paired <- paired_differences_vs_M5(raw)
  factor_summary <- summarise_by_factor(raw)

  message("Finished. Output written to: ", normalizePath(RESULT_DIR))

  invisible(list(
    raw = raw,
    summary = summary_long,
    win_rates = wins,
    paired_vs_M5 = paired,
    factor_summary = factor_summary
  ))
}