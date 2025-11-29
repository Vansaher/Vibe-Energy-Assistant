# app.py  — Energy Assistant dashboard using pre-trained Prophet models

from flask import Flask, render_template, redirect, url_for, request
import pandas as pd
import datetime
import statistics
import pickle

app = Flask(__name__)

# -------------------------------------------------------------------
# CONFIG: input data
# -------------------------------------------------------------------
ACCOUNTS_CSV = "accounts_meta.csv"
HOURLY_CSV = "mytnb_hourly.csv"  # change to mytnb_hourly_5_accounts.csv if that's your file name

accounts_df = pd.read_csv(ACCOUNTS_CSV, dtype={"account_number": str})
hourly_df = pd.read_csv(HOURLY_CSV, dtype={"account_number": str})

hourly_df["datetime"] = pd.to_datetime(hourly_df["datetime"])
hourly_df["date"] = hourly_df["datetime"].dt.date

# -------------------------------------------------------------------
# LOAD PRE-TRAINED PROPHET MODELS (trained in notebook on Kaggle data)
# -------------------------------------------------------------------

with open("backend/prophet_daily.pkl", "rb") as f:
    m_day = pickle.load(f)

# -------------------------------------------------------------------
# HELPER: scaled forecast from a global Prophet model
# -------------------------------------------------------------------
def make_scaled_future(model, user_mean, periods, freq):
    """
    Use a pre-trained Prophet model as a pattern:
      - take its latest `periods` predictions,
      - rescale by (user_mean / global_mean),
      - return DataFrame with ['ds','yhat'].
    Actual dates will be aligned to the user's timeline later.
    """
    hist = model.history.copy()
    global_mean = hist["y"].mean() if not hist.empty else 1.0
    if global_mean == 0:
        global_mean = 1.0

    scale = user_mean / global_mean if user_mean is not None else 1.0

    future = model.make_future_dataframe(periods=periods, freq=freq)
    forecast = model.predict(future)

    future_part = forecast.tail(periods).copy()
    future_part["yhat"] = future_part["yhat"] * scale

    return future_part[["ds", "yhat"]]


# -------------------------------------------------------------------
# HELPER: build all 3 chart blobs + stats for one account
# -------------------------------------------------------------------
def make_chart_blobs(acc_data):
    """
    acc_data: hourly rows for a single account (datetime, date, kwh)
    Returns:
        hourly_chart, daily_chart, monthly_chart, stats, peak_days
    """

    if acc_data.empty:
        empty_blob = {"labels": [], "history": [], "forecast": []}
        stats = {
            "total_past": 0,
            "avg_past": 0,
            "max_day_label": "-",
            "max_day_value": 0,
            "estimated_bill": 0,
        }
        peak_days = []
        return empty_blob, empty_blob, empty_blob, stats, peak_days

    # ---------------- DAILY AGGREGATION ----------------
    daily = (
        acc_data.groupby("date", as_index=False)["kwh"]
        .sum()
        .sort_values("date")
    )
    daily["date"] = pd.to_datetime(daily["date"])

    # last 30 days of history (for chart + KPIs)
    daily_sorted = daily.sort_values("date")
    if len(daily_sorted) > 30:
        hist_last30 = daily_sorted.tail(30).copy()
    else:
        hist_last30 = daily_sorted.copy()

    # ---------------- DAILY FORECAST (7 days) ----------------
    if not daily_sorted.empty:
        user_daily_mean = daily_sorted["kwh"].mean()
        future_daily = make_scaled_future(m_day, user_daily_mean, periods=7, freq="D")
        # align forecast dates to follow user's last date
        last_date = daily_sorted["date"].max()
        future_daily["ds"] = [
            last_date + datetime.timedelta(days=i)
            for i in range(1, len(future_daily) + 1)
        ]
        future_daily = future_daily.rename(columns={"ds": "date"})
    else:
        future_daily = pd.DataFrame(columns=["date", "yhat"])

    # Daily chart labels
    hist_last30["label"] = hist_last30["date"].dt.strftime("%b %d")  # Windows-safe
    future_daily["label"] = future_daily["date"].dt.strftime("%b %d")

    day_labels = list(hist_last30["label"]) + list(future_daily["label"])
    day_history = list(hist_last30["kwh"]) + [None] * len(future_daily)
    day_forecast = [None] * len(hist_last30) + [
        round(v, 3) for v in future_daily.get("yhat", [])
    ]

    daily_chart = {
        "labels": day_labels,
        "history": [
            round(v, 3) if v is not None else None for v in day_history
        ],
        "forecast": day_forecast,
    }

    # ---------------- HOURLY CHART (last 72h) ----------------
    acc_sorted = acc_data.sort_values("datetime").copy()
    last_72 = acc_sorted.tail(72).copy()

    last_72["label"] = last_72["datetime"].dt.strftime("%d %b\n%H:%M")

    hour_labels = list(last_72["label"])
    hour_history = list(last_72["kwh"])

    hourly_chart = {
        "labels": hour_labels,
        "history": [round(v, 3) for v in hour_history],
        "forecast": [None] * len(hour_labels)  # keep key for Chart.js, but empty
    }
    # ---- MONTHLY CHART (HISTORY) ----
    month_hist = daily_sorted.copy()
    month_hist["month"] = month_hist["date"].dt.to_period("M").dt.to_timestamp()
    month_hist = (
        month_hist.groupby("month", as_index=False)["kwh"]
        .sum()
        .sort_values("month")
    )

    month_hist["label"] = month_hist["month"].dt.strftime("%b %Y")

    monthly_chart = {
        "labels": list(month_hist["label"]),
        "history": [round(v, 3) for v in month_hist["kwh"]],
        "forecast": [None] * len(month_hist)  # no forecast
    }

    # ---------------- STATS & PEAK DAYS (from daily history) ----------------
    if not hist_last30.empty:
        past_values = list(hist_last30["kwh"])
        total_past = round(sum(past_values), 3)
        avg_past = round(statistics.mean(past_values), 3)
        max_value = max(past_values)
        max_day_label = hist_last30.loc[
            hist_last30["kwh"].idxmax(), "date"
        ].strftime("%b %d")
    else:
        total_past = 0
        avg_past = 0
        max_value = 0
        max_day_label = "-"

    estimated_bill = round(total_past * 0.50, 2)

    stats = {
        "total_past": total_past,
        "avg_past": avg_past,
        "max_day_label": max_day_label,
        "max_day_value": max_value,
        "estimated_bill": estimated_bill,
    }

    # Peak usage days (top 3 from last 30 days)
    peak_days_raw = hist_last30.copy()
    peak_days_raw["label"] = peak_days_raw["date"].dt.strftime("%b %d")
    peak_sorted = peak_days_raw.sort_values("kwh", ascending=False)

    peak_days = list(zip(peak_sorted["label"].head(3), peak_sorted["kwh"].head(3)))

    return hourly_chart, daily_chart, monthly_chart, stats, peak_days


# -------------------------------------------------------------------
# ROUTES
# -------------------------------------------------------------------
@app.route("/", methods=["GET", "POST"])
def home():
    error = None
    sample_accounts = accounts_df["account_number"].tolist()

    if request.method == "POST":
        acc_input = request.form.get("account_number", "").strip()

        if not acc_input:
            error = "Please enter an account number."
        elif acc_input not in accounts_df["account_number"].values:
            error = "Account not found. Please try again."
        else:
            return redirect(url_for("dashboard", account=acc_input))

    return render_template("home.html", error=error, sample_accounts=sample_accounts)


@app.route("/dashboard")
def dashboard():
    account_number = request.args.get("account", "").strip()

    if not account_number:
        return redirect(url_for("home"))

    acc_row = accounts_df[accounts_df["account_number"] == account_number]
    if acc_row.empty:
        return redirect(url_for("home"))

    account_info = acc_row.iloc[0].to_dict()

    acc_data = hourly_df[hourly_df["account_number"] == account_number].copy()
    if acc_data.empty:
        empty_blob = {"labels": [], "history": [], "forecast": []}
        return render_template(
            "dashboard.html",
            hourly_chart=empty_blob,
            daily_chart=empty_blob,
            monthly_chart=empty_blob,
            peak_days=[],
            stats={
                "total_past": 0,
                "avg_past": 0,
                "max_day_label": "-",
                "max_day_value": 0,
                "estimated_bill": 0,
            },
            account_info=account_info,
        )

    hourly_chart, daily_chart, monthly_chart, stats, peak_days = make_chart_blobs(acc_data)

    return render_template(
        "dashboard.html",
        hourly_chart=hourly_chart,
        daily_chart=daily_chart,
        monthly_chart=monthly_chart,
        peak_days=peak_days,
        stats=stats,
        account_info=account_info,
    )


if __name__ == "__main__":
    app.run(debug=True)
