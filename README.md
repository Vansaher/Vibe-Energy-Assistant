# Vibe Energy Assistant – Smart Appliance Energy Optimizer

Prototype Link: https://vibe-energy-assistant.onrender.com

## 🔌 Overview

VIBE Energy Assistant is a web-based smart energy optimization tool designed to help **B40 Malaysian households** reduce electricity costs by analyzing smart-meter data and providing personalized appliance scheduling recommendations. 

The system extracts household electricity usage using a registered **myTNB account number**, identifies peak usage patterns, forecasts the next 7 days of consumption, and suggests the best times to run energy-heavy appliances. 

This prototype was developed for **SDG Hackathon – Challenge 1: Affordable & Clean Energy.**

## 🎯 Objectives
* Help B40 households reduce electricity costs
* Provide personalized, actionable energy-saving recommendations
→ via forecasting, peak detection, and appliance scheduling.

## 💡 Problems We Solved

* ❗ Users lack control over electricity usage**

* Hard to identify which appliances consume the most

* No clarity on how to optimize appliance timing

**❗ Users need future consumption forecasts**

* myTNB currently shows only historical data

* No prediction of next week’s usage

* Users can’t budget or plan energy habits

## ⚡ Key Features

**1. Peak Hour Identification**

  Automatically highlights peak-usage days in the past 30 days.

**2. Smart Appliance Scheduler**

  Suggests optimal hours to run common appliances based on user patterns.

**3. 7-Day AI Forecasting**

  Forecasts daily energy consumption using a Prophet time-series model.

**4. Built-in Energy Chatbot**

  Assists users with appliance selection and scheduling recommendations.

## 🎯 Target Users

* Malaysian households using **myTNB**

* Especially **B40 families** who want to reduce monthly bills

## 🔄 System Flow

 **1.** User enters their **myTNB account number**

 **2.** System loads hourly energy data

 **3.** Generates:

  * Daily / Monthly bar charts

  * Peak day detection

  * 7-day usage forecast

 **4.** Chatbot recommends appliance scheduling

 **5.** User reviews forecast & suggestions in dashboard

## 🧪 Demo Account Numbers

Use any of the following in the prototype:

  * 210063746926
  * 210074377254
  * 210017514095
  * 210080926382
  * 210042648613


These accounts each have 6 months of hourly data (June–November 2025) for testing.

## 🖥️ Tech Stack

**Backend**

  * Python Flask

  * Prophet (7-day forecasting)

  * Pandas & NumPy (ETL + time-series processing)

**Frontend**

  * HTML + Jinja Templates

  * Bootstrap 5

  * Chart.js (interactive charts)

  * Custom CSS + gradient theme inspired by pitch deck

**Tools**

  * Google Colab (model training)

  * PyCharm (development)

  * Render.com (deployment)

**Datasets**

  * Household Power Consumption (Kaggle)

  * Household Appliances Dataset (Kaggle)

  * Synthetic myTNB hourly consumption (team-generated for accounts)

## 📊 Output

  * Elegant dashboard with usage trends
  * 7-day forecast shaded region
  * Peak usage days with highlights
  * Integrated chatbot window
  * Appliance suggestion cards

## 🤝 Team

**C1G13 – Vibe**
