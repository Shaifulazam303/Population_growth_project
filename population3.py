import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# ----------------- Logistic ODE Functions -----------------
def logistic_ode(P, r, K):
    return r * P * (1 - P / K)

def euler_method(P0, r, K, h, steps):
    P = P0
    populations = [P]
    for _ in range(steps):
        P = P + h * logistic_ode(P, r, K)
        populations.append(P)
    return np.array(populations)

def rk4_method(P0, r, K, h, steps):
    P = P0
    populations = [P]
    for _ in range(steps):
        k1 = logistic_ode(P, r, K)
        k2 = logistic_ode(P + 0.5*h*k1, r, K)
        k3 = logistic_ode(P + 0.5*h*k2, r, K)
        k4 = logistic_ode(P + h*k3, r, K)
        P = P + (h/6)*(k1 + 2*k2 + 2*k3 + k4)
        populations.append(P)
    return np.array(populations)

# ----------------- Logistic Regression Fit -----------------
def logistic_function(t, r, K, A):
    return K / (1 + A * np.exp(-r*t))

def fit_logistic(t_data, p_data):
    initial_guess = [0.2, 400, 5]
    params, _ = curve_fit(
        logistic_function,
        t_data,
        p_data,
        p0=initial_guess,
        maxfev=5000
    )
    return params

# ----------------- Resource Effects -----------------
def individual_resource_effect(food=0.0, water=0.0, medicine=0.0):
    r_multiplier = 1 + 0.5 * water
    K_multiplier = 1 + 0.7 * food
    shock_reduction = 1 - 0.8 * medicine
    return r_multiplier, K_multiplier, shock_reduction

# ----------------- Shock Loss -----------------
def shock_loss(P, K, shock_type, shock_reduction=1.0):
    if shock_type == "pandemic":
        alpha = 0.4
    elif shock_type == "war":
        alpha = 0.7
    elif shock_type == "environment":
        alpha = 0.25
    else:
        alpha = 0.3

    loss = alpha * (P / K) * P
    return loss * shock_reduction

# ----------------- Shock Simulation -----------------
def span_shock_simulation(
    P0, r, K, h, days,
    shock_start, shock_end,
    shock_type,
    food=0.0,
    water=0.0,
    medicine=0.0
):
    P = P0
    populations = []

    r_mul, K_mul, shock_red = individual_resource_effect(food, water, medicine)
    r_eff = r * r_mul
    K_eff = K * K_mul

    for day in range(days + 1):
        if shock_start <= day <= shock_end:
            loss = shock_loss(P, K_eff, shock_type, shock_red)
        else:
            loss = 0

        populations.append(P)
        P = P + h * logistic_ode(P, r_eff, K_eff) - loss

    return np.array(populations)

# ----------------- Streamlit UI -----------------
st.title("Population Growth Simulation with Shocks & Resources")

st.sidebar.header("Simulation Parameters")
P0 = st.sidebar.number_input("Initial Population (P0)", value=50)
r = st.sidebar.number_input("Growth Rate (r)", value=0.3, step=0.01)
K = st.sidebar.number_input("Carrying Capacity (K)", value=500)
days = st.sidebar.slider("Simulation Days", 50, 10000, 500)
h = 1.0   

st.sidebar.header("Shock Parameters")
shock_type = st.sidebar.selectbox("Shock Type", ["pandemic", "war", "environment"])
shock_start = st.sidebar.slider("Shock Start Day", 0, days-1, 15)
shock_end = st.sidebar.slider("Shock End Day", shock_start+1, days, 20)

st.sidebar.header("Resource Levels (0 to 1)")
food = st.sidebar.slider("Food", 0.0, 1.0, 0.3)
water = st.sidebar.slider("Water", 0.0, 1.0, 0.3)
medicine = st.sidebar.slider("Medicine", 0.0, 1.0, 0.3)

# ----------------- Simulations -----------------
time = np.arange(days + 1)

rk4_pop = rk4_method(P0, r, K, h, days)
euler_pop = euler_method(P0, r, K, h, days)

# ----------------- DataFrame View -----------------
df = pd.DataFrame({
    "Time": time,
    "Euler": euler_pop,
    "RK4": rk4_pop
})

st.subheader("Euler vs RK4 Numerical Data")
st.dataframe(df.head(20))

st.download_button(
    "Download CSV",
    df.to_csv(index=False),
    "population_data.csv",
    "text/csv"
)

# ----------------- Logistic Regression -----------------
r_est, K_est, A_est = fit_logistic(time, rk4_pop)
p_fit = logistic_function(time, r_est, K_est, A_est)

# ----------------- Shock + Resource Simulation -----------------
shock_pop = span_shock_simulation(
    P0, r_est, K_est, h, days,
    shock_start, shock_end,
    shock_type, food, water, medicine
)

# ----------------- Plots -----------------
st.subheader("Euler vs RK4 Comparison")

fig1, ax1 = plt.subplots(figsize=(8,5))

ax1.plot(
    time,
    rk4_pop,
    label="RK4 Method",
    linestyle='-',
    linewidth=2,
    alpha=0.9,
    marker='s',
    markersize=4,
    color='tab:blue'
)

ax1.plot(
    time,
    euler_pop,
    label="Euler Method",
    linestyle='--',
    linewidth=2,
    alpha=0.7,
    marker='o',
    markersize=4,
    color='tab:red'
)

ax1.set_xlabel("Time")
ax1.set_ylabel("Population")
ax1.set_title("Population Growth: Euler vs RK4")
ax1.legend()
ax1.grid(True)

st.pyplot(fig1)
# ===================================

st.subheader("Absolute Error |RK4 − Euler|")
error_num = np.abs(rk4_pop - euler_pop)
fig2, ax2 = plt.subplots(figsize=(8,4))
ax2.plot(time, error_num)
ax2.set_xlabel("Time")
ax2.set_ylabel("Error")
ax2.grid(True)
st.pyplot(fig2)

st.subheader("Logistic Regression Fit to RK4")

fig3, ax3 = plt.subplots(figsize=(8,5))


ax3.scatter(
    time,
    rk4_pop,
    s=18,
    color='red',
    edgecolor='red',
    alpha=0.8,
    label="RK4 Data"
)


ax3.plot(
    time,
    p_fit,
    color='tab:cyan',
    linewidth=2.5,
    alpha=0.9,
    label="Logistic Fit"
)

ax3.set_xlabel("Time")
ax3.set_ylabel("Population")
ax3.set_title("Logistic Regression Fit to RK4 Solution")
ax3.legend()
ax3.grid(True)

st.pyplot(fig3)


# ----------------- RK4 vs Logistic Fit Error -----------------
error_fit = rk4_pop - p_fit

st.subheader("Error: RK4 − Logistic Regression Fit")

fig4, ax4 = plt.subplots(figsize=(8,5))

ax4.plot(
    time,
    error_fit,
    marker='o',
    linestyle='--',
    linewidth=2,
    color='tab:red',
    label="RK4 − Logistic Fit Error"
)

ax4.axhline(
    0,
    linestyle=':',
    linewidth=2,
    color='black'
)

ax4.set_xlabel("Time")
ax4.set_ylabel("Error")
ax4.legend()
ax4.grid(True)

st.pyplot(fig4)

# ----------------- Shock Visualization -----------------
st.subheader("Population with Shock & Resources")
fig5, ax5 = plt.subplots(figsize=(9,5))
ax5.plot(time, rk4_pop, label="Normal RK4", linewidth=2)
ax5.plot(time, shock_pop, '--', label="Shock + Resources", linewidth=2)
ax5.axvspan(shock_start, shock_end, alpha=0.2, label="Shock Period")
ax5.legend()
ax5.grid(True)
st.pyplot(fig5)

# ----------------- Sidebar Style -----------------
st.markdown(
    """
    <style>
    [data-testid="stSidebar"] { background-color: #6AECE1; }
    </style>
    """,
    unsafe_allow_html=True
)
