📈 Population Growth Simulation with Shocks & Resources

This project is an interactive population growth simulation built using Python and Streamlit.
It compares Euler and Runge–Kutta (RK4) numerical methods, fits a logistic growth model, and studies the effect of external shocks (pandemic, war, environment) and resources (food, water, medicine) on population dynamics.

🚀 Features

📊 Numerical Simulation

Euler Method

4th Order Runge–Kutta (RK4)

📉 Error Analysis

Absolute error between Euler and RK4

Error between RK4 and logistic regression fit

📐 Logistic Regression Fit

Estimates growth rate, carrying capacity, and initial condition

⚠️ Shock Modeling

Pandemic

War

Environmental shock

🍞 Resource Effects

Food → increases carrying capacity

Water → increases growth rate

Medicine → reduces shock impact

📥 Data Export

Download simulation results as CSV

🖥️ Interactive UI

Built using Streamlit sliders and inputs

🧠 Mathematical Model
Logistic Growth Equation
𝑑
𝑃
𝑑
𝑡
=
𝑟
𝑃
(
1
−
𝑃
𝐾
)
dt
dP
	​

=rP(1−
K
P
	​

)

Where:

𝑃
P = population

𝑟
r = growth rate

𝐾
K = carrying capacity

🔢 Numerical Methods Used
Euler Method

Simple and fast

Higher numerical error

First-order accuracy

Runge–Kutta (RK4)

More accurate

Fourth-order method

Used as reference solution

⚠️ Shock Model

During a shock period, population loss is calculated as:

Loss
=
𝛼
×
𝑃
𝐾
×
𝑃
Loss=α×
K
P
	​

×P

Where:

𝛼
α depends on shock type

Medicine reduces shock severity

🍽️ Resource Effects
Resource	Effect
Food	Increases carrying capacity
Water	Increases growth rate
Medicine	Reduces shock damage
🖥️ User Interface (Streamlit)

Users can control:

Initial population

Growth rate

Carrying capacity

Simulation duration

Shock type and duration

Resource levels (0–1)

All plots and results update in real time.

📊 Visualizations Included

Euler vs RK4 population growth

Absolute error between Euler and RK4

Logistic regression fit to RK4

RK4 vs Logistic fit error

Population under shock and resources

📦 Project Structure
├── app.py              # Main Streamlit application
├── README.md           # Project documentation
├── population_data.csv # Downloaded simulation output

🛠️ Installation & Run
1️⃣ Install dependencies
pip install streamlit numpy pandas matplotlib scipy

2️⃣ Run the app
streamlit run app.py

📚 Technologies Used

Python 🐍

Streamlit

NumPy

Pandas

Matplotlib

SciPy

🎓 Academic Relevance

This project demonstrates:

Numerical ODE solving

Error comparison between methods

Logistic population modeling

Parameter estimation via regression

Real-world scenario modeling

Suitable for:

Numerical Methods

Mathematical Modeling

Data Science Projects

Computational Biology / Population Dynamics

👤 Author

Md. Shaiful Azam
📍 Germany , Rhine waal University of Applied Sciences
