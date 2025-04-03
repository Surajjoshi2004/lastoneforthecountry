import streamlit as st
import psutil
import pandas as pd
import plotly.express as px
import time
import numpy as np
from sklearn.ensemble import IsolationForest

# Set Streamlit Page Config
st.set_page_config(page_title="AI Task Manager", layout="wide")

st.title("AI Task Manager 📊")


# Function to get system stats
def get_system_stats():
    cpu = psutil.cpu_percent(interval=1)
    memory = psutil.virtual_memory().percent
    return cpu, memory


# Function to get process data
def get_process_data():
    processes = []
    for proc in psutil.process_iter(['pid', 'name', 'cpu_percent', 'memory_percent']):
        processes.append(proc.info)
    return processes


# Live Updating Graphs for CPU & Memory Usage
cpu_data = []
memory_data = []
time_data = []
graph_placeholder = st.empty()

for i in range(30):  # Collect data points for 30 seconds
    cpu, memory = get_system_stats()
    cpu_data.append(cpu)
    memory_data.append(memory)
    time_data.append(i)

    df = pd.DataFrame({'Time (s)': time_data, 'CPU Usage': cpu_data, 'Memory Usage': memory_data})
    fig = px.line(df, x='Time (s)', y=['CPU Usage', 'Memory Usage'], title="Live CPU & Memory Usage")
    graph_placeholder.plotly_chart(fig, use_container_width=True)

    time.sleep(1)

# Pie Chart for Top CPU-Consuming Processes
st.subheader("Top CPU-Consuming Processes")
processes = get_process_data()
df_process = pd.DataFrame(processes).sort_values(by=['cpu_percent'], ascending=False).head(5)
fig_pie = px.pie(df_process, values='cpu_percent', names='name', title="Top CPU Consuming Processes")
st.plotly_chart(fig_pie)

# Anomaly Detection using AI (Isolation Forest)
st.subheader("Anomaly Detection in System Processes")
cpu_usage = np.random.uniform(1, 100, 50)  # Simulating 50 processes
model = IsolationForest(contamination=0.1)
anomaly_scores = model.fit_predict(cpu_usage.reshape(-1, 1))

df_anomaly = pd.DataFrame({'Process ID': range(1, 51), 'Anomaly Score': anomaly_scores})
fig_bar = px.bar(df_anomaly, x='Process ID', y='Anomaly Score', title="Anomaly Detection Scores")
st.plotly_chart(fig_bar)

# Process Termination Impact (Before vs. After)
st.subheader("Before vs. After Process Termination")
cpu_before, mem_before = get_system_stats()
st.write(f"Before Termination - CPU: {cpu_before}% | Memory: {mem_before}%")

# Simulate process termination effect
time.sleep(5)  # Simulating waiting time after killing a process
cpu_after, mem_after = get_system_stats()
st.write(f"After Termination - CPU: {cpu_after}% | Memory: {mem_after}%")

df_comparison = pd.DataFrame({
    'State': ['Before', 'After'],
    'CPU Usage': [cpu_before, cpu_after],
    'Memory Usage': [mem_before, mem_after]
})

fig_comp = px.bar(df_comparison, x='State', y=['CPU Usage', 'Memory Usage'], barmode='group')
st.plotly_chart(fig_comp)
