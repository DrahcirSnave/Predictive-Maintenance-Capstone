# ❓ Project Q&A Session

**Q1: Why did you choose a Random Forest and LSTM hybrid approach?**
**A:** We chose **Random Forest** for its interpretability and ability to handle binary classification (Fail/No-Fail) effectively with limited compute resources. [cite_start]We chose **LSTM (Long Short-Term Memory)** neural networks for the regression task (predicting exact RUL) because LSTMs are specifically designed to recognize patterns in time-series data, making them ideal for analyzing sensor degradation over time[cite: 1].

**Q2: How do you handle the "Class Imbalance" in failure data?**
**A:** In predictive maintenance, "healthy" data vastly outweighs "failure" data. [cite_start]We addressed this during the **Data Preprocessing** phase by creating a rolling window of failure labels (e.g., labeling data points within 30 cycles of failure as "1")[cite: 1]. This increases the number of positive samples, allowing the model to learn failure patterns more effectively.

**Q3: What is the business impact of this model?**
[cite_start]**A:** Our analysis indicates a potential **40% reduction in downtime** and **25% savings in maintenance costs**[cite: 2]. By predicting failures before they occur, companies can switch from reactive maintenance (fixing broken machines) to predictive maintenance (fixing them right before they break).

**Q4: How scalable is this solution?**
**A:** The system was built using scalable libraries like TensorFlow and Scikit-learn. [cite_start]It runs efficiently on standard CPUs/GPUs and can be deployed via Docker containers or cloud platforms (like Google Colab or AWS) to handle data from thousands of sensors simultaneously[cite: 1].
