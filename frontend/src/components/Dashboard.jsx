import React from "react";
import Charts from "./Charts";

const Dashboard = ({ result }) => {
  if (!result) return null;

  const chartData =
    result.group_metrics?.map((item) => ({
      group: item.group,
      selection_rate: item.selection_rate,
    })) || [];

  return (
    <div className="card">
      <h2>Bias Analysis Dashboard</h2>

      <p><strong>Model Accuracy:</strong> {result.accuracy}</p>
      <p><strong>Bias Score:</strong> {result.bias_score}</p>
      <p><strong>Severity:</strong> {result.severity}</p>
      <p><strong>Message:</strong> {result.message}</p>

      <Charts chartData={chartData} title="Selection Rate by Group" />

      {result.top_features && (
        <div>
          <h3>Top Influencing Features</h3>
          <ul>
            {result.top_features.map((feature, idx) => (
              <li key={idx}>
                {feature.name}: {feature.importance}
              </li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
};

export default Dashboard;