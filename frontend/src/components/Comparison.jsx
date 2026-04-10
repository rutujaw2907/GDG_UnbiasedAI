import React from "react";

const Comparison = ({ beforeData, afterData }) => {
  if (!beforeData || !afterData) return null;

  return (
    <div className="card">
      <h3>Before vs After Bias Mitigation</h3>
      <div className="comparison-grid">
        <div>
          <h4>Before</h4>
          <p><strong>Accuracy:</strong> {beforeData.accuracy}</p>
          <p><strong>Bias Score:</strong> {beforeData.bias_score}</p>
          <p><strong>Severity:</strong> {beforeData.severity}</p>
        </div>

        <div>
          <h4>After</h4>
          <p><strong>Accuracy:</strong> {afterData.accuracy}</p>
          <p><strong>Bias Score:</strong> {afterData.bias_score}</p>
          <p><strong>Severity:</strong> {afterData.severity}</p>
        </div>
      </div>
    </div>
  );
};

export default Comparison;