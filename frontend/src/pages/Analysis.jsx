import React, { useState } from "react";
import { trainModel, mitigateBias } from "../services/api";
import Dashboard from "../components/Dashboard";
import Comparison from "../components/Comparison";
import Report from "../components/Report";

const Analysis = ({ uploadInfo }) => {
  const [result, setResult] = useState(null);
  const [mitigatedResult, setMitigatedResult] = useState(null);
  const [loadingTrain, setLoadingTrain] = useState(false);
  const [loadingMitigate, setLoadingMitigate] = useState(false);

  const handleTrain = async () => {
    try {
      setLoadingTrain(true);
      const data = await trainModel({
        file_path: uploadInfo.file_path,
        target_column: uploadInfo.targetColumn,
        sensitive_column: uploadInfo.sensitiveColumn,
      });
      setResult(data);
    } catch (error) {
      console.error(error);
      alert("Training failed.");
    } finally {
      setLoadingTrain(false);
    }
  };

  const handleMitigate = async () => {
    try {
      setLoadingMitigate(true);
      const data = await mitigateBias({
        file_path: uploadInfo.file_path,
        target_column: uploadInfo.targetColumn,
        sensitive_column: uploadInfo.sensitiveColumn,
      });
      setMitigatedResult(data);
    } catch (error) {
      console.error(error);
      alert("Bias mitigation failed.");
    } finally {
      setLoadingMitigate(false);
    }
  };

  return (
    <div className="container">
      <h1>Analysis</h1>

      <div className="button-row">
        <button onClick={handleTrain} disabled={loadingTrain}>
          {loadingTrain ? "Running..." : "Run Bias Analysis"}
        </button>

        <button
          onClick={handleMitigate}
          disabled={!result || loadingMitigate}
        >
          {loadingMitigate ? "Mitigating..." : "Apply Bias Mitigation"}
        </button>
      </div>

      <Dashboard result={result} />

      <Comparison beforeData={result} afterData={mitigatedResult} />

      {(result || mitigatedResult) && <Report />}
    </div>
  );
};

export default Analysis;