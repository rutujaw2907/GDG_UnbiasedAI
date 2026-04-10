import React from "react";
import { getReport } from "../services/api";

const Report = () => {
  const handleDownload = async () => {
    try {
      const blob = await getReport();
      const url = window.URL.createObjectURL(new Blob([blob]));
      const link = document.createElement("a");
      link.href = url;
      link.setAttribute("download", "bias_audit_report.txt");
      document.body.appendChild(link);
      link.click();
      link.remove();
    } catch (error) {
      console.error(error);
      alert("Failed to download report.");
    }
  };

  return (
    <div className="card">
      <h3>Audit Report</h3>
      <button onClick={handleDownload}>Download Report</button>
    </div>
  );
};

export default Report;