import React from "react";
import Upload from "../components/Upload";

const Home = ({ onUploadSuccess }) => {
  return (
    <div className="container">
      <h1>AI Bias Audit Tool</h1>
      <p>
        Upload a dataset, analyze bias, compare fairness before and after
        mitigation, and download the audit report.
      </p>
      <Upload onUploadSuccess={onUploadSuccess} />
    </div>
  );
};

export default Home;