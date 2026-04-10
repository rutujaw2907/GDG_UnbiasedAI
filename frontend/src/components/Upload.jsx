import React, { useState } from "react";
import { uploadDataset } from "../services/api";

const Upload = ({ onUploadSuccess }) => {
  const [file, setFile] = useState(null);
  const [targetColumn, setTargetColumn] = useState("");
  const [sensitiveColumn, setSensitiveColumn] = useState("");
  const [loading, setLoading] = useState(false);

  const handleUpload = async (e) => {
    e.preventDefault();

    if (!file || !targetColumn || !sensitiveColumn) {
      alert("Please select file, target column, and sensitive column.");
      return;
    }

    const formData = new FormData();
    formData.append("file", file);
    formData.append("target_column", targetColumn);
    formData.append("sensitive_column", sensitiveColumn);

    try {
      setLoading(true);
      const data = await uploadDataset(formData);
      onUploadSuccess({
        ...data,
        targetColumn,
        sensitiveColumn,
      });
    } catch (error) {
      console.error(error);
      alert("Upload failed.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="card">
      <h2>Upload Dataset</h2>
      <form onSubmit={handleUpload}>
        <input
          type="file"
          accept=".csv"
          onChange={(e) => setFile(e.target.files[0])}
        />

        <input
          type="text"
          placeholder="Enter target column"
          value={targetColumn}
          onChange={(e) => setTargetColumn(e.target.value)}
        />

        <input
          type="text"
          placeholder="Enter sensitive column"
          value={sensitiveColumn}
          onChange={(e) => setSensitiveColumn(e.target.value)}
        />

        <button type="submit" disabled={loading}>
          {loading ? "Uploading..." : "Upload"}
        </button>
      </form>
    </div>
  );
};

export default Upload;