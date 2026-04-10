import axios from "axios";

const API = axios.create({
  baseURL: "http://127.0.0.1:5000",
});

export const uploadDataset = async (formData) => {
  const res = await API.post("/upload", formData, {
    headers: { "Content-Type": "multipart/form-data" },
  });
  return res.data;
};

export const trainModel = async (payload) => {
  const res = await API.post("/train", payload);
  return res.data;
};

export const mitigateBias = async (payload) => {
  const res = await API.post("/mitigate", payload);
  return res.data;
};

export const getReport = async () => {
  const res = await API.get("/report", { responseType: "blob" });
  return res.data;
};