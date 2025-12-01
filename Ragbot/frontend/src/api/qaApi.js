// src/api/qaApi.js
import axiosInstance from "./axiosinstance";

export const askQuestion = async (question, documentId) => {
  // POST JSON body: backend should accept JSON or query params.
  const res = await axiosInstance.post("/answer/", { question, document_id: documentId });
  return res.data;
};
