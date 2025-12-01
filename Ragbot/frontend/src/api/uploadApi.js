import axiosInstance from "./axiosinstance.js";

export const uploadPDF = async (file) => {
    const formData = new FormData();
    formData.append("file", file);
    const res = await axiosInstance.post("/upload/", formData, {
        headers: {"Content-type": "multipart/form-data"},
    });
    return res.data;
}

