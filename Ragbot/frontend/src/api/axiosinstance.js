import axios from "axios";

const axiosInstant = axios.create({
    baseURL: "http://localhost:8000",
})

export default axiosInstant;