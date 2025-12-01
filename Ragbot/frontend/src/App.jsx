import Home from "./pages/Home";
import { Toaster } from "react-hot-toast";

function App() {
  return (
    <div className="bg-white">
      <div className="w-full max-w-7xl h-screen">
        <Home />
      </div>
      <Toaster />
    </div>
  );
}

export default App;
