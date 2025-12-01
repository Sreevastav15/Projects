import { useState, useRef } from "react";

function ChatBox({ onAsk, handleUploadClick, fileName }) {
  const [input, setInput] = useState("");
  const menuRef = useRef(null);

  const handleAsk = () => {
    if (input.trim()) {
      onAsk(input);
      setInput("");
    }
  };

  return (
    <div className={`${fileName ? "fixed bottom-1 -translate-x-3" : ""} flex justify-center items-center w-full max-w-3xl mt-8`}>
      {/* Chatbox container */}
      <div className="flex items-center bg-white shadow-md rounded-full px-4 py-2 w-[80%] max-w-[800px] border border-gray-200">
        
        {/* Upload Button */}
        <div className="relative" ref={menuRef}>
          <button
            onClick={() => handleUploadClick()}
            className={`flex items-center justify-center w-10 h-10 bg-white text-gray-700 rounded-full text-xl font-bold hover:bg-gray-300 transition-all duration-200 hover:scale-105`}
            title="Upload PDF"
          >
            +
          </button>
        </div>

        {/* Input Field */}
        <input
          type="text"
          value={input}
          placeholder="Ask your own question..."
          onChange={(e) => setInput(e.target.value)}
          className="flex-1 bg-transparent outline-none border-none text-gray-700 text-base placeholder-gray-400 px-3 focus:ring-0"
        />

        {/* Ask Button */}
        <button
          onClick={handleAsk}
          className="ml-3 bg-blue-600 text-white font-semibold px-5 py-2 rounded-full hover:bg-blue-700 transition-all duration-200"
        >
          Ask
        </button>
      </div>
    </div>
  );
}

export default ChatBox;
