import { useState, useRef, useEffect } from "react";
import UploadSection from "../components/UploadSection";
import ChatBox from "../components/ChatBox";
import Sidebar from "../components/Sidebar";
import { uploadPDF } from "../api/uploadApi";
import toast from "react-hot-toast";
import AnswerDisplay from "../components/AnswerDisplay";
import { askQuestion } from "../api/qaApi";
import { loadChatSession, fullChatHistory } from "../api/chatHistory";
import "./Home.css";

function Home() {
  const [questions, setQuestions] = useState([]);
  const [docId, setDocId] = useState(null);
  const [conversation, setConversation] = useState([]);
  const [loading, setLoading] = useState(false);
  const [fileName, setFileName] = useState("");
  const [refresh, setRefresh] = useState(false);
  const uploadSectionRef = useRef(null);

  const handleUpload = async (e) => {
    e.preventDefault();
    const fileInput = uploadSectionRef.current;
    const file = fileInput?.files?.[0];

    if (!file) {
      toast.error("Please select a PDF file!");
      return;
    }

    setLoading(true);
    toast.loading("Uploading and processing PDF...");

    try {
      const data = await uploadPDF(file);
      toast.dismiss();
      toast.success("Document Uploaded");
      setDocId(data.document_id);
      setFileName(file.name);
      setRefresh(prev => !prev)
    } catch (error) {
      console.error(error);
      toast.dismiss();
      toast.error("Upload failed. Please try again.");
    } finally {
      setLoading(false);
    }
  };

  const handleSelectChat = async (doc_id) => {
    try {
      toast.loading("Loading Chat")
      const data = await loadChatSession(doc_id);
      const answer = await fullChatHistory(doc_id);

      const formattedConversation = [];
      let lastUserMessage = null;

      data.messages.forEach(msg => {
        if (msg.role === "user") {
          lastUserMessage = { question: msg.content, answer: "" };
          formattedConversation.push(lastUserMessage);
        } else if (msg.role === "assistant" && lastUserMessage) {
          if (lastUserMessage){
            lastUserMessage.answer = msg.content;
          } else{
            formattedConversation.push({
              question: "(No question — summary or system message)",
              answer: msg.content
            });
          }
        }
      });
      setDocId(data.document_id)
      setFileName(data.filename);
      setConversation(answer.conversation);
      toast.dismiss()
      toast.success(`Loaded chat`)
    } catch (error) {
      toast.dismiss()
      toast.error("Failed to load chat history");
      console.error(error);
    }
  };

  const handleNewChat = () => {
    setQuestions([]);
    setConversation([]);
    setFileName("");
    setDocId(null);
  };

  const handleUploadClick = () => {
    if (uploadSectionRef.current) uploadSectionRef.current.click();
  };

  const handleHiddenFileChange = async (e) => {
    await handleUpload(e);
  };

  const handleAsk = async (question) => {
    if (!docId) return toast.error("Upload a document first.");

    setConversation((prev) => [...prev, { question, answer: "..." }]);
    toast.loading("Getting answer...");

    try {
      const data = await askQuestion(question, docId);
      toast.dismiss();
      toast.success("Answer ready!");

      setConversation((prev) =>
        prev.map((msg, i) =>
          i === prev.length - 1 ? { ...msg, answer: data.answer } : msg
        )
      );
    } catch (e) {
      toast.dismiss();
      toast.error("Failed to get answer.");
    }
  };

  const messagesEndRef = useRef(null);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [conversation]);

  return (
    <>
      <header className="bg-black h-[100px] justify-center items-center p-2">
        <h1 className="font-bold text-white text-3xl translate-y-5">RAGBot</h1>
      </header>

      <div className="flex bg-white border border-white overflow-y-hidden">
        <Sidebar onNewChat={handleNewChat} onSelectChat={handleSelectChat} refresh={refresh} />
        <div className="flex flex-col items-center justify-start w-full py-10 px-4">
          {/* Upload Section */}
          <div className="w-full">
            <UploadSection fileName={fileName} />
          </div>

          {/* Hidden File Input */}
          <input
            type="file"
            accept="application/pdf"
            ref={uploadSectionRef}
            onChange={handleHiddenFileChange}
            className="hidden"
          />

          <div className="w-full gap-2 mt-4 items-start max-h-[500px] overflow-y-auto scrollbar-hidden">
            {/* Question List
            <div className="mt-4">
              <QuestionList questions={questions} onSelect={handleAsk} />
            </div> */}

            {/* Answer Display */}
            <div className="mt-4">
              <div className="flex flex-col space-y-5">
                {conversation.map((msg, index) => (
                  <AnswerDisplay
                    key={index}
                    question={msg.question}
                    answer={msg.answer}
                  />
                ))}
              </div>
              <div ref={messagesEndRef}></div>
            </div>
          </div>

          {/* Chat Box */}
          <div className="w-full max-w-3xl mt-6">
            <ChatBox
              onAsk={handleAsk}
              handleUploadClick={handleUploadClick}
              fileName={fileName}
            />
          </div>
        </div>
      </div>
    </>
  );
}

export default Home;
