import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

function AnswerDisplay({ question, answer }) {
  if (!answer) return null;

  return (
    <div className="bg-white rounded-lg p-4 w-full">
      <div className="flex flex-col space-y-3">
        <div className="self-end mt-2 bg-gray-200 text-black rounded-3xl p-3 max-w-[100%]">
          <h3 className="font-semibold text-gray-700 mb-2 text-right">Question:</h3>
          <p className="mb-3 text-right">{question}</p>
        </div>

        <div className="self-start rounded-3xl max-w-[100] p-2">
          <h3 className="font-semibold text-blue-700 mb-2">Assistant:</h3>
          <ReactMarkdown remarkPlugins={[remarkGfm]}>{answer}</ReactMarkdown>
        </div>
      </div>
    </div>
  );
}

export default AnswerDisplay;
