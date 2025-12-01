import { useState, useEffect } from "react";
import { Menu } from "lucide-react";
import { fileHistory } from "../api/chatHistory";
import { deleteChat } from "../api/delete";
import { Trash2 } from "lucide-react";
import { toast } from "react-hot-toast";
import "./Sidebar.css"

function Sidebar({ onNewChat, onSelectChat, refresh }) {
    const [isOpen, setIsOpen] = useState(false);
    const [fileList, setFileList] = useState([]);

    const toggleSidebar = () => {
        setIsOpen((prev) => !prev);
    };

    const getFiles = async () => {
        try {
            const data = await fileHistory();
            setFileList(data);
        } catch (error) {
            console.error("Error fetching chat history", error);
        }
    }

    const handleDelete = (doc_id) => {
        try {
            toast.loading("Deleting Chat")
            deleteChat(doc_id);
            toast.dismiss()
            toast.success("Chat deleted")
            setFileList((prev) => prev.filter((f) => f.doc_id !== doc_id))
        } catch (error) {
            console.error("Failed to delete", error)
        }
    }

    useEffect(() => {
        if (isOpen) {
            getFiles();
        }
    }, [isOpen]);

    useEffect(() => {
        if (isOpen) getFiles();
    }, [refresh])

    return (
        <>
            <div className="flex flex-col">
                <button
                    onClick={toggleSidebar}
                    className={`p-4 ${isOpen ? "bg-black text-white" : ""}`}
                >
                    <Menu />
                </button>

                {isOpen && (
                    <aside className="flex flex-col divide-y divide-gray-200 bg-black h-screen w-60 text-white p-4 overflow-y-auto scrollbar-hidden">
                        <p
                            className="w-full h-10 rounded-lg p-2 hover:bg-gray-800 cursor-pointer mb-2"
                            onClick={onNewChat}
                        >
                            New Chat
                        </p>
                        <div className="flex flex-col p-2">
                            <p className="text-sm mb-2 text-gray-400">Chats</p>

                            <ul className="space-y-2">
                                {fileList.map((file) => (
                                        <li
                                            className="flex rounded cursor-pointer p-2 hover:bg-gray-800 focus:bg-gray-800"
                                            key={file.doc_id}
                                        >
                                            <span
                                                onClick={() => onSelectChat(file.doc_id)}
                                                className="truncate flex-1"
                                            >
                                                {file.filename}
                                            </span>

                                            <button
                                                onClick={(e) => {
                                                    e.stopPropagation(); // prevents triggering chat open
                                                    handleDelete(file.doc_id);
                                                    onNewChat();
                                                }}
                                                className="ml-2 text-gray-400 hover:text-red-500"
                                                title="Delete chat"
                                            >
                                                <Trash2 size={16} />
                                            </button>
                                        </li>
                                ))}

                            </ul>
                        </div>

                    </aside>
                )}
            </div>
        </>
    );
}

export default Sidebar;
