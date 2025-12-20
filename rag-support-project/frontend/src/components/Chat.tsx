import { useEffect, useMemo, useState } from "react";
import { supabase } from "../lib/supabase";
import { Button } from "./ui/button";
import { Input } from "./ui/input";
import { ScrollArea } from "./ui/scroll-area";
import { Switch } from "./ui/switch";
import { Avatar, AvatarFallback } from "./ui/avatar";

type Role = "user" | "assistant";

interface Message {
  role: Role;
  content: string;
}

interface Conversation {
  id: string;
  title: string;
  messages: Message[];
}

export default function Chat() {
  const [conversations, setConversations] = useState<Conversation[]>([{
    id: "default",
    title: "New conversation",
    messages: [],
  }]);
  const [activeId, setActiveId] = useState<string>("default");
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [darkMode, setDarkMode] = useState(true);

  const activeConversation = useMemo(
    () => conversations.find((c) => c.id === activeId) ?? conversations[0],
    [conversations, activeId]
  );

  const messages = activeConversation?.messages ?? [];

  useEffect(() => {
    document.documentElement.classList.toggle("dark", darkMode);
  }, [darkMode]);

  const handleNewChat = () => {
    const id = crypto.randomUUID();
    const next: Conversation = {
      id,
      title: "New conversation",
      messages: [],
    };
    setConversations((prev) => [next, ...prev]);
    setActiveId(id);
  };

  const appendMessage = (conversationId: string, message: Message) => {
    setConversations((prev) =>
      prev.map((c) =>
        c.id === conversationId
          ? { ...c, messages: [...c.messages, message] }
          : c
      )
    );
  };

  const updateLastAssistantMessage = (conversationId: string, content: string) => {
    setConversations((prev) =>
      prev.map((c) => {
        if (c.id !== conversationId) return c;
        const nextMessages = [...c.messages];
        const lastIndex = nextMessages.length - 1;
        if (lastIndex >= 0 && nextMessages[lastIndex].role === "assistant") {
          nextMessages[lastIndex] = { ...nextMessages[lastIndex], content };
        }
        return { ...c, messages: nextMessages };
      })
    );
  };

  const sendMessage = async () => {
    const trimmed = input.trim();
    if (!trimmed || !activeConversation) return;

    const conversationId = activeConversation.id;
    setLoading(true);

    appendMessage(conversationId, { role: "user", content: trimmed });
    setInput("");

    const token = (await supabase.auth.getSession()).data.session?.access_token;
    const response = await fetch("/api/chat", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Authorization: `Bearer ${token}`,
      },
      body: JSON.stringify({ query: trimmed }),
    });

    const reader = response.body?.getReader();
    if (!reader) {
      appendMessage(conversationId, {
        role: "assistant",
        content: "Sorry, I couldn't get a response.",
      });
      setLoading(false);
      return;
    }

    let botContent = "";
    appendMessage(conversationId, { role: "assistant", content: "" });

    // SSE parser: buffer text, split events on '\n\n', extract lines starting with 'data:'
    let buffer = "";
    let finished = false;

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      const text = new TextDecoder().decode(value);
      buffer += text;

      // Process all complete SSE events in the buffer
      let idx: number;
      while ((idx = buffer.indexOf("\n\n")) !== -1) {
        const rawEvent = buffer.slice(0, idx);
        buffer = buffer.slice(idx + 2);

        // collect all data: lines for this event
        const lines = rawEvent.split(/\r?\n/);
        const dataLines = lines
          .filter((l) => l.startsWith("data:"))
          .map((l) => l.replace(/^data:\s?/, ""));

        if (dataLines.length === 0) continue;

        const payload = dataLines.join("\n");
        if (payload.trim() === "[DONE]") {
          finished = true;
          break;
        }

        botContent += payload;
        updateLastAssistantMessage(conversationId, botContent);
      }

      if (finished) break;
    }

    // Flush any remaining data lines if the stream ended without trailing '\n\n'
    if (!finished && buffer.length > 0) {
      const lines = buffer.split(/\r?\n/);
      const dataLines = lines
        .filter((l) => l.startsWith("data:"))
        .map((l) => l.replace(/^data:\s?/, ""));
      if (dataLines.length > 0) {
        const payload = dataLines.join("\n");
        if (payload.trim() !== "[DONE]") {
          botContent += payload;
          updateLastAssistantMessage(conversationId, botContent);
        }
      }
    }

    setLoading(false);
  };

  const handleKeyDown: React.KeyboardEventHandler<HTMLInputElement> = (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      if (!loading) {
        sendMessage();
      }
    }
  };

  return (
    <div
      className={`flex h-screen w-full bg-slate-950 text-slate-50`}
    >
      {/* Sidebar: chat history */}
      <aside className="hidden h-full w-64 flex-col border-r border-slate-800 bg-slate-950/80 p-3 md:flex">
        <div className="mb-3 flex items-center justify-between gap-2">
          <div>
            <p className="text-xs font-semibold uppercase tracking-wide text-slate-400">
              Conversations
            </p>
          </div>
          <Button
            size="sm"
            variant="outline"
            onClick={handleNewChat}
            className="h-7 px-2 text-xs"
          >
            + New
          </Button>
        </div>
        <ScrollArea className="flex-1">
          <div className="space-y-1 pr-1 text-sm">
            {conversations.map((c) => {
              const isActive = c.id === activeConversation?.id;
              return (
                <button
                  key={c.id}
                  onClick={() => setActiveId(c.id)}
                  className={`w-full rounded-md px-3 py-2 text-left transition-colors ${isActive
                      ? "bg-slate-800 text-slate-50"
                      : "bg-transparent text-slate-300 hover:bg-slate-800/80"
                    }`}
                >
                  <div className="line-clamp-1 text-xs font-medium">
                    {c.title || "Conversation"}
                  </div>
                  <div className="mt-0.5 line-clamp-1 text-[11px] text-slate-400">
                    {c.messages[c.messages.length - 1]?.content ||
                      "Start chatting with your assistant"}
                  </div>
                </button>
              );
            })}
          </div>
        </ScrollArea>
      </aside>

      {/* Main chat area */}
      <div className="flex h-full flex-1 flex-col">
        {/* Header with dark mode toggle */}
        <header className="flex h-14 items-center justify-between border-b border-slate-800 bg-slate-950/90 px-4">
          <div className="flex items-center gap-2">
            <Avatar className="h-7 w-7 bg-slate-800">
              <AvatarFallback>AI</AvatarFallback>
            </Avatar>
            <div className="flex flex-col">
              <span className="text-sm font-semibold">Support assistant</span>
              <span className="text-xs text-slate-400">
                Ask anything about your project documentation
              </span>
            </div>
          </div>

          <div className="flex items-center gap-2 text-xs text-slate-400">
            <span>{darkMode ? "Dark" : "Light"} mode</span>
            <Switch checked={darkMode} onCheckedChange={setDarkMode} />
          </div>
        </header>

        {/* Messages */}
        <div className="flex-1 bg-gradient-to-b from-slate-950 via-slate-950 to-slate-950/90">
          <ScrollArea className="h-full w-full">
            <div className="mx-auto flex h-full w-full max-w-3xl flex-col gap-3 px-4 py-4">
              {messages.length === 0 && !loading && (
                <div className="mt-20 text-center text-sm text-slate-500">
                  <p className="mb-1 text-base font-medium text-slate-300">
                    Welcome 👋
                  </p>
                  <p>Start a conversation by asking a question below.</p>
                </div>
              )}

              {messages.map((msg, index) => {
                const isUser = msg.role === "user";
                return (
                  <div
                    key={index}
                    className={`flex w-full ${isUser ? "justify-end" : "justify-start"}`}
                  >
                    <div
                      className={`flex max-w-[80%] items-end gap-2 ${isUser ? "flex-row-reverse" : "flex-row"
                        }`}
                    >
                      <Avatar className="h-7 w-7 bg-slate-800/80 text-[11px]">
                        <AvatarFallback>{isUser ? "You" : "AI"}</AvatarFallback>
                      </Avatar>
                      <div
                        className={`rounded-2xl px-3 py-2 text-sm shadow-sm ${isUser
                            ? "bg-slate-50 text-slate-900"
                            : "bg-slate-900/80 text-slate-50 border border-slate-800/80"
                          }`}
                      >
                        <p className="whitespace-pre-wrap leading-relaxed">
                          {msg.content}
                        </p>
                      </div>
                    </div>
                  </div>
                );
              })}

              {/* Typing indicator */}
              {loading && (
                <div className="flex w-full justify-start">
                  <div className="flex max-w-[80%] items-end gap-2">
                    <Avatar className="h-7 w-7 bg-slate-800/80 text-[11px]">
                      <AvatarFallback>AI</AvatarFallback>
                    </Avatar>
                    <div className="rounded-2xl bg-slate-900/80 px-3 py-2 text-xs text-slate-300 shadow-sm">
                      <div className="flex items-center gap-1">
                        <span className="h-1.5 w-1.5 animate-bounce rounded-full bg-slate-400" />
                        <span className="h-1.5 w-1.5 animate-bounce rounded-full bg-slate-500 [animation-delay:0.15s]" />
                        <span className="h-1.5 w-1.5 animate-bounce rounded-full bg-slate-600 [animation-delay:0.3s]" />
                      </div>
                    </div>
                  </div>
                </div>
              )}

              <div className="h-4" />
            </div>
          </ScrollArea>
        </div>

        {/* Input area */}
        <div className="border-t border-slate-800 bg-slate-950/95 px-4 py-3">
          <div className="mx-auto flex w-full max-w-3xl flex-col gap-2">
            <div className="flex items-center gap-2">
              <Input
                value={input}
                placeholder="Type your question and press Enter to send..."
                onChange={(e) => setInput(e.target.value)}
                onKeyDown={handleKeyDown}
                disabled={loading}
              />
              <Button
                onClick={sendMessage}
                disabled={loading || !input.trim()}
                className="shrink-0"
              >
                {loading ? "Sending..." : "Send"}
              </Button>
            </div>
            <p className="text-[11px] text-slate-500">
              Press <span className="font-semibold text-slate-300">Enter</span> to
              send.
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}