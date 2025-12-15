import { supabase } from "../lib/supabase";

interface StreamOptions {
  onToken?: (token: string) => void;
  onStart?: () => void;
  onDone?: (fullText: string) => void;
  onError?: (error: Error) => void;
}

/**
 * Sends a prompt to the /api/chat endpoint and streams the response.
 *
 * - Retrieves the Supabase auth token
 * - POSTs { query } to /api/chat with bearer token
 * - Reads the streaming response and emits partial tokens
 */
export async function sendMessage(
  query: string,
  options: StreamOptions = {}
): Promise<void> {
  const { onToken, onStart, onDone, onError } = options;

  const trimmed = query.trim();
  if (!trimmed) return;

  try {
    onStart?.();

    const token = (await supabase.auth.getSession()).data.session?.access_token;

    const response = await fetch("/api/chat", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        ...(token ? { Authorization: `Bearer ${token}` } : {}),
      },
      body: JSON.stringify({ query: trimmed }),
    });

    if (!response.body) {
      throw new Error("Empty response body from /api/chat");
    }

    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let fullText = "";

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      const chunk = decoder.decode(value);
      // If the backend uses "data: " SSE-style chunks, strip the prefix
      const tokenText = chunk.replace(/^data:\s*/g, "");

      fullText += tokenText;
      onToken?.(tokenText);
    }

    onDone?.(fullText);
  } catch (error: any) {
    const err = error instanceof Error ? error : new Error(String(error));
    onError?.(err);
    throw err;
  }
}


