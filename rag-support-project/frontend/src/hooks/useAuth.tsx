import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useState,
  ReactNode,
} from "react";
import { supabase } from "../lib/supabase";

type SupabaseSession = any; // avoid hard dependency on @supabase/supabase-js types

interface AuthContextValue {
  session: SupabaseSession | null;
  user: any | null;
  loading: boolean;
  error: string | null;
  login: (email: string, password: string) => Promise<void>;
  signup: (email: string, password: string) => Promise<void>;
  logout: () => Promise<void>;
}

const AuthContext = createContext<AuthContextValue | undefined>(undefined);

const SESSION_STORAGE_KEY = "sb-session";

interface AuthProviderProps {
  children: ReactNode;
}

export function AuthProvider({ children }: AuthProviderProps) {
  const [session, setSession] = useState<SupabaseSession | null>(null);
  const [user, setUser] = useState<any | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  // Persist session in localStorage whenever it changes
  const persistSession = (nextSession: SupabaseSession | null) => {
    if (typeof window === "undefined") return;
    if (nextSession) {
      window.localStorage.setItem(SESSION_STORAGE_KEY, JSON.stringify(nextSession));
    } else {
      window.localStorage.removeItem(SESSION_STORAGE_KEY);
    }
  };

  useEffect(() => {
    let isMounted = true;

    const init = async () => {
      try {
        setLoading(true);

        // 1) Try to get the active session from Supabase
        const { data, error } = await supabase.auth.getSession();
        if (!isMounted) return;

        if (error) {
          setError(error.message);
        }

        const currentSession = data?.session ?? null;
        setSession(currentSession);
        setUser(currentSession?.user ?? null);
        persistSession(currentSession);
      } catch (e: any) {
        if (!isMounted) return;
        setError(e?.message ?? "Failed to initialize auth session");
      } finally {
        if (isMounted) setLoading(false);
      }
    };

    init();

    // 2) Subscribe to auth state changes
    const { data: subscription } = supabase.auth.onAuthStateChange(
      (_event, nextSession) => {
        if (!isMounted) return;
        setSession(nextSession);
        setUser(nextSession?.user ?? null);
        persistSession(nextSession);
      }
    );

    return () => {
      isMounted = false;
      subscription?.subscription?.unsubscribe();
    };
  }, []);

  const login = useCallback(
    async (email: string, password: string) => {
      setError(null);
      const { data, error } = await supabase.auth.signInWithPassword({
        email,
        password,
      });
      if (error) {
        setError(error.message);
        throw error;
      }
      setSession(data.session);
      setUser(data.session?.user ?? null);
      persistSession(data.session);
    },
    []
  );

  const signup = useCallback(
    async (email: string, password: string) => {
      setError(null);
      const { data, error } = await supabase.auth.signUp({
        email,
        password,
      });
      if (error) {
        setError(error.message);
        throw error;
      }
      // Depending on your Supabase settings, a session may or may not be created immediately
      setSession(data.session ?? null);
      setUser(data.session?.user ?? null);
      persistSession(data.session ?? null);
    },
    []
  );

  const logout = useCallback(async () => {
    setError(null);
    const { error } = await supabase.auth.signOut();
    if (error) {
      setError(error.message);
      throw error;
    }
    setSession(null);
    setUser(null);
    persistSession(null);
  }, []);

  const value: AuthContextValue = {
    session,
    user,
    loading,
    error,
    login,
    signup,
    logout,
  };

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
}

export function useAuth(): AuthContextValue {
  const ctx = useContext(AuthContext);
  if (!ctx) {
    throw new Error("useAuth must be used within an AuthProvider");
  }
  return ctx;
}

interface ProtectedRouteProps {
  children: ReactNode;
  redirectTo?: string;
  fallback?: ReactNode;
}

/**
 * Simple client-side protected route.
 * Redirects to /login (or custom path) when unauthenticated.
 */
export function ProtectedRoute({
  children,
  redirectTo = "/login",
  fallback,
}: ProtectedRouteProps) {
  const { session, loading } = useAuth();

  useEffect(() => {
    if (!loading && !session && typeof window !== "undefined") {
      window.location.href = redirectTo;
    }
  }, [loading, session, redirectTo]);

  if (loading || !session) {
    // While checking auth / redirecting, show a simple fallback
    return (
      <>
        {fallback ?? (
          <div className="flex h-screen items-center justify-center text-sm text-slate-400">
            Checking authentication...
          </div>
        )}
      </>
    );
  }

  return <>{children}</>;
}


