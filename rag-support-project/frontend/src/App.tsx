import { AuthProvider } from './hooks/useAuth'
import Chat from './components/Chat'

function App() {
  return (
    <AuthProvider>
      <div className="min-h-screen bg-slate-950 text-slate-50">
        <Chat />
      </div>
    </AuthProvider>
  )
}

export default App

