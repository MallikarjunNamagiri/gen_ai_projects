import { createClient } from '@supabase/supabase-js'

const supabaseUrl = import.meta.env.VITE_SUPABASE_URL || ''
const supabaseAnonKey = import.meta.env.VITE_SUPABASE_ANON_KEY || ''

// Validate environment variables
let supabase: ReturnType<typeof createClient>
// helper: ensure the url includes a valid scheme
function isValidUrl(url: string) {
  try {
    const u = new URL(url)
    return u.protocol === 'http:' || u.protocol === 'https:'
  } catch (e) {
    return false
  }
}

if (!supabaseUrl || !supabaseAnonKey || !isValidUrl(supabaseUrl)) {
  const errorMsg = `
⚠️  Missing Supabase environment variables!

Please create a .env file in the project root with:
VITE_SUPABASE_URL=your_supabase_project_url
VITE_SUPABASE_ANON_KEY=your_supabase_anon_key

You can find these values in your Supabase project settings:
https://app.supabase.com/project/_/settings/api
`
  console.error(errorMsg)

  // Use placeholder values that won't crash but will fail gracefully when used
  // This allows the app to load and show a proper error message
  const placeholderUrl = 'https://placeholder.supabase.co'
  const placeholderKey = 'placeholder-key'

  supabase = createClient(placeholderUrl, placeholderKey)
} else {
  supabase = createClient(supabaseUrl, supabaseAnonKey)
}

export { supabase }
