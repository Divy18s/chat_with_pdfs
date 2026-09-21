import { useState } from 'react';
const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8000/api';
export default function Home() {
  const [q, setQ] = useState(''); const [a, setA] = useState('');
  async function ask() {
    const r = await fetch(`${API}/chat`, {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({question: q})});
    const j = await r.json(); setA(j.answer || JSON.stringify(j));
  }
  return (<main style={{padding:24, fontFamily:'sans-serif'}}>
    <h1>Chat with PDFs (Next.js + Django RAG)</h1>
    <p style={{fontSize:12}}>Backend: {API} — full ChatGPT split UI lives in Django template at / for local test (no Node needed).</p>
    <input value={q} onChange={e=>setQ(e.target.value)} placeholder="Ask…" style={{width:'60%', padding:8}}/>
    <button onClick={ask} style={{marginLeft:8}}>Ask</button>
    <pre style={{whiteSpace:'pre-wrap', marginTop:16}}>{a}</pre>
  </main>);
}
