import { NextResponse } from 'next/server'
export async function POST(req:Request){
 const {topic,grade='Grade 7'}=await req.json()
 if(!process.env.OPENAI_API_KEY)return NextResponse.json({error:'OPENAI_API_KEY is not configured.'},{status:400})
 const prompt=`Create a short ${grade} science quiz about "${topic}". Return ONLY JSON: {"questions":[{"question":"...","options":["A","B","C","D"],"answer":"A","explanation":"..."}]} with 5 questions.`
 const r=await fetch('https://api.openai.com/v1/responses',{method:'POST',headers:{'Content-Type':'application/json','Authorization':`Bearer ${process.env.OPENAI_API_KEY}`},body:JSON.stringify({model:'gpt-5.6',input:prompt})})
 const d=await r.json(), t=d.output?.map((x:any)=>x.content?.map((c:any)=>c.text).join('')).join('')||''
 try{return NextResponse.json(JSON.parse(t))}catch{return NextResponse.json({questions:[]})}
}