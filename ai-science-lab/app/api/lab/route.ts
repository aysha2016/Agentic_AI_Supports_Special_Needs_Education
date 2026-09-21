import { NextResponse } from 'next/server'
export async function POST(req:Request){
 const {query,grade='Grade 7',subject='Science'}=await req.json()
 if(!process.env.OPENAI_API_KEY)return NextResponse.json({error:'OPENAI_API_KEY is not configured.'},{status:400})
 const prompt=`You are an educational laboratory designer. Create a safe curriculum-friendly interactive ${subject} laboratory for ${grade} from: "${query}". Return ONLY JSON with keys title,subject,grade,objective,explanation,variables(array of {name,min,max,value,unit}),formula,steps(array),questions(array),simulationType. Avoid unsafe or dangerous procedures.`
 const r=await fetch('https://api.openai.com/v1/responses',{method:'POST',headers:{'Content-Type':'application/json','Authorization':`Bearer ${process.env.OPENAI_API_KEY}`},body:JSON.stringify({model:'gpt-5.6',input:prompt})})
 const d=await r.json()
 if(!r.ok)return NextResponse.json({error:d},{status:r.status})
 const t=d.output?.map((x:any)=>x.content?.map((c:any)=>c.text).join('')).join('')||''
 try{return NextResponse.json(JSON.parse(t))}catch{return NextResponse.json({title:query,subject,grade,objective:'Explore the concept safely.',explanation:t,variables:[],formula:'',steps:[],questions:[],simulationType:'generic'})}
}