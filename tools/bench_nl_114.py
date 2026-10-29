import hashlib, json, os, statistics, sys, time, urllib.request

arm = sys.argv[1]
url = 'http://127.0.0.1:18888/v1/chat/completions'
prompts = [
 ('nl', 'Leg in helder Nederlands uit hoe een warmtepomp werkt. Geef voorbeelden en bespreek voor- en nadelen.'),
 ('nl', 'Schrijf een praktische handleiding voor het aanleggen van een kleine moestuin, van bodemvoorbereiding tot oogst.'),
 ('nl', 'Beschrijf hoe een gemeente wateroverlast kan verminderen. Geef zes maatregelen met uitleg en beperkingen.'),
 ('nl', 'Vertel een origineel verhaal over een machinist die tijdens een storm een onverwacht probleem oplost.'),
 ('nl', 'Leg het verschil uit tussen vermogen en energie, met berekeningen voor huishoudelijke apparaten.'),
 ('nl', 'Schrijf een beknopt advies aan een ondernemer over het veilig maken van industriële netwerken.'),
 ('en', 'Explain how a heat pump works, including practical benefits and limitations.'),
 ('en', 'Describe six methods a city can use to reduce flooding and their tradeoffs.'),
 ('code', 'Write a Python function that calculates a moving average, with edge cases and tests.'),
 ('code', 'Write a Python function to merge overlapping intervals. Explain complexity and provide tests.'),
]

def request(prompt, draft=True):
 body = {'model':'Qwen3.8-Flash-Next','messages':[{'role':'user','content':prompt}], 'max_tokens':512,'temperature':0,'seed':4321,'stream':True,'stream_options':{'include_usage':True},'chat_template_kwargs':{'enable_thinking':False}}
 if not draft: body['draft']=False
 req=urllib.request.Request(url,json.dumps(body).encode(),{'Content-Type':'application/json'})
 start=time.monotonic(); first=None; text=''; tf={}; usage={}; finish=None
 with urllib.request.urlopen(req,timeout=600) as f:
  for line in f:
   line=line.decode().strip()
   if not line.startswith('data:') or line.endswith('[DONE]'): continue
   x=json.loads(line[5:]); choice=(x.get('choices') or [{}])[0]; delta=choice.get('delta') or {}; part=delta.get('content') or ''
   if part and first is None: first=time.monotonic()
   text+=part; tf=x.get('tensorfold') or tf; usage=x.get('usage') or usage; finish=choice.get('finish_reason') or finish
 return {'text_sha256':hashlib.sha256(text.encode()).hexdigest(),'text':text,'finish':finish,'ttft':None if first is None else first-start,'elapsed':time.monotonic()-start,'tensorfold':tf,'usage':usage}

out=[]
for i,(kind,prompt) in enumerate(prompts):
 for draft in ([True,False] if i in (0,1,6,8) else [True]):
  result=request(prompt,draft); result.update({'arm':arm,'prompt':i,'kind':kind,'draft_requested':draft}); out.append(result)
  with open('results.jsonl','a',encoding='utf8') as f: f.write(json.dumps(result,ensure_ascii=False)+'\n')
  print(json.dumps({k:v for k,v in result.items() if k != 'text'},ensure_ascii=False),flush=True)
