// Three reusable displays, driven by deterministic per-step models.
// Patch SVG attributes and text in place; the physical console never rebuilds.
const motionPlayers=new Map();
const motionReduced=matchMedia('(prefers-reduced-motion: reduce)');
function patchMotion(target,source){
 const old=[...target.childNodes],fresh=[...source.childNodes];
 fresh.forEach((node,i)=>{
  const current=old[i];
  if(!current){target.append(node.cloneNode(true));return;}
  if(current.nodeType!==node.nodeType||current.nodeName!==node.nodeName){current.replaceWith(node.cloneNode(true));return;}
  if(node.nodeType===3){if(current.nodeValue!==node.nodeValue)current.nodeValue=node.nodeValue;return;}
  if(node.nodeType!==1)return;
  for(const a of [...current.attributes])if(!node.hasAttribute(a.name))current.removeAttribute(a.name);
  for(const a of [...node.attributes])if(current.getAttribute(a.name)!==a.value)current.setAttribute(a.name,a.value);
  patchMotion(current,node);
 });
 old.slice(fresh.length).forEach(node=>node.remove());
}
function createMotionPlayer(root){
 const body=root.querySelector('[data-motion-body]'),svg=root.querySelector('.iso-svg'),screen=root.querySelector('.motion-screen');
 const cursor=root.querySelector('[data-motion-cursor]'),offset=root.dataset.offset.split(',').map(Number);
 const play=root.querySelector('[data-motion-play]'),seek=root.querySelector('[data-motion-seek]');
 const phase=root.querySelector('[data-motion-phase]'),readout=root.querySelector('[data-motion-readout]'),caption=root.querySelector('[data-motion-caption]'),time=root.querySelector('[data-motion-time]');
 const source=document.createElementNS('http://www.w3.org/2000/svg','g');
 let stream=+root.dataset.motionStream,step=0,p=motionReduced.matches?.5:0,playing=!motionReduced.matches,visible=false,last=0,frame=0,drawTime=0,flat=false;
 function draw(){
  const model=(stream===0?datasetMotion:poseMotion)(step,p);
  source.innerHTML=model.body;patchMotion(body,source);
  cursor.setAttribute('transform',`translate(${offset[0]*p} ${offset[1]*p})`);
  root.querySelector('[data-readout]').textContent=`0${step+1} / 06`;
  phase.textContent=model.phase;readout.textContent=model.readout;caption.textContent=model.caption;
  seek.value=String(Math.round(p*1000));seek.setAttribute('aria-valuetext',`${Math.round(p*100)} percent: ${model.phase}`);
  time.textContent=`${(p*12).toFixed(1).padStart(4,'0')} / 12s`;
  root.dataset.progress=p.toFixed(4);root.dataset.motionStep=String(step);root.dataset.motionStream=String(stream);
  play.textContent=motionReduced.matches?'Next frame':playing?'Pause':p>=1?'Replay':'Play';play.setAttribute('aria-pressed',String(playing));
  svg.setAttribute('aria-label',`${streams[stream].stages[step].title}. ${model.phase}. ${model.readout}. Illustrative animation.`);
 }
 function run(){if(!frame&&playing&&visible&&!document.hidden&&!motionReduced.matches){last=0;frame=requestAnimationFrame(tick);}}
 function tick(now){
  frame=0;if(!playing||!visible||document.hidden||motionReduced.matches){last=0;return;}
  const dt=last?Math.min((now-last)/1000,.1):0;last=now;p=Math.min(1,p+dt/12);
  if(now-drawTime>=40||p===1){if(p===1)playing=false;draw();drawTime=now;}
  if(playing)frame=requestAnimationFrame(tick);
 }
 function halt(){cancelAnimationFrame(frame);frame=0;last=0;}
 play.addEventListener('click',()=>{
  if(motionReduced.matches){p=p>=1?0:Math.min(1,p+.1);playing=false;}
  else{playing=!playing;if(playing&&p>=1)p=0;}
  halt();draw();run();
 });
 root.querySelector('[data-motion-replay]').addEventListener('click',()=>{p=0;playing=!motionReduced.matches;halt();draw();run();});
 seek.addEventListener('input',()=>{p=Number(seek.value)/1000;playing=false;halt();draw();});
 const flatButton=root.querySelector('[data-motion-flat]');
 function setFlat(value){flat=value;root.classList.toggle('motion-flat',flat);svg.setAttribute('viewBox',flat?'0 0 480 260':'0 0 640 400');screen.setAttribute('transform',flat?'':`matrix(${screen.dataset.matrix})`);flatButton.textContent=flat?'Console view':'Flat view';flatButton.setAttribute('aria-pressed',String(flat));}
 flatButton.addEventListener('click',()=>setFlat(!flat));
 setFlat(matchMedia('(max-width:800px)').matches||root.id==='deep-motion');
 new IntersectionObserver(entries=>{visible=entries.at(-1).isIntersecting;if(visible)run();else halt();},{threshold:.05}).observe(root);
 document.addEventListener('visibilitychange',()=>{if(document.hidden)halt();else run();});
 motionReduced.addEventListener('change',()=>{if(motionReduced.matches){playing=false;halt();}draw();});
 const api={root,setStage(i,j){if(i===stream&&j===step)return;stream=i;step=j;p=motionReduced.matches?.5:0;playing=!motionReduced.matches;halt();draw();run();},getState(){return {stream,step,p,playing,visible,flat};}};
 draw();motionPlayers.set(root.id,api);return api;
}
document.querySelectorAll('.motion-player').forEach(createMotionPlayer);
